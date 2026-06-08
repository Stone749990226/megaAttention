#!/bin/bash
# =============================================================================
# ncu profiling for the fused O_proj GEMM + one-shot NVLS AllReduce SM90 kernel
# (megaattn_oproj_ar_sm90.py :: OProjARFusedKernelSM90).
#
# Modeled on the multi-GPU warp-specialization+NVSHMEM reference script:
#   - one ncu instance PER RANK, all launched concurrently;
#   - the instances are kept in lockstep (--communicator tcp +
#     --lockstep-kernel-launch + --communicator-tcp-num-peers N) so every rank
#     steps through the SAME kernel launch together. This is MANDATORY for this
#     kernel: it does cross-rank multimem AllReduce with per-tile spin-lock
#     barriers, so if ncu paused/replayed one rank while the others advanced it
#     would deadlock.
#   - --replay-mode application (NOT kernel replay): a collective kernel's
#     cross-rank symmetric-memory state cannot be save/restored per kernel, so
#     the whole app is re-run for each pass, with the 8 ranks kept in lockstep.
#
# Sections: every event-replay section ncu bundles in `--set full`
# (Compute/Memory+charts+tables/Scheduler/Warp/Occupancy/SOL + all Rooflines/
# InstructionStats/SourceCounters/NVLink tables+topology) PLUS C2CLink and base
# Nvlink. NVLink/C2C matter here because the AllReduce rides NVLink/NVSwitch.
# PM SAMPLING (incl. warp PM sampling) IS OMITTED ON PURPOSE: it is multi-group /
# multi-pass and, on top of the ~40+ NCCL rendezvous that full application replay
# needs, made the run hang on an app relaunch. Dropping it ~halves the pass count
# and removes the fragile passes while keeping every event-based metric, incl.
# instruction-level SASS (which app-range replay cannot give for a JIT kernel).
#
# Each pass re-runs bench.py (python import + NCCL init + CuTe-DSL JIT compile),
# so a full-section run takes a while. ncu profiles the FIRST fused-kernel launch
# (the correctness-check launch) and --kill yes terminates the app right after.
#
# Usage:
#   ./ncu_oproj_ar.sh [-n N] [-o OUTDIR] [-p MASTER_PORT] [extra bench.py args]
# Examples:
#   ./ncu_oproj_ar.sh                       # 8 ranks, default out dir
#   ./ncu_oproj_ar.sh -n 8 -o /tmp/ncu_run  # custom output dir
# =============================================================================
set -uo pipefail

PYTHON=/usr/bin/python                       # cutlass-dsl lives here (NOT vllm venv)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

NUM_PROCESSES=8
OUTPUT_DIR="/myworkspace/log/ncu_oproj_ar_${TS}"
MASTER_PORT=29591
KERNEL_REGEX='OProjARFusedKernelSM90'        # stable substring of the JIT symbol
SECTION_SET=1                                # 1 = core compute/occupancy; 2 = comm / data-movement
bench_args=()

while (( "$#" )); do
  case "$1" in
    -n|--num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    -o|--output)        OUTPUT_DIR="$2";    shift 2 ;;
    -p|--port)          MASTER_PORT="$2";   shift 2 ;;
    --section|--sections|-s) SECTION_SET="$2"; shift 2 ;;   # 1=core, 2=comm/data-movement
    -h|--help)
      echo "Usage: $0 [-n N] [-o OUTDIR] [-p MASTER_PORT] [--section 1|2] [extra bench.py args]"
      echo "  --section 1 : core compute/scheduler/warp/occupancy (default)"
      echo "  --section 2 : communication / data-movement (NVLink, C2C, memory tables, SASS, roofline)"
      exit 0 ;;
    *) bench_args+=("$1"); shift ;;
  esac
done

mkdir -p "$OUTPUT_DIR" /myworkspace/log
RUN_LOG="/myworkspace/log/ncu_oproj_ar_${TS}.log"

# Profiling defaults: ncu kills the app after the 1st matching kernel, so keep
# the bench workload tiny (the profiled launch is bench.py's correctness check).
if (( ${#bench_args[@]} == 0 )); then
  bench_args=(--iters 1 --warmup 0)
fi

# Distributed env that bench.py / dist.init_process_group("nccl") need
# (torchrun normally sets these; we launch one process per rank by hand).
export MASTER_ADDR=127.0.0.1
export MASTER_PORT
export WORLD_SIZE="$NUM_PROCESSES"
export NCCL_NVLS_ENABLE=1
export NCCL_ALGO=NVLS
export CUTE_DSL_LINEINFO=1                    # best-effort source line info for SourceCounters

echo "Python        : $PYTHON"
echo "Num ranks     : $NUM_PROCESSES"
echo "Output dir    : $OUTPUT_DIR"
echo "Master port   : $MASTER_PORT"
echo "Kernel regex  : $KERNEL_REGEX"
echo "Section set   : $SECTION_SET  (1=core compute, 2=comm/data-movement)"
echo "bench.py args : ${bench_args[*]}"
echo "Run log       : $RUN_LOG"
echo "================================================================"

# ---- section set ------------------------------------------------------------
# WHY THIS IS A *CORE* SET, NOT `full`:
# Application replay re-runs the whole 8-rank job once per metric pass. Two runs
# at near-`full` coverage (~38 passes) BOTH hung at pass ~38 on an app relaunch
# -- a deterministic resource wall (SIGKILLed ranks leak /dev/shm cuda.shm.*
# segments; after ~38 relaunches the next NCCL rendezvous can't complete). So we
# keep only the CORE bottleneck sections, which fit in ~15 passes -- safely under
# the wall -- to reliably get a first report out. The pass-heavy extras (the five
# Roofline charts, Memory chart/tables, SourceCounters/SASS, NVLink/C2C/NUMA) are
# parked in SECTIONS_EXTRA below; add them back (or batch them into a 2nd report)
# once the core report is in hand.
# --- SET 1: core (compute / scheduling / occupancy) --- the "is the kernel
# efficient on-SM" view. Fits in ~15 passes; this is the one already validated.
SECTIONS1=(
  --section SpeedOfLight           # GPU SOL: compute vs memory throughput headline
  --section ComputeWorkloadAnalysis
  --section MemoryWorkloadAnalysis
  --section SchedulerStats
  --section WarpStateStats         # stall-reason breakdown (key for a comm-overlapped kernel)
  --section Occupancy
  --section InstructionStats
  --section LaunchStats
  --section WorkloadDistribution
)

# --- SET 2: communication / data-movement (GEMM<->AllReduce fusion view) ---
# NO OVERLAP with SET 1. This is the set to use for analyzing the compute/comm
# fusion: the AllReduce rides NVLink/NVSwitch (multimem ld_reduce), so the NVLink
# sections show achieved interconnect BW vs peak; the Memory chart/tables show the
# L2/DRAM/peer data-movement path of the reduce; SourceCounters pins the comm
# warp-group's multimem/red instructions (and their stalls) to SASS lines; the
# Rooflines show whether the GEMM is compute-bound enough to hide the comm.
# C2CLink is included for completeness (on H200/x86 there is no C2C link, so it
# will simply be empty -- harmless).
SECTIONS2=(
  --section Nvlink                 # NVLink throughput: the AllReduce transport
  --section Nvlink_Tables          # per-link achieved BW vs peak
  --section Nvlink_Topology        # link/switch topology
  --section C2CLink                # (empty on H200 -- kept for completeness)
  --section MemoryWorkloadAnalysis_Chart    # L1/L2/DRAM/peer data-movement diagram
  --section MemoryWorkloadAnalysis_Tables   # detailed L2/DRAM counters (reduce path)
  --section SourceCounters         # SASS/source-level: locate multimem/red hotspots
  --section SpeedOfLight_RooflineChart            # compute-vs-memory roofline (fusion headroom)
  --section SpeedOfLight_HierarchicalTensorRooflineChart  # tensor-core roofline (the GEMM)
)

# select the requested set
case "$SECTION_SET" in
  1) SECTIONS=( "${SECTIONS1[@]}" ) ;;
  2) SECTIONS=( "${SECTIONS2[@]}" ) ;;
  *) echo "ERROR: --section must be 1 or 2 (got '$SECTION_SET')"; exit 2 ;;
esac

ncu_args=(
  --config-file off
  --force-overwrite
  --kernel-name "regex:${KERNEL_REGEX}"
  --kernel-name-base function
  "${SECTIONS[@]}"
  --import-source yes
  --rule LocalMemoryUsage
  --replay-mode application
  --app-replay-buffer memory
  --launch-skip 0
  --launch-count 1
  --lockstep-kernel-launch
  --communicator tcp
  --communicator-tcp-num-peers "$NUM_PROCESSES"
  --clock-control none
  --kill yes
)

{
echo "[ncu_oproj_ar] start $(date -Is)"
echo "[ncu_oproj_ar] ncu args: ${ncu_args[*]}"

# Optional warm-up / sanity run (validates the 8-rank job + shapes before the
# long profiling). NOTE: each application-replay pass re-runs the whole 8-rank
# job, so the dominant per-pass cost is torch + NCCL init (~15-20s), not JIT;
# this warm-up does not remove that. Comment out to skip.
echo "[ncu_oproj_ar] warm-up sanity run ..."
warm_pids=()
for (( i=0; i<NUM_PROCESSES; ++i )); do
  RANK=$i LOCAL_RANK=$i "$PYTHON" "$SCRIPT_DIR/bench.py" "${bench_args[@]}" \
      >"$OUTPUT_DIR/warmup.rank$i.out" 2>&1 &
  warm_pids+=($!)
done
warm_fail=0
for pid in "${warm_pids[@]}"; do wait "$pid" || warm_fail=1; done
if (( warm_fail )); then
  echo "[ncu_oproj_ar] WARN: warm-up run reported a non-zero exit (see warmup.rank*.out)"
fi
sleep 2

echo "[ncu_oproj_ar] launching ${NUM_PROCESSES} lockstep ncu instances ..."
pids=()
# NOTE: all GPUs stay visible to every process (do NOT set CUDA_VISIBLE_DEVICES) --
# the kernel's symmetric-memory multicast + NVLS AllReduce span all ranks' GPUs.
# Each rank picks its own device via LOCAL_RANK (bench.py: torch.cuda.set_device(lr)).
for (( i=0; i<NUM_PROCESSES; ++i )); do
  RANK=$i LOCAL_RANK=$i \
    ncu "${ncu_args[@]}" -o "$OUTPUT_DIR/oproj_ar.s${SECTION_SET}.rank$i" \
        "$PYTHON" "$SCRIPT_DIR/bench.py" "${bench_args[@]}" \
        >"$OUTPUT_DIR/ncu.rank$i.out" 2>&1 &
  pids+=($!)
done

echo "[ncu_oproj_ar] waiting for ${NUM_PROCESSES} instances ..."

# --- progress heartbeat + stall watchdog -------------------------------------
# App replay re-runs the whole 8-rank job per pass (~15-20s NCCL init each), and
# per-pass progress only lands in ncu.rank*.out -- so heartbeat the current pass
# into the main log, and abort if NO pass advances for STALL_LIMIT seconds
# (catches a deadlock in minutes instead of waiting indefinitely).
STALL_LIMIT=${STALL_LIMIT:-300}            # seconds with no forward progress => abort
rank0_out="$OUTPUT_DIR/ncu.rank0.out"
last_pass=-1; last_change=$(date +%s)
rc=0
while :; do
  # all done?
  still=0; for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && still=1; done
  cur=$(grep -c 'replay pass' "$rank0_out" 2>/dev/null); cur=${cur:-0}
  ndone=$(ls "$OUTPUT_DIR"/oproj_ar.rank*.ncu-rep 2>/dev/null | wc -l)
  now=$(date +%s)
  if (( cur != last_pass )); then
    echo "[ncu_oproj_ar] $(date +%T) rank0 replay pass=$cur  reports_done=$ndone"
    last_pass=$cur; last_change=$now
  fi
  if (( still == 0 )); then break; fi
  if (( now - last_change > STALL_LIMIT )); then
    echo "[ncu_oproj_ar] STALL: no replay-pass progress for ${STALL_LIMIT}s -> aborting (likely deadlock)."
    pkill -9 -x ncu 2>/dev/null; pkill -9 -f "$SCRIPT_DIR/bench.py" 2>/dev/null
    rc=124; break
  fi
  sleep 15
done
for pid in "${pids[@]}"; do wait "$pid" || rc=$(( rc==0 ? 1 : rc )); done

echo "[ncu_oproj_ar] done $(date -Is) (aggregate rc=$rc)"
echo "[ncu_oproj_ar] reports:"
ls -la "$OUTPUT_DIR"/oproj_ar.s${SECTION_SET}.rank*.ncu-rep 2>/dev/null || echo "  (no .ncu-rep produced -- check ncu.rank*.out)"
echo "[ncu_oproj_ar] open with:  ncu-ui $OUTPUT_DIR/oproj_ar.s${SECTION_SET}.rank0.ncu-rep"
echo "[ncu_oproj_ar]   or CLI :  ncu --import $OUTPUT_DIR/oproj_ar.s${SECTION_SET}.rank0.ncu-rep --page details | less"
exit $rc
} 2>&1 | tee "$RUN_LOG"

exit "${PIPESTATUS[0]}"
