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
SECTION_SET=1                                # 1=core; 2a=SASS; 2b=interconnect/mem; 2c=roofline
# Warm-launch profiling: skip this many (fully-initialised) launches before the
# one ncu profiles, so we measure a steady-state launch -- not the cold first one
# (cold L2 + symmetric-mem first-touch inflate long_scoreboard). bench.py fires
# exactly this many run_fused() warm-ups via --prof-warmup.
PROF_WARMUP=${PROF_WARMUP:-4}
# Clock control: MUST stay 'none' for this lockstep collective. Locking clocks
# ('base') makes ncu insert a per-pass clock-settle step whose retry count varies
# PER RANK -> ranks then disagree on the application-replay pass count, the first
# rank to finish exits, and the remaining ranks deadlock in the cross-rank
# AllReduce spin-lock waiting on the departed peer (observed: with 'base' only
# 1 of 8 ranks produced a report; the other 7 hung at the final pass). 'none'
# keeps every rank on the same pass count, so the lockstep holds. The cost is the
# "unmodified GPU clocks" inconsistency warning -- acceptable; do NOT switch to
# 'base' to silence it. (Override only if you fully understand the above.)
CLOCK_CONTROL=${CLOCK_CONTROL:-none}
bench_args=()

while (( "$#" )); do
  case "$1" in
    -n|--num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    -o|--output)        OUTPUT_DIR="$2";    shift 2 ;;
    -p|--port)          MASTER_PORT="$2";   shift 2 ;;
    --section|--sections|-s) SECTION_SET="$2"; shift 2 ;;   # 1 | 2a | 2b | 2c
    -h|--help)
      echo "Usage: $0 [-n N] [-o OUTDIR] [-p MASTER_PORT] [--section 1|2a|2b|2c] [extra bench.py args]"
      echo "  --section 1  : core compute/scheduler/warp/occupancy (default, ~15 passes)"
      echo "  --section 2a : SASS / source-level stall hotspots (SourceCounters)"
      echo "  --section 2b : interconnect + data-movement (NVLink, C2C, memory chart/tables)"
      echo "  --section 2c : rooflines (compute headroom)"
      echo "  env: PROF_WARMUP=$PROF_WARMUP  CLOCK_CONTROL=$CLOCK_CONTROL  STALL_LIMIT=300"
      echo "  NOTE: 2a/2b/2c REPLACE the old monolithic --section 2, which exceeded the"
      echo "        ~38-pass application-replay wall and got SIGKILLed. Run them separately."
      exit 0 ;;
    *) bench_args+=("$1"); shift ;;
  esac
done

mkdir -p "$OUTPUT_DIR" /myworkspace/log
RUN_LOG="/myworkspace/log/ncu_oproj_ar_${TS}.log"

# Profiling defaults: ncu kills the app after the 1st matching kernel, so keep
# the bench workload tiny. We fire PROF_WARMUP warm launches first (then ncu skips
# them via --launch-skip) so the profiled launch is steady-state, not cold.
if (( ${#bench_args[@]} == 0 )); then
  bench_args=(--iters 1 --warmup 0)
fi
# Ensure bench.py gets the matching --prof-warmup unless the caller set one.
if ! printf '%s\n' "${bench_args[@]}" | grep -q -- '--prof-warmup'; then
  bench_args+=(--prof-warmup "$PROF_WARMUP")
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
echo "Section set   : $SECTION_SET  (1=core compute, 2a=SASS, 2b=interconnect/mem, 2c=roofline)"
echo "Prof warm-up  : $PROF_WARMUP (warm launches skipped before the profiled one)"
echo "Clock control : $CLOCK_CONTROL"
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
# the wall -- to reliably get a first report out. The pass-heavy extras (Roofline
# charts, Memory chart/tables, SourceCounters/SASS, NVLink/C2C) are split into
# SET 2a/2b/2c below, each run as a SEPARATE invocation so none crosses the wall.
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

# --- The old monolithic "SET 2" (NVLink+C2C+Memory tables+SourceCounters+2x
# Roofline) is split into 2a/2b/2c. WHY: as one ncu invocation it needed >38
# application-replay passes and got SIGKILLed at pass ~37 (the documented wall),
# so it produced a report for only ONE rank. Each sub-set below fits well under
# the wall, so all 8 ranks complete. Run them as separate invocations.
#
# --- SET 2a: SASS / source-level stall hotspots --- the single most useful
# comm view: pins the multimem-reduce / spin-lock / barrier stalls to source
# lines. Light on passes, so split out on its own to guarantee it completes.
SECTIONS2a=(
  --section SourceCounters         # SASS/source-level: locate multimem/red hotspots
)

# --- SET 2b: interconnect + data-movement --- the AllReduce rides NVLink/NVSwitch
# (multimem ld_reduce), so NVLink sections show achieved interconnect BW vs peak
# and the Memory chart/tables show the L2/DRAM/peer reduce path. C2CLink is empty
# on H200/x86 (no C2C link) -- kept for completeness, harmless.
SECTIONS2b=(
  --section Nvlink                 # NVLink throughput: the AllReduce transport
  --section Nvlink_Tables          # per-link achieved BW vs peak
  --section Nvlink_Topology        # link/switch topology
  --section MemoryWorkloadAnalysis_Chart    # L1/L2/DRAM/peer data-movement diagram
  --section MemoryWorkloadAnalysis_Tables   # detailed L2/DRAM counters (reduce path)
)

# --- SET 2c: rooflines --- whether the GEMM is compute-bound enough to hide the
# comm. Heaviest on passes (lots of derived metrics), lowest priority, isolated.
SECTIONS2c=(
  --section SpeedOfLight_RooflineChart            # compute-vs-memory roofline (fusion headroom)
  --section SpeedOfLight_HierarchicalTensorRooflineChart  # tensor-core roofline (the GEMM)
)

# select the requested set
case "$SECTION_SET" in
  1)  SECTIONS=( "${SECTIONS1[@]}" )  ;;
  2a) SECTIONS=( "${SECTIONS2a[@]}" ) ;;
  2b) SECTIONS=( "${SECTIONS2b[@]}" ) ;;
  2c) SECTIONS=( "${SECTIONS2c[@]}" ) ;;
  2)  echo "ERROR: --section 2 was split (it hit the ~38-pass wall). Use 2a, 2b, or 2c."; exit 2 ;;
  *)  echo "ERROR: --section must be 1 | 2a | 2b | 2c (got '$SECTION_SET')"; exit 2 ;;
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
  --launch-skip "$PROF_WARMUP"     # skip the warm-up launches -> profile a steady-state one
  --launch-count 1
  --lockstep-kernel-launch
  --communicator tcp
  --communicator-tcp-num-peers "$NUM_PROCESSES"
  --clock-control "$CLOCK_CONTROL" # base = reproducible; none = floating (set via env)
  --kill yes
)

{
echo "[ncu_oproj_ar] start $(date -Is)"
echo "[ncu_oproj_ar] ncu args: ${ncu_args[*]}"
echo "[ncu_oproj_ar] bench args: ${bench_args[*]}"

# --- reset the application-replay budget ------------------------------------
# A SIGKILLed prior run (e.g. one that hit the ~38-pass wall) leaves stale ncu /
# bench processes and leaked /dev/shm segments (cuda.shm.*, nccl-*, torch symm-mem
# files). Those lower the NCCL-rendezvous ceiling for THIS run, so a previously
# crashed run can make the next one die even earlier. Clear them up front.
echo "[ncu_oproj_ar] cleaning stale processes + /dev/shm segments ..."
pkill -9 -x ncu 2>/dev/null
pkill -9 -f "$SCRIPT_DIR/bench.py" 2>/dev/null
rm -f /dev/shm/cuda.shm.* /dev/shm/nccl-* /dev/shm/torch_* /dev/shm/sem.* 2>/dev/null
sleep 1

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
last_pass=-1; last_change=$(date +%s)
rc=0
while :; do
  # all done?
  still=0; for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && still=1; done
  # progress = the FURTHEST-advanced rank, not just rank0: rank0 may be a fast
  # "worker" that idles while another rank still drives passes (or vice versa).
  cur=0
  for (( i=0; i<NUM_PROCESSES; ++i )); do
    p=$(grep -c 'replay pass' "$OUTPUT_DIR/ncu.rank$i.out" 2>/dev/null); p=${p:-0}
    (( p > cur )) && cur=$p
  done
  ndone=$(ls "$OUTPUT_DIR"/oproj_ar.s${SECTION_SET}.rank*.ncu-rep 2>/dev/null | wc -l)
  now=$(date +%s)
  if (( cur != last_pass )); then
    echo "[ncu_oproj_ar] $(date +%T) max replay pass=$cur  reports_done=$ndone"
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

# --- per-rank kernel duration: which rank to actually analyse ----------------
# This is a collective with cross-rank spin-lock barriers, so per-rank duration
# diverges wildly under application replay: early-arriving ranks spin in the
# AllReduce barrier (huge duration, ~0% active = "waiter") while the last arrival
# does the real work (short duration = "worker"). The WORKER (shortest duration)
# is the representative report; the long ones are spin-wait artifacts, NOT latency.
echo "[ncu_oproj_ar] per-rank kernel duration (shortest = 'worker' = analyse this one):"
for f in "$OUTPUT_DIR"/oproj_ar.s${SECTION_SET}.rank*.ncu-rep; do
  [ -e "$f" ] || continue
  d=$(ncu --import "$f" --csv --page raw --metrics gpu__time_duration.sum 2>/dev/null \
        | tail -1 | tr -d '"' | awk -F, '{print $NF}')
  echo "    $(basename "$f")  duration=${d:-?}"
done

echo "[ncu_oproj_ar] open with:  ncu-ui $OUTPUT_DIR/oproj_ar.s${SECTION_SET}.rank0.ncu-rep"
echo "[ncu_oproj_ar]   or CLI :  ncu --import $OUTPUT_DIR/oproj_ar.s${SECTION_SET}.rank0.ncu-rep --page details | less"
exit $rc
} 2>&1 | tee "$RUN_LOG"

exit "${PIPESTATUS[0]}"
