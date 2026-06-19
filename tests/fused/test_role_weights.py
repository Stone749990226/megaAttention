"""role 权重只影响调度/性能, 不影响 FA 数值与调度 invariant (tp=1, 单序列)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_fused_fa_path import run_case, _check   # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs H200")


@pytest.mark.parametrize("w", [(4, 1, 1), (8, 2, 1), (6, 1, 0)])
def test_role_weights_preserve_correctness(w):
    r = run_case([512], H_local=4, hidden=512, num_ctas=8,
                 w_fa=w[0], w_oproj=w[1], w_ar=w[2])
    assert _check(f"w={w}", r), f"invariants violated for weights {w}"
