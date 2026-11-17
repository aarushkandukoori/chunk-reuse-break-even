import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import costmodel as cm  # noqa: E402


def synthetic_fit(const=0.01, per_tok=1e-4, attn=1e-8, restore_per_tok=1e-6):
    return {
        "forward_coef": [const, per_tok, attn, 0.0],
        "restore_mem": [0.0, restore_per_tok],
        "restore_file": [0.0, 10 * restore_per_tok],
    }


def test_fit_forward_recovers_known_coefficients():
    true = [0.005, 2e-4, 3e-8, 1e-8]
    rows = []
    for k in [8, 32, 128, 512, 2048]:
        for ctx in [0, 256, 1024, 4096]:
            rows.append({"k": k, "ctx": ctx, "sec": float(np.dot(true, cm._feat(k, ctx)))})
    got = cm.fit_forward(rows)
    assert np.allclose(got, true, rtol=1e-6, atol=1e-12)


def test_fit_line():
    a, b = cm.fit_line([1, 2, 3], [3, 5, 7])
    assert a == pytest.approx(1) and b == pytest.approx(2)


def test_forward_zero_tokens_costs_nothing():
    assert cm.forward_time([1, 1, 1, 1], 0, 100) == 0.0
    assert cm.restore_time([5, 1], 0) == 0.0


def test_recompute_tokens_caps_at_chunk_tokens():
    assert cm.recompute_tokens(2, 100, 0.1, 0) == 20
    assert cm.recompute_tokens(2, 100, 5.0, 0) == 200
    assert cm.recompute_tokens(1, 100, 0.0, 16) == 16


def test_free_restore_and_no_recompute_beats_full_prefill():
    fit = synthetic_fit(restore_per_tok=0.0)
    assert cm.speedup(fit, M=4, c=256, q=32, rho=0.0) > 1


def test_more_recompute_is_never_faster():
    fit = synthetic_fit()
    times = [cm.reuse_time(fit, 4, 256, 32, r) for r in [0, 0.1, 0.3, 0.6, 1.0]]
    assert all(a <= b for a, b in zip(times, times[1:]))


def test_full_recompute_with_overheads_loses():
    # rho = 1 recomputes every chunk token, and the reuse path also pays restore + a separate query pass
    fit = synthetic_fit()
    assert cm.speedup(fit, 4, 256, 32, rho=1.0) < 1


def test_break_even_rho_is_where_times_match():
    fit = synthetic_fit()
    r = cm.break_even_rho(fit, 4, 512, 32)
    assert 0 < r < 1
    full = cm.full_time(fit, 4, 512, 32)
    assert cm.reuse_time(fit, 4, 512, 32, r) <= full * (1 + 1e-3)
    assert cm.reuse_time(fit, 4, 512, 32, min(1.0, r + 0.01)) > full


def test_break_even_edges():
    slow_restore = synthetic_fit(restore_per_tok=1.0)  # restore far slower than compute
    assert cm.break_even_rho(slow_restore, 1, 256, 16) == 0.0


def test_fixed_call_overhead_hurts_short_contexts_more():
    fit = synthetic_fit(const=0.02)
    short = cm.speedup(fit, 1, 64, 16, rho=0.1)
    long = cm.speedup(fit, 8, 512, 16, rho=0.1)
    assert short < long


def test_daric_default_rho():
    assert cm.daric_default_rho(4, 256) == pytest.approx(0.125)
    assert cm.daric_default_rho(4, 16) == 1.0


def test_merged_query_pass_saves_one_call_overhead():
    fit = synthetic_fit(const=0.02)
    sep = cm.reuse_time(fit, 2, 128, 32, 0.1)
    mer = cm.reuse_time(fit, 2, 128, 32, 0.1, merged=True)
    assert sep - mer == pytest.approx(cm.forward_time(fit["forward_coef"], 32, 0))
