"""Break-even model for reusing a moved chunk's KV cache plus recomputing some of its tokens.

Two ways to get the KV cache for a prompt  [prefix P][M chunks of c tokens][query q]:

  full prefill   : the prefix P is already cached (plain prefix caching), so run
                   M*c + q tokens through the model on top of P.
  chunk reuse    : restore the M stored chunk caches (lookup + copy/load + RoPE
                   re-encode), compute the query on its own, then recompute k
                   selected tokens against the whole context. k = rho * (M*c) + w,
                   where w is the "latest window" (DARIC always recomputes it).

All times come from a fitted forward-cost function T(k, ctx) (k new tokens on top
of ctx cached tokens) and a linear restore cost R(C), both fit to saved timings.
Costs only: this says nothing about output quality.
"""
import json
import math

import numpy as np

# ---------- fitting ----------

FEATURES = ("const", "k", "k*ctx", "k^2")


def _feat(k, ctx):
    return [1.0, k, k * ctx, k * k]


def fit_forward(rows):
    """rows: list of dicts with k, ctx, sec. Least squares on relative error."""
    X = np.array([_feat(r["k"], r["ctx"]) for r in rows], dtype=float)
    y = np.array([r["sec"] for r in rows], dtype=float)
    w = 1.0 / y
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return [float(c) for c in coef]


def fit_line(xs, ys):
    """y = a + b x, ordinary least squares."""
    b, a = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return [float(a), float(b)]


def fit_all(bench):
    rows = [{"k": p["L"], "ctx": 0, "sec": p["sec"]} for p in bench["prefill"]] + bench["forward"]
    fwd = fit_forward(rows)
    C = [r["C"] for r in bench["restore"]]
    restore_mem = fit_line(C, [r["mem_sec"] + r["reencode_sec"] + r["lookup_sec"] for r in bench["restore"]])
    restore_file = fit_line(C, [r["file_sec"] + r["reencode_sec"] + r["lookup_sec"] for r in bench["restore"]])
    pred = [forward_time(fwd, r["k"], r["ctx"]) for r in rows]
    errs = [abs(p - r["sec"]) / r["sec"] for p, r in zip(pred, rows)]
    return {
        "model": bench["model"],
        "forward_coef": fwd,
        "forward_features": list(FEATURES),
        "restore_mem": restore_mem,
        "restore_file": restore_file,
        "fit_median_rel_err": float(np.median(errs)),
        "fit_max_rel_err": float(np.max(errs)),
        "n_points": len(rows),
    }


# ---------- cost model ----------


def forward_time(coef, k, ctx):
    if k <= 0:
        return 0.0
    return float(np.dot(coef, _feat(k, ctx)))


def restore_time(line, C):
    if C <= 0:
        return 0.0
    return line[0] + line[1] * C


def full_time(fit, M, c, q, P=0):
    return forward_time(fit["forward_coef"], M * c + q, P)


def recompute_tokens(M, c, rho, w):
    """Tokens recomputed: a fraction rho of the chunk tokens plus a latest window w (capped)."""
    return min(M * c, int(math.ceil(rho * M * c))) + w


def reuse_time(fit, M, c, q, rho, P=0, w=None, source="mem", merged=False):
    """Chunk reuse path. w defaults to the query length (the latest window is the query).
    merged=False follows DARIC's two worker stages: the new query chunk gets its own
    cache-generation pass, then one recompute pass. merged=True folds the query into the
    recompute pass (one forward call), an optimistic variant for a phone port."""
    if w is None:
        w = q
    coef = fit["forward_coef"]
    line = fit["restore_mem"] if source == "mem" else fit["restore_file"]
    L = P + M * c + q
    t = M * restore_time(line, c)          # each chunk restored on its own
    if not merged:
        t += forward_time(coef, q, 0)       # query chunk computed on its own (cache-generation stage)
    k = recompute_tokens(M, c, rho, min(w, q))
    t += forward_time(coef, k, L - k)       # one recompute pass over the whole context
    return t


def speedup(fit, M, c, q, rho, P=0, source="mem", merged=False):
    return full_time(fit, M, c, q, P) / reuse_time(fit, M, c, q, rho, P, source=source, merged=merged)


def break_even_rho(fit, M, c, q, P=0, source="mem", tol=1e-4, merged=False):
    """Largest recompute fraction rho in [0, 1] at which reuse is still no slower than full prefill.
    Returns 0.0 if reuse loses even at rho = 0, and 1.0 if it wins even at rho = 1."""
    f = lambda r: full_time(fit, M, c, q, P) - reuse_time(fit, M, c, q, r, P, source=source, merged=merged)
    if f(0.0) < 0:
        return 0.0
    if f(1.0) >= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


def daric_default_rho(M, c, n_avg=32):
    """DARIC's default budget is at most M * n_avg recomputed tokens (n_avg = 32)."""
    return min(1.0, (M * n_avg) / (M * c))


if __name__ == "__main__":
    import sys

    out = {}
    for path in sys.argv[1:]:
        with open(path) as f:
            b = json.load(f)
        fit = fit_all(b)
        out[b["model"]] = fit
        print(json.dumps(fit, indent=1))
    with open("results/fits.json", "w") as f:
        json.dump(out, f, indent=1)
