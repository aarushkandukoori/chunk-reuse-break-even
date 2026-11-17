"""Make the two figures and results/summary.json from the saved timings.

usage: python3 analyze.py results/qwen2.5-0.5b-4bit.json results/qwen2.5-1.5b-4bit.json
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import costmodel as cm  # noqa: E402

# Okabe-Ito colorblind-safe palette
C_BLUE, C_ORANGE, C_GREEN, C_PINK, C_SKY, C_VERM, C_BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

Q = 32          # query tokens
RHOS = [0.0, 0.05, 0.15, 0.30]
CHUNKS = [16, 32, 64, 128, 256, 512, 1024]
PLOT_CHUNKS = [16, 32, 64, 128, 256, 512]


def short(name):
    return "0.5B" if "0.5B" in name else "1.5B"


def main(paths):
    benches = [json.load(open(p)) for p in paths]
    fits = {short(b["model"]): cm.fit_all(b) for b in benches}
    big = [b for b in benches if short(b["model"]) == "1.5B"][0]
    bigfit = fits["1.5B"]

    # ---------- Figure 1: measured costs ----------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a = ax[0]
    for b, col in zip(benches, [C_SKY, C_BLUE]):
        L = [p["L"] for p in b["prefill"]]
        a.plot(L, [1e3 * p["sec"] / p["L"] for p in b["prefill"]], "o-", color=col,
               label=f"full prefill, {short(b['model'])}")
    C = [r["C"] for r in big["restore"]]
    a.plot(C, [1e3 * r["mem_sec"] / r["C"] for r in big["restore"]], "s--", color=C_GREEN, label="restore from memory, 1.5B")
    a.plot(C, [1e3 * r["file_sec"] / r["C"] for r in big["restore"]], "^--", color=C_ORANGE, label="restore from file, 1.5B")
    a.plot(C, [1e3 * r["reencode_sec"] / r["C"] for r in big["restore"]], "d:", color=C_PINK, label="RoPE re-encode, 1.5B")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("tokens (prompt length, or chunk length)")
    a.set_ylabel("ms per token")
    a.set_title("(a) prefill vs restore, per token")
    a.legend(fontsize=6.5, frameon=False)

    a = ax[1]
    ctxs = sorted({r["ctx"] for r in big["forward"]})
    cols = [C_SKY, C_BLUE, C_GREEN, C_ORANGE, C_VERM]
    for ctx, col in zip(ctxs, cols):
        rows = sorted([r for r in big["forward"] if r["ctx"] == ctx], key=lambda r: r["k"])
        a.plot([r["k"] for r in rows], [1e3 * r["sec"] for r in rows], "o-", color=col, ms=3, label=f"context {ctx}")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("tokens recomputed in one pass (k)")
    a.set_ylabel("ms for the pass")
    a.set_title("(b) recompute pass, 1.5B")
    a.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig("figures/fig1_measured_costs.png", dpi=200)
    fig.savefig("figures/fig1_measured_costs.pdf")

    # ---------- Figure 2: break-even ----------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a = ax[0]
    short_chunks = [c for c in CHUNKS if c <= 256]
    for rho, col in zip(RHOS, [C_BLUE, C_GREEN, C_ORANGE, C_VERM]):
        a.plot(short_chunks, [cm.speedup(bigfit, 4, c, Q, rho) for c in short_chunks], "o-", color=col, ms=3,
               label=f"recompute {int(rho*100)}%")
    a.axhline(1, color=C_BLACK, lw=0.8, ls="--")
    a.set_xscale("log", base=2)
    a.set_xlabel("chunk length c (4 chunks + 32-token query)")
    a.set_ylabel("speedup over full prefill")
    a.set_title("(a) 1.5B, chunks restored from memory")
    a.legend(fontsize=6.5, frameon=False)

    a = ax[1]
    for (name, fit), col in zip(sorted(fits.items()), [C_SKY, C_BLUE]):
        for M, ls in [(1, "--"), (4, "-")]:
            a.plot([M * c for c in PLOT_CHUNKS], [100 * cm.break_even_rho(fit, M, c, Q) for c in PLOT_CHUNKS],
                   ls, marker="o", ms=3, color=col, label=f"{name}, {M} chunk{'s' if M > 1 else ''}")
    xs = [4 * c for c in PLOT_CHUNKS]
    a.plot(xs, [100 * cm.daric_default_rho(4, c) for c in PLOT_CHUNKS], ":", color=C_VERM,
           label="DARIC default budget (32/chunk)")
    a.set_xscale("log", base=2)
    a.set_ylim(-3, 103)
    a.set_xlabel("reused tokens in the prompt (M x c)")
    a.set_ylabel("break-even recompute %")
    a.set_title("(b) most recompute before reuse loses")
    a.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/fig2_break_even.png", dpi=200)
    fig.savefig("figures/fig2_break_even.pdf")

    # ---------- summary numbers ----------
    summ = {"fits": fits, "query_tokens": Q, "scenarios": []}
    for name, fit in sorted(fits.items()):
        for M in [1, 4]:
            for c in CHUNKS:
                row = {"model": name, "M": M, "c": c, "q": Q,
                       "full_ms": 1e3 * cm.full_time(fit, M, c, Q),
                       "break_even_rho": cm.break_even_rho(fit, M, c, Q),
                       "break_even_rho_file": cm.break_even_rho(fit, M, c, Q, source="file"),
                       "break_even_rho_merged": cm.break_even_rho(fit, M, c, Q, merged=True),
                       "daric_default_rho": cm.daric_default_rho(M, c)}
                for rho in RHOS:
                    row[f"speedup_rho{rho}"] = cm.speedup(fit, M, c, Q, rho)
                    row[f"reuse_ms_rho{rho}"] = 1e3 * cm.reuse_time(fit, M, c, Q, rho)
                row["speedup_daric_default"] = cm.speedup(fit, M, c, Q, row["daric_default_rho"])
                row["speedup_daric_default_merged"] = cm.speedup(fit, M, c, Q, row["daric_default_rho"], merged=True)
                row["speedup_daric_default_file"] = cm.speedup(fit, M, c, Q, row["daric_default_rho"], source="file")
                summ["scenarios"].append(row)
    # measured-only check (no fit): one chunk of C tokens, no query, one recompute pass of k tokens
    # reuse = restore_mem(C) + lookup + RoPE + forward(k, ctx=C)   vs   full = prefill(C)
    summ["measured_check"] = []
    for b in benches:
        pre = {p["L"]: p["sec"] for p in b["prefill"]}
        fw = {(r["k"], r["ctx"]): r["sec"] for r in b["forward"]}
        for r in b["restore"]:
            C = r["C"]
            if C not in pre:
                continue
            for k in [32, 64, 128, 256]:
                if (k, C) not in fw or k > C:
                    continue
                reuse = r["mem_sec"] + r["reencode_sec"] + r["lookup_sec"] + fw[(k, C)]
                summ["measured_check"].append({"model": short(b["model"]), "C": C, "k": k, "rho": k / C,
                                               "full_ms": 1e3 * pre[C], "reuse_ms": 1e3 * reuse,
                                               "speedup": pre[C] / reuse})
    for m in summ["measured_check"]:
        print("measured", m)
    json.dump(summ, open("results/summary.json", "w"), indent=1)
    json.dump(fits, open("results/fits.json", "w"), indent=1)
    for r in summ["scenarios"]:
        print(f"{r['model']} M={r['M']} c={r['c']:5d}: full {r['full_ms']:7.1f} ms | speedup r0 {r['speedup_rho0.0']:.2f} "
              f"r5 {r['speedup_rho0.05']:.2f} r15 {r['speedup_rho0.15']:.2f} r30 {r['speedup_rho0.3']:.2f} "
              f"| BE {100*r['break_even_rho']:.0f}% (file {100*r['break_even_rho_file']:.0f}%) "
              f"| DARIC default {100*r['daric_default_rho']:.0f}% -> {r['speedup_daric_default']:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1:])
