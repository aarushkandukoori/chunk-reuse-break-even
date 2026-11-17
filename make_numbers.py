"""Write note/numbers.tex (LaTeX macros) from the saved timings and results/summary.json,
so every number in the note comes straight from saved outputs."""
import json

import costmodel as cm

B = {"0.5B": json.load(open("results/qwen2.5-0.5b-4bit.json")),
     "1.5B": json.load(open("results/qwen2.5-1.5b-4bit.json"))}
S = json.load(open("results/summary.json"))
fits = S["fits"]
out = {}


def put(name, val, fmt):
    out[name] = fmt.format(val)


def scen(model, M, c):
    return [r for r in S["scenarios"] if r["model"] == model and r["M"] == M and r["c"] == c][0]


for tag, key in [("A", "0.5B"), ("B", "1.5B")]:
    b = B[key]
    pre = {p["L"]: p["sec"] for p in b["prefill"]}
    put(f"pre{tag}short", 1e3 * pre[128] / 128, "{:.3f}")
    put(f"pre{tag}long", 1e3 * pre[2048] / 2048, "{:.3f}")
    put(f"pre{tag}msshort", 1e3 * pre[128], "{:.1f}")
    put(f"pre{tag}mslong", 1e3 * pre[2048], "{:.0f}")
    put(f"pre{tag}msthirtytwo", 1e3 * pre[32], "{:.1f}")
    rs = {r["C"]: r for r in b["restore"]}
    put(f"mem{tag}", 1e3 * rs[512]["mem_sec"] / 512, "{:.4f}")
    put(f"file{tag}", 1e3 * rs[512]["file_sec"] / 512, "{:.4f}")
    put(f"rope{tag}", 1e3 * rs[512]["reencode_sec"], "{:.2f}")
    put(f"lookup{tag}", 1e3 * rs[512]["lookup_sec"], "{:.3f}")
    put(f"filemb{tag}", rs[512]["file_bytes"] / 1e6, "{:.1f}")
    f = fits[key]
    put(f"const{tag}", 1e3 * f["forward_coef"][0], "{:.1f}")
    put(f"fiterr{tag}", 100 * f["fit_median_rel_err"], "{:.1f}")
    put(f"fitmax{tag}", 100 * f["fit_max_rel_err"], "{:.0f}")
    put(f"peak{tag}", b["peak_mem_gb"], "{:.2f}")
    put(f"wall{tag}", b["wall_sec"], "{:.0f}")
    for M in [1, 4]:
        for c in [32, 64, 128, 256, 512, 1024]:
            r = scen(key, M, c)
            w = {32: "xxxii", 64: "lxiv", 128: "cxxviii", 256: "cclvi", 512: "dxii", 1024: "mxxiv"}[c]
            m = "one" if M == 1 else "four"
            put(f"be{tag}{m}{w}", 100 * r["break_even_rho"], "{:.0f}")
            put(f"befile{tag}{m}{w}", 100 * r["break_even_rho_file"], "{:.0f}")
            put(f"bemerged{tag}{m}{w}", 100 * r["break_even_rho_merged"], "{:.0f}")
            put(f"sdaric{tag}{m}{w}", r["speedup_daric_default"], "{:.2f}")
            put(f"sdaricm{tag}{m}{w}", r["speedup_daric_default_merged"], "{:.2f}")
            put(f"sfifteen{tag}{m}{w}", r["speedup_rho0.15"], "{:.2f}")
            put(f"szero{tag}{m}{w}", r["speedup_rho0.0"], "{:.2f}")

    # smallest reused-token count (M=4) where reuse at 15% recompute is faster than full prefill
    best = None
    for c in range(4, 2049, 4):
        if cm.speedup(f, 4, c, 32, 0.15) >= 1:
            best = 4 * c
            break
    out[f"crossfifteen{tag}"] = str(best) if best else "none up to 8192"

for m in S["measured_check"]:
    tag = "A" if m["model"] == "0.5B" else "B"
    if m["C"] == 256:
        w = {32: "xxxii", 64: "lxiv", 128: "cxxviii", 256: "cclvi"}[m["k"]]
        put(f"meas{tag}{w}", m["speedup"], "{:.2f}")

B5 = B["1.5B"]
out["chip"] = B5["chip"]
out["mlxver"] = B5["mlx_version"]

with open("note/numbers.tex", "w") as fh:
    for k, v in out.items():
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
json.dump(out, open("results/note_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
