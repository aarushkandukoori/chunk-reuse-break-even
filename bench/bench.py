"""Time the pieces of chunk KV reuse on a small 4-bit MLX model.

What gets timed (all on one Apple laptop, batch size 1):
  1. forward(k, ctx): run k new tokens through the model on top of a KV cache
     that already holds ctx tokens. ctx = 0 is a plain full prefill of k tokens.
     With ctx > 0 this is also our stand-in for "recompute k selected tokens
     against the whole context" (same weight matmuls, attention over ctx keys).
  2. restore_mem(C): build a cache from KV arrays for a C-token chunk that are
     already in memory (concatenate after a prefix, then evaluate).
  3. restore_file(C): load the same chunk's KV from a safetensors file
     (file was just written, so this is a warm read from the OS page cache).
  4. reencode(C): one RoPE rotation pass over the chunk's keys in every layer
     (mx.fast.rope), a timing stand-in for position re-encoding.
  5. lookup(C): sha256 of the chunk's token ids, a stand-in for the cache index lookup.

Token ids are random. Timing does not depend on the text.
Results are written after every block so a pause loses nothing.

usage: python3 bench.py <hf model id> <out.json> [--quick]
"""
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import (
    KVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)

REPEATS = 3


def now():
    return time.perf_counter()


def last_logits(model, tokens, cache):
    """Forward pass that only computes logits for the last position."""
    hidden = model.model(tokens, cache=cache)
    h = hidden[:, -1:, :]
    if getattr(model.args, "tie_word_embeddings", False):
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    states = []
    for c in cache:
        states.extend([c.keys, c.values])
    mx.eval(logits, *states)
    return logits


def rand_tokens(n, vocab, seed):
    key = mx.random.key(seed)
    return mx.random.randint(100, vocab - 100, (1, n), key=key).astype(mx.int32)


def time_forward(model, vocab, k, ctx):
    """Median seconds to push k tokens on top of a ctx-token cache."""
    cache = make_prompt_cache(model)
    if ctx > 0:
        last_logits(model, rand_tokens(ctx, vocab, 1), cache)
    x = rand_tokens(k, vocab, 2)
    # warm-up for this shape, then trim back so every repeat sees the same ctx
    last_logits(model, x, cache)
    trim_prompt_cache(cache, k)
    ts = []
    for _ in range(REPEATS):
        t0 = now()
        last_logits(model, x, cache)
        ts.append(now() - t0)
        trim_prompt_cache(cache, k)
    return statistics.median(ts)


def chunk_kv(model, vocab, C):
    """KV arrays (per layer) for a C-token chunk computed on its own."""
    cache = make_prompt_cache(model)
    last_logits(model, rand_tokens(C, vocab, 3), cache)
    out = []
    for c in cache:
        k, v = c.keys[..., :C, :], c.values[..., :C, :]
        mx.eval(k, v)
        out.append((k, v))
    return out, cache


def time_restore_mem(kv, prefix_kv):
    ts = []
    for _ in range(REPEATS + 1):
        t0 = now()
        new = []
        for (k, v), (pk, pv) in zip(kv, prefix_kv):
            c = KVCache()
            nk = mx.concatenate([pk, k], axis=2)
            nv = mx.concatenate([pv, v], axis=2)
            c.state = (nk, nv, nk.shape[2])
            new.append(c)
        mx.eval(*[c.keys for c in new], *[c.values for c in new])
        ts.append(now() - t0)
    return statistics.median(ts[1:])


def time_restore_file(cache, path):
    save_prompt_cache(path, cache)
    ts = []
    for _ in range(REPEATS + 1):
        t0 = now()
        loaded = load_prompt_cache(path)
        mx.eval(*[c.keys for c in loaded], *[c.values for c in loaded])
        ts.append(now() - t0)
    size = os.path.getsize(path)
    os.remove(path)
    return statistics.median(ts[1:]), size


def time_reencode(kv, head_dim, base):
    ts = []
    for _ in range(REPEATS + 1):
        t0 = now()
        outs = [mx.fast.rope(k, head_dim, traditional=False, base=base, scale=1.0, offset=37) for k, _ in kv]
        mx.eval(*outs)
        ts.append(now() - t0)
    return statistics.median(ts[1:])


def time_lookup(C):
    ids = list(range(C))
    ts = []
    for _ in range(REPEATS):
        t0 = now()
        hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        ts.append(now() - t0)
    return statistics.median(ts)


def main():
    model_id, out_path = sys.argv[1], sys.argv[2]
    quick = "--quick" in sys.argv
    model, tok = load(model_id)
    args = model.args
    vocab = args.vocab_size
    head_dim = getattr(args, "head_dim", None) or args.hidden_size // args.num_attention_heads
    base = getattr(args, "rope_theta", 10000.0)

    if quick:
        prefill_L = [64, 256]
        fwd_ctx = [0, 512]
        fwd_k = [16, 128]
        chunk_C = [256]
    else:
        prefill_L = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        fwd_ctx = [256, 512, 1024, 2048, 4096]
        fwd_k = [8, 16, 32, 64, 128, 256, 512]
        chunk_C = [128, 256, 512, 1024, 2048]

    res = {
        "model": model_id,
        "machine": platform.platform(),
        "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
        "mlx_version": mx.__version__,
        "repeats": REPEATS,
        "stat": "median seconds",
        "n_layers": len(model.layers),
        "head_dim": head_dim,
        "n_kv_heads": args.num_key_value_heads,
        "prefill": [],
        "forward": [],
        "restore": [],
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    # warm the kernels once
    last_logits(model, rand_tokens(64, vocab, 9), make_prompt_cache(model))

    t_start = now()
    for L in prefill_L:
        t = time_forward(model, vocab, L, 0)
        res["prefill"].append({"L": L, "sec": t})
        print(f"prefill L={L}: {t*1000:.1f} ms", flush=True)
        save()

    for ctx in fwd_ctx:
        for k in fwd_k:
            t = time_forward(model, vocab, k, ctx)
            res["forward"].append({"k": k, "ctx": ctx, "sec": t})
            print(f"forward k={k} ctx={ctx}: {t*1000:.1f} ms", flush=True)
        save()

    prefix_kv, _ = chunk_kv(model, vocab, 64)  # a 64-token prefix the chunk is appended to
    tmpdir = tempfile.mkdtemp(prefix="crbe-")
    for C in chunk_C:
        kv, cache = chunk_kv(model, vocab, C)
        tm = time_restore_mem(kv, prefix_kv)
        tf, size = time_restore_file(cache, os.path.join(tmpdir, f"chunk{C}.safetensors"))
        tr = time_reencode(kv, head_dim, base)
        tl = time_lookup(C)
        res["restore"].append(
            {"C": C, "mem_sec": tm, "file_sec": tf, "file_bytes": size, "reencode_sec": tr, "lookup_sec": tl}
        )
        print(f"restore C={C}: mem {tm*1000:.2f} ms, file {tf*1000:.2f} ms ({size/1e6:.1f} MB), "
              f"rope {tr*1000:.2f} ms, lookup {tl*1000:.3f} ms", flush=True)
        save()
    os.rmdir(tmpdir)
    res["wall_sec"] = now() - t_start
    res["peak_mem_gb"] = mx.get_peak_memory() / 1e9
    save()
    print(f"done in {res['wall_sec']:.0f} s, peak {res['peak_mem_gb']:.2f} GB")


if __name__ == "__main__":
    main()
