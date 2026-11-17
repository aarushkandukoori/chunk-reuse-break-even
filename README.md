# chunk-reuse-break-even

When does reusing a moved chunk's KV cache, plus recomputing some of its tokens, beat a plain full prefill for a small model?

This repo times the pieces on two small 4-bit models (Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, MLX) on an Apple M4 laptop, fits a simple cost model, and finds the break-even recompute fraction across chunk length, number of chunks and context length.

It is a cost study only. It does not reproduce any method's accuracy, and the machine is a laptop, not a phone.

The setting follows position-independent KV cache reuse as described in DARIC (arXiv:2611.09109): restore each stored chunk cache, re-encode its positions, and recompute a few selected tokens against the full context.

## What is timed

- `T(k, ctx)`: k new tokens through the model on top of a cache holding ctx tokens. With ctx = 0 this is a full prefill. With ctx > 0 it stands in for a recompute pass.
- `R(c)`: restoring a c-token chunk. This is a hash lookup, a copy of the stored KV (from memory, or a warm safetensors file read), and one RoPE pass over the keys.

Batch size 1, random token ids, one warm-up per shape, median of 3 repeats.

## The two paths

- full prefill = `T(M*c + q, 0)`
- chunk reuse = `M*R(c) + T(q, 0) + T(k, L - k)`, with `k = ceil(rho*M*c) + q` and `L = M*c + q`

The query gets its own pass and is then recomputed as the latest window. `merged=True` folds the query into the recompute pass instead.

## Files

- `bench/bench.py` runs the timings and saves JSON after every block
- `results/*.json` holds the saved timings, the fits and the scenario table
- `costmodel.py` has the fit and the break-even model
- `analyze.py` makes the figures and `results/summary.json`
- `tests/` holds the unit tests for the cost model (`python3 -m pytest -q tests`)
- `note/Kandukoori_Chunk_Reuse_Break_Even.pdf` is a 3-page write-up
- `docs/index.html` is an interactive page where you can change chunk length, chunk count, query length and recompute fraction

## Run it

```
pip install mlx mlx-lm numpy matplotlib pytest
bash bench/run_all.sh
python3 analyze.py results/qwen2.5-0.5b-4bit.json results/qwen2.5-1.5b-4bit.json
python3 -m pytest -q tests
```

## Limits

- Laptop, not phone.
- Recompute is timed as appending k tokens, not as an in-place overwrite of scattered positions.
- RoPE and the hash lookup stand in for re-encoding and a real cache index.
- No quality measurement.

Made by Aarush Kandukoori
