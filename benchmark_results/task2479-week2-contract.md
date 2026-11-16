# Six-day Week 2 implementation contract

This change has one evolving `kv-cache`, growing its storage in 256-token chunks. `capacity=None` is the default and has no request-length bound; an explicit capacity is an upper bound, not a request to allocate it all. `allocated_capacity` reports physical slots, and `offset` reports the initialized prefix. Rewind and reset retain storage and hide abandoned tokens. The byte counters describe requested copies and writes, not measured GPU traffic.

| Day | Chapter | Gate / focused filters | Checkpoint | Benchmark variant | Replay case |
|---|---|---|---|---|---|
| 1 KV Cache | week2-01-kv-cache.md | test_week_2_day_1.py; task_1, tasks_2_and_3, chunk_growth, rewind or overflow | kv-cache | week2-kv-cache | kv-cache:decode:128 |
| 2 Packed W4 Weights | week2-02-quantize-model.md | test_week_2_day_2.py; task_1, task_3, task_4 | packed-weights | week2-packed-weights | packed-weights:decode:128 |
| 3 Decode Matvec | week2-03-decode-matvec.md | test_week_2_day_3.py; task_1, task_2, task_3 | quantized-matvec | week2-quantized-matvec | quantized-matvec:decode:128 |
| 4 SIMD Prefill | week2-04-simd-matrix-prefill.md | test_week_2_day_4.py; task_1, task_2, task_3 | simd-matmul | week2-simd-matmul | simd-matmul:prefill:128 |
| 5 Fused Kernels | week2-05-fused-model-kernels.md | test_week_2_day_5.py; task_1, task_2, task_3, task_4 | rmsnorm / rope / swiglu | week2-rmsnorm / week2-rope / week2-swiglu | rmsnorm:decode:128 / rope:decode:128 / swiglu:decode:128 |
| 6 Tiled Attention | week2-06-tiled-prefill-attention.md | test_week_2_day_6.py; task_1, task_2, task_3 | tiled-prefill / selected | week2-tiled-prefill / week2-selected | tiled-prefill:prefill:128 / selected:prefill:128 |

Every day uses `pdm run test --week 2 --day N` (reference: `test-refsol`), then `pdm run main --solution tiny_llm --loader week2 --week2-checkpoint NAME --model qwen3-0.6b --max-tokens 16`. Benchmark uses the matching `bench-week2-progression --variant` and replay uses `profile-week2-kernels --case`. Reference-only selected scheduling remains explicitly unsupported by complete public-operator replay; that limitation is unchanged.

## Starter files and functions

- Day 1: `src/tiny_llm/kv_cache.py`: `TinyKvFullCache.update_and_fetch`, `_logical_key_values`, `allocated_capacity`, `rewind`, `reset`; `qwen3_week2.py`: cached model and factory; `generate.py`: `simple_generate_with_kv_cache`. One cache, no second cache implementation.
- Day 2 Task 1: `quantize.py`: `QuantizedWeights.from_mlx_layer`, `dequantize_weights`; `embedding.py`: `QuantizedEmbedding.__call__`. Task 2: `src/extensions/src/quantized_matmul.cpp`: `tiny_llm_ext::quantized_matmul`, `QuantizedMatmul::eval_cpu`. Task 3: the same file’s `QuantizedMatmul::eval_gpu`, plus `quantized_matmul_vanilla_w4a16_g128` in `quantized_matmul.metal`; `quantize.py`: flatten/contiguous/native/reshape wrapper. Task 4: `qwen3_week2.py`: model uses packed projections, embedding and output head, with `simdgroup_matvec=False` and `simdgroup_matmul=False`. The existing C++ flag selects the simple kernel even for M=1; no new native entry is required.
- Day 3 Task 1: `quantized_matvec_x4_fast_w4a16_g128` in `quantized_matmul.metal`. Task 2: `QuantizedMatmul::eval_gpu` in `quantized_matmul.cpp`, and `quantized_matvec_custom` in `quantize.py`. Task 3: `Qwen3ModelWeek2.__init__` selects `simdgroup_matvec=True` for `quantized-matvec` and later. Prefill stays simple until Day 4.
- Day 4: cooperative matrix helper and `quantized_matmul_simdgroup_w4a16_g128`; model enables `simdgroup_matmul`. Former Day 3 task numbers are unchanged.
- Day 5: `week2_kernels.cpp` / `.metal`: RMSNorm, RoPE and SwiGLU; model wiring. Former Day 4 tasks are unchanged.
- Day 6: `_dense_attention_prefill_mma` and tiled dense attention in the same extension files, plus model wiring. Former Day 5 tasks are unchanged.

## Explicit compatibility

`capacity-cache` selects the same current `kv-cache` implementation and is accepted by model, main, benchmark, profiler and capture parsers. It is absent from the default progression. `week2-capacity-cache` is a benchmark alias for `week2-kv-cache`.

Old five-day tests use `pdm run test --week 2 --day N --legacy-week2-days` or `test-refsol`. Old Day 1 maps to new Day 1; old Day 2 runs new Days 2 and 3; old Days 3, 4 and 5 map to new Days 4, 5 and 6. Without the flag, day numbers follow the new book. Old copied learner test files may be replaced with `pdm run copy-test --week 2 --day N --force` before running the new day.

The matching book change supplies the six chapter paths above, summary/README progression and old-address redirects. These changes are reviewed and land together. No checkpoint promises a timing win. The new cache comparison reports current fixed-size versus chunked behavior separately from historical fixed-size versus per-append concatenation.
