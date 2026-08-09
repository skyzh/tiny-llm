# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
import tiny_llm_ext


LEARNER_EXTENSION_INTERFACES = {
    "quantized_matmul": "Week 2, Day 3",
    "rms_norm": "Week 2, Day 4",
    "rope": "Week 2, Day 4",
    "swiglu": "Week 2, Day 4",
    "decode_attention": "Week 2, Day 5",
    "paged_cache_update": "Week 3, Day 3",
    "quantized_embedding": "Week 3, Day 4",
    "paged_attention": "Week 3, Day 4",
}

missing = sorted(
    name for name in LEARNER_EXTENSION_INTERFACES if not hasattr(tiny_llm_ext, name)
)
if missing:
    raise RuntimeError(f"starter extension is missing learner interfaces: {missing}")

a = mx.ones((3, 4))
b = mx.ones((3, 4))
c = tiny_llm_ext.axpby(a, b, 4.0, 2.0, stream=mx.cpu)

print(f"c shape: {c.shape}")
print(f"c dtype: {c.dtype}")
print(f"c correct: {mx.all(c == 6.0).item()}")
