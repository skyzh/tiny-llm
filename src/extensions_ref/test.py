from tiny_llm_ext_ref import quantized_matmul
import mlx.core as mx
import numpy as np

input = mx.array(np.random.randn(3, 128)).astype(mx.bfloat16)
weight = mx.array(np.random.randn(5, 128)).astype(mx.bfloat16)
w_q, scales, biases = mx.quantize(weight, group_size=128, bits=4)
user_out = quantized_matmul(
    scales=scales,
    biases=biases,
    group_size=128,
    bits=4,
    a=input,
    b=w_q,
    transpose_b=True,
)
print(user_out)
