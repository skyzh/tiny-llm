import mlx.core as mx
from .tiny_llm_base import quantized_matmul
from .utils import assert_allclose


def quantized_matmul_helper(
    stream: mx.Stream,
    precision: mx.Dtype,
    identity_matrix: bool,
):
    with mx.stream(stream):
        group_size = 128
        if identity_matrix:
            input = mx.eye(group_size, dtype=precision)
        else:
            input = mx.random.normal(shape=(3, group_size), dtype=precision)
        weight = mx.random.normal(shape=(5, group_size), dtype=precision)
        w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
        user_out = quantized_matmul(
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=4,
            a=input,
            b=w_q,
            transpose_b=True,
        )
        ref_out = mx.quantized_matmul(
            input,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=4,
            transpose=True,
        )
        if identity_matrix:
            assert_allclose(user_out, ref_out, precision)
        else:
            assert_allclose(
                user_out,
                ref_out,
                precision,
                atol=5.0e-1,
                message=f"quantized matmul {precision} comparison",
            )


def test_task_2_quantized_matmul_simple_bf16_cpu():
    quantized_matmul_helper(mx.cpu, mx.bfloat16, True)


def test_task_2_quantized_matmul_complex_bf16_cpu():
    quantized_matmul_helper(mx.cpu, mx.bfloat16, False)


def test_task_2_quantized_matmul_simple_f16_cpu():
    quantized_matmul_helper(mx.cpu, mx.float16, True)


def test_task_2_quantized_matmul_complex_f16_cpu():
    quantized_matmul_helper(mx.cpu, mx.float16, False)


def test_task_3_quantized_matmul_simple_bf16_gpu():
    quantized_matmul_helper(mx.gpu, mx.bfloat16, True)


def test_task_3_quantized_matmul_complex_bf16_gpu():
    quantized_matmul_helper(mx.gpu, mx.bfloat16, False)


def test_task_3_quantized_matmul_simple_f16_gpu():
    quantized_matmul_helper(mx.gpu, mx.float16, True)


def test_task_3_quantized_matmul_complex_f16_gpu():
    quantized_matmul_helper(mx.gpu, mx.float16, False)
