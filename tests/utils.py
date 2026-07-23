import numpy as np
import mlx.core as mx
import huggingface_hub
from types import SimpleNamespace

AVAILABLE_STREAMS = [mx.cpu, mx.gpu]
AVAILABLE_STREAMS_IDS = ["cpu", "gpu"]
PRECISIONS = [mx.float32, mx.float16]
PRECISION_IDS = ["f32", "f16"]


def tiny_qwen3_mlx_model(num_hidden_layers: int = 1) -> SimpleNamespace:
    """Build a small MLX-shaped Qwen3 model for integration tests."""

    def quantized_layer(out_dim: int, in_dim: int) -> SimpleNamespace:
        weight = mx.random.normal((out_dim, in_dim)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        return SimpleNamespace(
            weight=packed,
            scales=scales,
            biases=biases,
            group_size=128,
            bits=4,
        )

    args = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=128,
        vocab_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        max_position_embeddings=256,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    layers = []
    for _ in range(num_hidden_layers):
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=quantized_layer(128, 128),
                    k_proj=quantized_layer(64, 128),
                    v_proj=quantized_layer(64, 128),
                    o_proj=quantized_layer(128, 128),
                    q_norm=SimpleNamespace(weight=mx.ones((64,), mx.bfloat16)),
                    k_norm=SimpleNamespace(weight=mx.ones((64,), mx.bfloat16)),
                ),
                mlp=SimpleNamespace(
                    gate_proj=quantized_layer(128, 128),
                    up_proj=quantized_layer(128, 128),
                    down_proj=quantized_layer(128, 128),
                ),
                input_layernorm=SimpleNamespace(weight=mx.ones((128,), mx.bfloat16)),
                post_attention_layernorm=SimpleNamespace(
                    weight=mx.ones((128,), mx.bfloat16)
                ),
            )
        )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            embed_tokens=quantized_layer(128, 128),
            layers=layers,
            norm=SimpleNamespace(weight=mx.ones((128,), mx.bfloat16)),
        ),
    )


def assert_allclose(
    a: mx.array,
    b: mx.array,
    precision: mx.Dtype,
    rtol: float | None = None,
    atol: float | None = None,
    message: str | None = None,
):
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    if b.dtype == mx.bfloat16:
        b = b.astype(mx.float32)
    a = np.array(a)
    b = np.array(b)
    if precision == mx.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-6
    elif precision == mx.float16:
        rtol = rtol or 5.0e-2
        atol = atol or 1.0e-3
    elif precision == mx.bfloat16:
        rtol = rtol or 5.0e-2
        atol = atol or 1.0e-2
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.invert(np.isclose(a, b, rtol=rtol, atol=atol))
        with np.printoptions(precision=3, suppress=True):
            print("a=", a)
            print("b=", b)
            print("diff_a=", a * diff)
            print("diff_b=", b * diff)
            print("diff_a_val=", a[diff])
            print("diff_b_val=", b[diff])
            assert False, f"result mismatch: {message}"


def np_type_to_mx_type(np_type: np.dtype) -> mx.Dtype:
    if np_type == np.float32:
        return mx.float32
    elif np_type == np.float16:
        return mx.float16
    else:
        raise ValueError(f"Unsupported numpy type: {np_type}")


def qwen3_0_6b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen3-0.6B-MLX-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-0.6B-4bit model: {e}")
        return False


def qwen3_1_7b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen3-1.7B-MLX-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-1.7B-4bit model: {e}")
        return False


def qwen3_4b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen3-4B-MLX-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-4B-4bit model: {e}")
        return False
