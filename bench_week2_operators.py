import argparse
from statistics import median
from time import perf_counter

import mlx.core as mx
from mlx_lm import load

from model_names import shortcut_name_to_full_name
from tiny_llm_ref.embedding import QuantizedEmbedding
from tiny_llm_ref.quantize import (
    QuantizedWeights,
    quantized_linear,
    quantized_matmul,
    quantized_matmul_vanilla,
)
from tiny_llm_ref.week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    decode_attention_custom,
    scaled_dot_product_attention,
    swiglu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Week 2 operators at shapes used by the selected model."
    )
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def benchmark(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        mx.eval(function())
    timings = []
    for _ in range(iterations):
        start = perf_counter()
        mx.eval(function())
        timings.append(perf_counter() - start)
    return median(timings) * 1_000_000


def report(name: str, course_us: float, mlx_us: float) -> None:
    relative = course_us / mlx_us
    print(
        f"{name:<22} course={course_us:>9.1f} us  "
        f"mlx={mlx_us:>9.1f} us  latency={relative:>5.2f}x"
    )


def report_matmul_progression(
    name: str, vanilla_us: float, simdgroup_us: float, mlx_us: float
) -> None:
    print(
        f"{name:<22} vanilla={vanilla_us:>9.1f} us  "
        f"simdgroup={simdgroup_us:>9.1f} us  mlx={mlx_us:>9.1f} us  "
        f"speedup={vanilla_us / simdgroup_us:>5.2f}x"
    )


def main() -> None:
    args = parse_args()
    if args.context <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("context and iterations must be positive; warmup cannot be negative")

    model, _ = load(shortcut_name_to_full_name(args.model))
    layer = model.model.layers[0]
    precision = model.model.embed_tokens.scales.dtype
    hidden_size = model.args.hidden_size
    head_dim = model.args.head_dim
    num_heads = model.args.num_attention_heads
    num_kv_heads = model.args.num_key_value_heads

    print(f"Model={shortcut_name_to_full_name(args.model)} context={args.context}")
    print("Median synchronized latency; lower is better.")

    embedding_weights = QuantizedWeights.from_mlx_layer(model.model.embed_tokens)
    embedding = QuantizedEmbedding(
        model.args.vocab_size, hidden_size, embedding_weights
    )
    token = mx.array([[42]], dtype=mx.int32)
    report(
        "quantized embedding",
        benchmark(lambda: embedding(token), args.warmup, args.iterations),
        benchmark(
            lambda: model.model.embed_tokens(token),
            args.warmup,
            args.iterations,
        ),
    )
    projections = [
        ("q projection", layer.self_attn.q_proj, hidden_size),
        ("k projection", layer.self_attn.k_proj, hidden_size),
        ("v projection", layer.self_attn.v_proj, hidden_size),
        ("o projection", layer.self_attn.o_proj, num_heads * head_dim),
        ("gate projection", layer.mlp.gate_proj, hidden_size),
        ("up projection", layer.mlp.up_proj, hidden_size),
        ("down projection", layer.mlp.down_proj, model.args.intermediate_size),
        ("lm head", model.model.embed_tokens, hidden_size),
    ]
    for name, mlx_layer, input_dim in projections:
        weights = QuantizedWeights.from_mlx_layer(mlx_layer)
        x = mx.random.normal((1, 1, input_dim)).astype(precision)
        mx.eval(x, weights.weight, weights.scales, weights.biases)
        course_us = benchmark(
            lambda x=x, weights=weights: quantized_linear(x, weights),
            args.warmup,
            args.iterations,
        )
        mlx_us = benchmark(
            lambda x=x, weights=weights: mx.quantized_matmul(
                x,
                weights.weight,
                weights.scales,
                weights.biases,
                transpose=True,
                group_size=weights.group_size,
                bits=weights.bits,
            ),
            args.warmup,
            args.iterations,
        )
        report(name, course_us, mlx_us)

    prefill_weights = QuantizedWeights.from_mlx_layer(layer.self_attn.q_proj)
    prefill_x = mx.random.normal((args.context, hidden_size)).astype(precision)
    mx.eval(prefill_x)
    report_matmul_progression(
        "prefill q matmul",
        benchmark(
            lambda: quantized_matmul_vanilla(
                prefill_weights.scales,
                prefill_weights.biases,
                prefill_weights.group_size,
                prefill_weights.bits,
                prefill_x,
                prefill_weights.weight,
                True,
            ),
            args.warmup,
            args.iterations,
        ),
        benchmark(
            lambda: quantized_matmul(
                prefill_weights.scales,
                prefill_weights.biases,
                prefill_weights.group_size,
                prefill_weights.bits,
                prefill_x,
                prefill_weights.weight,
                True,
            ),
            args.warmup,
            args.iterations,
        ),
        benchmark(
            lambda: mx.quantized_matmul(
                prefill_x,
                prefill_weights.weight,
                prefill_weights.scales,
                prefill_weights.biases,
                transpose=True,
                group_size=prefill_weights.group_size,
                bits=prefill_weights.bits,
            ),
            args.warmup,
            args.iterations,
        ),
    )

    x_norm = mx.random.normal((1, 1, hidden_size)).astype(precision)
    rms = FastRMSNorm(
        hidden_size,
        layer.input_layernorm.weight,
        eps=model.args.rms_norm_eps,
    )
    mx.eval(x_norm)
    report(
        "RMSNorm",
        benchmark(lambda: rms(x_norm), args.warmup, args.iterations),
        benchmark(
            lambda: mx.fast.rms_norm(
                x_norm, layer.input_layernorm.weight, model.args.rms_norm_eps
            ),
            args.warmup,
            args.iterations,
        ),
    )

    x_rope = mx.random.normal((1, 1, num_heads, head_dim)).astype(precision)
    rope = FastRoPE(head_dim, model.args.max_position_embeddings, model.args.rope_theta)
    mx.eval(x_rope)
    report(
        "RoPE",
        benchmark(lambda: rope(x_rope, 17), args.warmup, args.iterations),
        benchmark(
            lambda: mx.fast.rope(
                x_rope,
                head_dim,
                traditional=False,
                base=model.args.rope_theta,
                scale=1.0,
                offset=17,
            ),
            args.warmup,
            args.iterations,
        ),
    )

    gate = mx.random.normal((1, 1, model.args.intermediate_size)).astype(precision)
    up = mx.random.normal(gate.shape).astype(precision)
    mx.eval(gate, up)
    report(
        "SwiGLU",
        benchmark(lambda: swiglu(gate, up), args.warmup, args.iterations),
        benchmark(
            lambda: gate * mx.sigmoid(gate) * up,
            args.warmup,
            args.iterations,
        ),
    )

    query = mx.random.normal((1, num_heads, 1, head_dim)).astype(precision)
    key = mx.random.normal((1, num_kv_heads, args.context, head_dim)).astype(precision)
    value = mx.random.normal(key.shape).astype(precision)
    scale = head_dim**-0.5
    mx.eval(query, key, value)
    report(
        "SDPA (two matmuls)",
        benchmark(
            lambda: scaled_dot_product_attention(query, key, value, scale, None),
            args.warmup,
            args.iterations,
        ),
        benchmark(
            lambda: mx.fast.scaled_dot_product_attention(
                query, key, value, scale=scale, mask=None
            ),
            args.warmup,
            args.iterations,
        ),
    )
    custom_attention_us = benchmark(
        lambda: decode_attention_custom(query, key, value, scale, None),
        args.warmup,
        args.iterations,
    )
    print(f"{'SDPA (online Metal)':<22} course={custom_attention_us:>9.1f} us")


if __name__ == "__main__":
    main()
