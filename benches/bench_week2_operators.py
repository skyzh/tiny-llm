import argparse
import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable

import mlx.core as mx
from mlx_lm import load

from model_names import shortcut_name_to_full_name


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SECTIONS = (
    "embedding",
    "decode-projections",
    "prefill-projections",
    "model-kernels",
    "attention",
)


@dataclass(frozen=True)
class OperatorImplementation:
    name: str
    silu: Callable[[mx.array], mx.array]
    embedding_type: Any
    quantized_embedding_type: Any
    rms_norm_type: Any
    rope_type: Any
    quantized_weights_type: Any
    dequantize_weights: Callable[..., mx.array]
    quantized_linear: Callable[..., mx.array]
    quantized_matmul: Callable[..., mx.array]
    quantized_matmul_vanilla: Callable[..., mx.array]
    fast_rms_norm_type: Any
    fast_rope_type: Any
    decode_attention: Callable[..., mx.array]
    readable_attention: Callable[..., mx.array]
    swiglu: Callable[[mx.array, mx.array], mx.array]


def load_implementation(name: str) -> OperatorImplementation:
    basics = importlib.import_module(f"{name}.basics")
    embedding = importlib.import_module(f"{name}.embedding")
    layer_norm = importlib.import_module(f"{name}.layer_norm")
    positional_encoding = importlib.import_module(f"{name}.positional_encoding")
    quantize = importlib.import_module(f"{name}.quantize")
    kernels = importlib.import_module(f"{name}.week2_kernels")
    return OperatorImplementation(
        name=name,
        silu=basics.silu,
        embedding_type=embedding.Embedding,
        quantized_embedding_type=embedding.QuantizedEmbedding,
        rms_norm_type=layer_norm.RMSNorm,
        rope_type=positional_encoding.RoPE,
        quantized_weights_type=quantize.QuantizedWeights,
        dequantize_weights=quantize.dequantize_weights,
        quantized_linear=quantize.quantized_linear,
        quantized_matmul=quantize.quantized_matmul,
        quantized_matmul_vanilla=quantize.quantized_matmul_vanilla,
        fast_rms_norm_type=kernels.FastRMSNorm,
        fast_rope_type=kernels.FastRoPE,
        decode_attention=kernels.decode_attention_custom,
        readable_attention=kernels.scaled_dot_product_attention,
        swiglu=kernels.swiglu,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Week 2 operators at selected model shapes."
    )
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument(
        "--solution",
        choices=("tiny_llm", "tiny_llm_ref"),
        default="tiny_llm_ref",
        help="implementation to benchmark (default: tiny_llm_ref)",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=SECTIONS,
        help="operator family to run; repeat as needed (default: all)",
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--prefill-projection",
        action="append",
        choices=("q", "k", "v", "o", "gate", "up", "down"),
        help="prefill projection to benchmark; repeat to select several (default: q)",
    )
    parser.add_argument(
        "--include-split-k",
        action="store_true",
        help="also benchmark the Day 7 split-K path",
    )
    return parser.parse_args()


def benchmark(function: Callable[[], mx.array], warmup: int, iterations: int) -> float:
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


def report_progression(
    name: str, readable_us: float, optimized_us: float, mlx_us: float
) -> None:
    print(
        f"{name:<22} readable={readable_us:>9.1f} us  "
        f"optimized={optimized_us:>9.1f} us  mlx={mlx_us:>9.1f} us  "
        f"speedup={readable_us / optimized_us:>5.2f}x"
    )


def report_split_k(
    name: str, simdgroup_us: float, split_k_us: float, mlx_us: float
) -> None:
    print(
        f"{name:<22} simd={simdgroup_us:>9.1f} us  "
        f"split-k={split_k_us:>9.1f} us  mlx={mlx_us:>9.1f} us  "
        f"speedup={simdgroup_us / split_k_us:>5.2f}x"
    )


def benchmark_embedding(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> None:
    hidden_size = model.args.hidden_size
    weights = ops.quantized_weights_type.from_mlx_layer(model.model.embed_tokens)
    embedding = ops.quantized_embedding_type(
        model.args.vocab_size, hidden_size, weights
    )
    dense_embedding = ops.embedding_type(
        model.args.vocab_size,
        hidden_size,
        ops.dequantize_weights(
            weights.weight,
            weights.scales,
            weights.biases,
            weights.group_size,
            weights.bits,
        ),
    )
    token = mx.array([[42]], dtype=mx.int32)
    mx.eval(dense_embedding.weight)
    report_progression(
        "quantized embedding",
        benchmark(lambda: dense_embedding(token), args.warmup, args.iterations),
        benchmark(lambda: embedding(token), args.warmup, args.iterations),
        benchmark(
            lambda: model.model.embed_tokens(token), args.warmup, args.iterations
        ),
    )


def benchmark_decode_projections(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> None:
    layer = model.model.layers[0]
    precision = model.model.embed_tokens.scales.dtype
    hidden_size = model.args.hidden_size
    head_dim = model.args.head_dim
    num_heads = model.args.num_attention_heads
    projections = (
        ("q projection", layer.self_attn.q_proj, hidden_size),
        ("k projection", layer.self_attn.k_proj, hidden_size),
        ("v projection", layer.self_attn.v_proj, hidden_size),
        ("o projection", layer.self_attn.o_proj, num_heads * head_dim),
        ("gate projection", layer.mlp.gate_proj, hidden_size),
        ("up projection", layer.mlp.up_proj, hidden_size),
        ("down projection", layer.mlp.down_proj, model.args.intermediate_size),
        ("lm head", model.model.embed_tokens, hidden_size),
    )
    for name, mlx_layer, input_dim in projections:
        weights = ops.quantized_weights_type.from_mlx_layer(mlx_layer)
        x = mx.random.normal((1, 1, input_dim)).astype(precision)
        mx.eval(x, weights.weight, weights.scales, weights.biases)
        course_us = benchmark(
            lambda x=x, weights=weights: ops.quantized_linear(x, weights),
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
        readable_us = benchmark(
            lambda x=x, weights=weights: ops.quantized_matmul_vanilla(
                weights.scales,
                weights.biases,
                weights.group_size,
                weights.bits,
                x,
                weights.weight,
                True,
            ),
            args.warmup,
            args.iterations,
        )
        report_progression(name, readable_us, course_us, mlx_us)


def benchmark_prefill_projections(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> None:
    layer = model.model.layers[0]
    precision = model.model.embed_tokens.scales.dtype
    hidden_size = model.args.hidden_size
    head_dim = model.args.head_dim
    num_heads = model.args.num_attention_heads
    prefill_layers = {
        "q": (layer.self_attn.q_proj, hidden_size),
        "k": (layer.self_attn.k_proj, hidden_size),
        "v": (layer.self_attn.v_proj, hidden_size),
        "o": (layer.self_attn.o_proj, num_heads * head_dim),
        "gate": (layer.mlp.gate_proj, hidden_size),
        "up": (layer.mlp.up_proj, hidden_size),
        "down": (layer.mlp.down_proj, model.args.intermediate_size),
    }
    for projection in args.prefill_projection or ["q"]:
        mlx_layer, input_dim = prefill_layers[projection]
        weights = ops.quantized_weights_type.from_mlx_layer(mlx_layer)
        x = mx.random.normal((args.context, input_dim)).astype(precision)
        mx.eval(x)
        simdgroup_us = benchmark(
            lambda: ops.quantized_matmul(
                weights.scales,
                weights.biases,
                weights.group_size,
                weights.bits,
                x,
                weights.weight,
                True,
                use_simdgroup=True,
            ),
            args.warmup,
            args.iterations,
        )
        mlx_us = benchmark(
            lambda: mx.quantized_matmul(
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
        name = f"prefill {projection} matmul"
        if not args.include_split_k:
            report(name, simdgroup_us, mlx_us)
            continue
        split_k_us = benchmark(
            lambda: ops.quantized_matmul(
                weights.scales,
                weights.biases,
                weights.group_size,
                weights.bits,
                x,
                weights.weight,
                True,
                use_simdgroup=True,
                use_split_k=True,
            ),
            args.warmup,
            args.iterations,
        )
        report_split_k(name, simdgroup_us, split_k_us, mlx_us)


def benchmark_model_kernels(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> None:
    layer = model.model.layers[0]
    precision = model.model.embed_tokens.scales.dtype
    hidden_size = model.args.hidden_size
    head_dim = model.args.head_dim
    num_heads = model.args.num_attention_heads

    x_norm = mx.random.normal((1, 1, hidden_size)).astype(precision)
    rms = ops.fast_rms_norm_type(
        hidden_size, layer.input_layernorm.weight, eps=model.args.rms_norm_eps
    )
    readable_rms = ops.rms_norm_type(
        hidden_size, layer.input_layernorm.weight, eps=model.args.rms_norm_eps
    )
    mx.eval(x_norm)
    report_progression(
        "RMSNorm",
        benchmark(lambda: readable_rms(x_norm), args.warmup, args.iterations),
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
    rope = ops.fast_rope_type(
        head_dim, model.args.max_position_embeddings, model.args.rope_theta
    )
    readable_rope = ops.rope_type(
        head_dim, model.args.max_position_embeddings, model.args.rope_theta
    )
    x_rope_mlx = x_rope.transpose(0, 2, 1, 3)
    mx.eval(x_rope, x_rope_mlx)
    mx.eval(readable_rope.cos_freqs, readable_rope.sin_freqs)
    report_progression(
        "RoPE",
        benchmark(
            lambda: readable_rope(x_rope, slice(17, 18)),
            args.warmup,
            args.iterations,
        ),
        benchmark(lambda: rope(x_rope, 17), args.warmup, args.iterations),
        benchmark(
            lambda: mx.fast.rope(
                x_rope_mlx,
                head_dim,
                traditional=False,
                base=model.args.rope_theta,
                scale=1.0,
                offset=17,
            ).transpose(0, 2, 1, 3),
            args.warmup,
            args.iterations,
        ),
    )

    gate = mx.random.normal((1, 1, model.args.intermediate_size)).astype(precision)
    up = mx.random.normal(gate.shape).astype(precision)
    mx.eval(gate, up)
    report_progression(
        "SwiGLU",
        benchmark(lambda: ops.silu(gate) * up, args.warmup, args.iterations),
        benchmark(lambda: ops.swiglu(gate, up), args.warmup, args.iterations),
        benchmark(
            lambda: gate * mx.sigmoid(gate) * up,
            args.warmup,
            args.iterations,
        ),
    )


def benchmark_attention(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> None:
    precision = model.model.embed_tokens.scales.dtype
    head_dim = model.args.head_dim
    num_heads = model.args.num_attention_heads
    num_kv_heads = model.args.num_key_value_heads
    query = mx.random.normal((1, num_heads, 1, head_dim)).astype(precision)
    key = mx.random.normal((1, num_kv_heads, args.context, head_dim)).astype(precision)
    value = mx.random.normal(key.shape).astype(precision)
    scale = head_dim**-0.5
    mx.eval(query, key, value)
    readable_us = benchmark(
        lambda: ops.readable_attention(query, key, value, scale, None),
        args.warmup,
        args.iterations,
    )
    optimized_us = benchmark(
        lambda: ops.decode_attention(query, key, value, scale, None),
        args.warmup,
        args.iterations,
    )
    mlx_us = benchmark(
        lambda: mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=None
        ),
        args.warmup,
        args.iterations,
    )
    report_progression("decode attention", readable_us, optimized_us, mlx_us)


def main() -> None:
    args = parse_args()
    if args.context <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError(
            "context and iterations must be positive; warmup cannot be negative"
        )
    ops = load_implementation(args.solution)
    model_name = shortcut_name_to_full_name(args.model)
    model, _ = load(model_name)
    print(
        f"Solution={ops.name} Model={model_name} context={args.context} "
        f"MLX={importlib.metadata.version('mlx')} "
        f"mlx-lm={importlib.metadata.version('mlx-lm')}"
    )
    print("Median synchronized latency; lower is better.")
    selected = set(args.section or SECTIONS)
    runners = {
        "embedding": benchmark_embedding,
        "decode-projections": benchmark_decode_projections,
        "prefill-projections": benchmark_prefill_projections,
        "model-kernels": benchmark_model_kernels,
        "attention": benchmark_attention,
    }
    for section in SECTIONS:
        if section in selected:
            runners[section](args, model, ops)


if __name__ == "__main__":
    main()
