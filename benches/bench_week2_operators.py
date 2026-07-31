import argparse
import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from itertools import permutations
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
    parser.add_argument(
        "--query-length",
        type=int,
        default=1,
        help="query rows for the attention section (default: 1)",
    )
    parser.add_argument(
        "--gqa-ratio",
        type=int,
        help=(
            "query heads per KV head for the attention section "
            "(default: model configuration)"
        ),
    )
    parser.add_argument(
        "--attention-mask",
        choices=("none", "causal", "explicit"),
        default="none",
        help="mask contract for the attention section (default: none)",
    )
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=60)
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


def benchmark_comparison(
    functions: list[tuple[str, Callable[[], mx.array]]],
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    orders = list(permutations(functions))
    for round_index in range(warmup):
        for _, function in orders[round_index % len(orders)]:
            mx.eval(function())

    timings = {name: [] for name, _ in functions}
    for round_index in range(iterations):
        order = orders[(warmup + round_index) % len(orders)]
        for name, function in order:
            start = perf_counter()
            mx.eval(function())
            timings[name].append(perf_counter() - start)
    return {name: median(samples) * 1_000_000 for name, samples in timings.items()}


def report(name: str, course_us: float, mlx_us: float) -> None:
    relative = course_us / mlx_us
    print(
        f"{name:<22} course={course_us:>9.1f} us  "
        f"mlx={mlx_us:>9.1f} us  latency={relative:>5.2f}x"
    )


def report_progression(
    name: str,
    baseline_us: float,
    optimized_us: float,
    mlx_us: float,
    *,
    baseline_label: str = "readable",
) -> None:
    print(
        f"{name:<22} {baseline_label}={baseline_us:>9.1f} us  "
        f"optimized={optimized_us:>9.1f} us  mlx={mlx_us:>9.1f} us  "
        f"speedup={baseline_us / optimized_us:>5.2f}x"
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
    timings = benchmark_comparison(
        [
            ("readable", lambda: dense_embedding(token)),
            ("optimized", lambda: embedding(token)),
            ("mlx", lambda: model.model.embed_tokens(token)),
        ],
        args.warmup,
        args.iterations,
    )
    report_progression(
        "quantized embedding",
        timings["readable"],
        timings["optimized"],
        timings["mlx"],
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
        timings = benchmark_comparison(
            [
                (
                    "vanilla",
                    lambda x=x, weights=weights: ops.quantized_matmul_vanilla(
                        weights.scales,
                        weights.biases,
                        weights.group_size,
                        weights.bits,
                        x,
                        weights.weight,
                        True,
                    ),
                ),
                (
                    "optimized",
                    lambda x=x, weights=weights: ops.quantized_linear(x, weights),
                ),
                (
                    "mlx",
                    lambda x=x, weights=weights: mx.quantized_matmul(
                        x,
                        weights.weight,
                        weights.scales,
                        weights.biases,
                        transpose=True,
                        group_size=weights.group_size,
                        bits=weights.bits,
                    ),
                ),
            ],
            args.warmup,
            args.iterations,
        )
        report_progression(
            name,
            timings["vanilla"],
            timings["optimized"],
            timings["mlx"],
            baseline_label="vanilla",
        )


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
        functions = [
            (
                "simd",
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
            ),
            (
                "mlx",
                lambda: mx.quantized_matmul(
                    x,
                    weights.weight,
                    weights.scales,
                    weights.biases,
                    transpose=True,
                    group_size=weights.group_size,
                    bits=weights.bits,
                ),
            ),
        ]
        if args.include_split_k:
            functions.append(
                (
                    "split-k",
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
                )
            )
        timings = benchmark_comparison(
            functions,
            args.warmup,
            args.iterations,
        )
        name = f"prefill {projection} matmul"
        if args.include_split_k:
            report_split_k(name, timings["simd"], timings["split-k"], timings["mlx"])
        else:
            report(name, timings["simd"], timings["mlx"])


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
    timings = benchmark_comparison(
        [
            ("readable", lambda: readable_rms(x_norm)),
            ("optimized", lambda: rms(x_norm)),
            (
                "mlx",
                lambda: mx.fast.rms_norm(
                    x_norm, layer.input_layernorm.weight, model.args.rms_norm_eps
                ),
            ),
        ],
        args.warmup,
        args.iterations,
    )
    report_progression(
        "RMSNorm",
        timings["readable"],
        timings["optimized"],
        timings["mlx"],
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
    timings = benchmark_comparison(
        [
            ("readable", lambda: readable_rope(x_rope, slice(17, 18))),
            ("optimized", lambda: rope(x_rope, 17)),
            (
                "mlx",
                lambda: mx.fast.rope(
                    x_rope_mlx,
                    head_dim,
                    traditional=False,
                    base=model.args.rope_theta,
                    scale=1.0,
                    offset=17,
                ).transpose(0, 2, 1, 3),
            ),
        ],
        args.warmup,
        args.iterations,
    )
    report_progression(
        "RoPE",
        timings["readable"],
        timings["optimized"],
        timings["mlx"],
    )

    gate = mx.random.normal((1, 1, model.args.intermediate_size)).astype(precision)
    up = mx.random.normal(gate.shape).astype(precision)
    mx.eval(gate, up)
    timings = benchmark_comparison(
        [
            ("readable", lambda: ops.silu(gate) * up),
            ("optimized", lambda: ops.swiglu(gate, up)),
            ("mlx", lambda: gate * mx.sigmoid(gate) * up),
        ],
        args.warmup,
        args.iterations,
    )
    report_progression(
        "SwiGLU",
        timings["readable"],
        timings["optimized"],
        timings["mlx"],
    )


def benchmark_attention(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> None:
    precision = model.model.embed_tokens.scales.dtype
    head_dim = model.args.head_dim
    num_heads = model.args.num_attention_heads
    gqa_ratio = args.gqa_ratio or num_heads // model.args.num_key_value_heads
    if num_heads % gqa_ratio != 0:
        raise ValueError(
            f"gqa-ratio {gqa_ratio} must divide the model's {num_heads} query heads"
        )
    if args.attention_mask == "causal" and args.context < args.query_length:
        raise ValueError("causal attention requires context >= query-length")
    num_kv_heads = num_heads // gqa_ratio
    query = mx.random.normal((1, num_heads, args.query_length, head_dim)).astype(
        precision
    )
    key = mx.random.normal((1, num_kv_heads, args.context, head_dim)).astype(precision)
    value = mx.random.normal(key.shape).astype(precision)
    scale = head_dim**-0.5
    mask: mx.array | str | None = None
    if args.attention_mask == "causal":
        mask = "causal"
    elif args.attention_mask == "explicit":
        mask = mx.where(
            mx.arange(args.context) % 5 == 0,
            mx.array(-2.0, dtype=precision),
            mx.array(0.0, dtype=precision),
        ).reshape(1, 1, 1, args.context)
    arrays = [query, key, value]
    if isinstance(mask, mx.array):
        arrays.append(mask)
    mx.eval(*arrays)
    timings = benchmark_comparison(
        [
            (
                "readable",
                lambda: ops.readable_attention(
                    query.astype(mx.float32),
                    key.astype(mx.float32),
                    value.astype(mx.float32),
                    scale,
                    mask,
                ).astype(precision),
            ),
            (
                "optimized",
                lambda: ops.decode_attention(query, key, value, scale, mask),
            ),
            (
                "mlx",
                lambda: mx.fast.scaled_dot_product_attention(
                    query, key, value, scale=scale, mask=mask
                ),
            ),
        ],
        args.warmup,
        args.iterations,
    )
    report_progression(
        "decode attention",
        timings["readable"],
        timings["optimized"],
        timings["mlx"],
    )


def main() -> None:
    args = parse_args()
    if (
        args.context <= 0
        or args.query_length <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or (args.gqa_ratio is not None and args.gqa_ratio <= 0)
    ):
        raise ValueError(
            "context, query-length, gqa-ratio, and iterations must be positive; "
            "warmup cannot be negative"
        )
    ops = load_implementation(args.solution)
    model_name = shortcut_name_to_full_name(args.model)
    model, _ = load(model_name)
    print(
        f"Solution={ops.name} Model={model_name} context={args.context} "
        f"query_length={args.query_length} "
        f"gqa_ratio={args.gqa_ratio or 'model'} "
        f"attention_mask={args.attention_mask} "
        f"MLX={importlib.metadata.version('mlx')} "
        f"mlx-lm={importlib.metadata.version('mlx-lm')}"
    )
    print(
        "Median synchronized latency with rotated implementation order; "
        "lower is better."
    )
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
