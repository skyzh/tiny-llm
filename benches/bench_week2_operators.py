import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class BenchmarkComparison:
    medians_us: dict[str, float]
    samples_us: dict[str, list[float]]
    measurement_orders: list[list[str]]

    def __getitem__(self, name: str) -> float:
        return self.medians_us[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "medians_us": self.medians_us,
            "samples_us": self.samples_us,
            "measurement_orders": self.measurement_orders,
        }


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
    parser.add_argument(
        "--context",
        dest="contexts",
        type=int,
        action="append",
        help="context length; repeat to run a balanced context sweep (default: 128)",
    )
    parser.add_argument(
        "--context-repeats",
        type=int,
        help=(
            "number of full context/query-shape sweeps; defaults to 1 for one "
            "shape and 2 for multiple shapes"
        ),
    )
    parser.add_argument(
        "--query-length",
        dest="query_lengths",
        type=int,
        action="append",
        help=(
            "query rows for the attention section; repeat to run a balanced "
            "query-length sweep (default: 1)"
        ),
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=60,
        help=(
            "timed rounds per case; must be divisible by every implementation-"
            "order cycle (default: 60)"
        ),
    )
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
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optionally save metadata, execution order, raw samples, and medians",
    )
    args = parser.parse_args()
    args.contexts = args.contexts or [128]
    args.query_lengths = args.query_lengths or [1]
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("--context values must be unique")
    if any(context <= 0 for context in args.contexts):
        parser.error("--context values must be positive")
    if len(set(args.query_lengths)) != len(args.query_lengths):
        parser.error("--query-length values must be unique")
    if any(query_length <= 0 for query_length in args.query_lengths):
        parser.error("--query-length values must be positive")
    shape_count = len(args.contexts) * len(args.query_lengths)
    if args.context_repeats is None:
        args.context_repeats = 1 if shape_count == 1 else 2
    if args.context_repeats <= 0:
        parser.error("--context-repeats must be positive")
    if shape_count > 1 and args.context_repeats % 2 != 0:
        parser.error(
            "--context-repeats must be even for a multi-shape sweep so "
            "forward and reverse context/query orders are balanced"
        )
    if len(args.query_lengths) > 1 and set(args.section or SECTIONS) != {"attention"}:
        parser.error("a --query-length sweep requires --section attention")
    return args


def context_execution_order(contexts: list[int], repeats: int) -> list[list[int]]:
    return [
        [context for context, _ in order]
        for order in shape_execution_order(contexts, [1], repeats)
    ]


def shape_execution_order(
    contexts: list[int], query_lengths: list[int], repeats: int
) -> list[list[tuple[int, int]]]:
    shapes = [
        (context, query_length)
        for context in contexts
        for query_length in query_lengths
    ]
    return [
        list(shapes if repeat % 2 == 0 else reversed(shapes))
        for repeat in range(repeats)
    ]


def benchmark_comparison(
    functions: list[tuple[str, Callable[[], mx.array]]],
    warmup: int,
    iterations: int,
) -> BenchmarkComparison:
    orders = list(permutations(functions))
    if iterations % len(orders) != 0:
        raise ValueError(
            f"iterations must be divisible by {len(orders)} to balance every "
            "implementation order"
        )
    for round_index in range(warmup):
        for _, function in orders[round_index % len(orders)]:
            mx.eval(function())

    timings = {name: [] for name, _ in functions}
    measurement_orders = []
    for round_index in range(iterations):
        order = orders[(warmup + round_index) % len(orders)]
        measurement_orders.append([name for name, _ in order])
        for name, function in order:
            start = perf_counter()
            mx.eval(function())
            timings[name].append((perf_counter() - start) * 1_000_000)
    return BenchmarkComparison(
        medians_us={name: median(samples) for name, samples in timings.items()},
        samples_us=timings,
        measurement_orders=measurement_orders,
    )


def comparison_record(name: str, result: BenchmarkComparison) -> dict[str, Any]:
    return {"name": name, **result.as_dict()}


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
) -> list[dict[str, Any]]:
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
    return [comparison_record("quantized embedding", timings)]


def benchmark_decode_projections(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> list[dict[str, Any]]:
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
    results = []
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
        results.append(comparison_record(name, timings))
    return results


def benchmark_prefill_projections(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> list[dict[str, Any]]:
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
    results = []
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
        results.append(comparison_record(name, timings))
    return results


def benchmark_model_kernels(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> list[dict[str, Any]]:
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
    results = []
    report_progression(
        "RMSNorm",
        timings["readable"],
        timings["optimized"],
        timings["mlx"],
    )
    results.append(comparison_record("RMSNorm", timings))

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
    results.append(comparison_record("RoPE", timings))

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
    results.append(comparison_record("SwiGLU", timings))
    return results


def benchmark_attention(
    args: argparse.Namespace, model: Any, ops: OperatorImplementation
) -> list[dict[str, Any]]:
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
    return [comparison_record("decode attention", timings)]


def _run_text(command: list[str], cwd: Path | None = None) -> str | None:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def collect_metadata(
    root: Path, requested_model: str, resolved_model: str, model: Any
) -> dict[str, Any]:
    model_args = {}
    if is_dataclass(model.args):
        model_args = {
            field.name: _jsonable(getattr(model.args, field.name))
            for field in fields(model.args)
        }
    else:
        for name in (
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
        ):
            if hasattr(model.args, name):
                model_args[name] = _jsonable(getattr(model.args, name))

    hardware = None
    if sys.platform == "darwin":
        profile_text = _run_text(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"]
        )
        if profile_text:
            profile = json.loads(profile_text)
            system = profile.get("SPHardwareDataType", [{}])[0]
            display = profile.get("SPDisplaysDataType", [{}])[0]
            hardware = {
                "machine_name": system.get("machine_name"),
                "machine_model": system.get("machine_model"),
                "chip_type": system.get("chip_type"),
                "cpu_cores": system.get("number_processors"),
                "gpu_model": display.get("sppci_model"),
                "gpu_cores": display.get("sppci_cores"),
                "physical_memory": system.get("physical_memory"),
            }

    git_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    git_status = _run_text(["git", "status", "--porcelain"], cwd=root)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": git_commit,
            "git_dirty": bool(git_status),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hardware": hardware,
            "mlx_device": _jsonable(mx.device_info()),
        },
        "software": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "numpy_version": importlib.metadata.version("numpy"),
            "xcode_version": _run_text(["xcodebuild", "-version"]),
            "metal_version": _run_text(["xcrun", "metal", "--version"]),
        },
        "model": {
            "requested": requested_model,
            "resolved": resolved_model,
            "configuration": model_args,
            "weight_dtype": str(model.model.embed_tokens.weight.dtype),
            "scale_dtype": str(model.model.embed_tokens.scales.dtype),
            "quantization": {
                "group_size": getattr(model.model.embed_tokens, "group_size", None),
                "bits": getattr(model.model.embed_tokens, "bits", None),
            },
        },
    }


def summarize_runs(
    runs: list[dict[str, Any]], contexts: list[int], query_lengths: list[int]
) -> list[dict[str, Any]]:
    summary = []
    for context in contexts:
        for query_length in query_lengths:
            section_samples: dict[str, dict[str, dict[str, list[float]]]] = {}
            for run in runs:
                if run["context"] != context or run["query_length"] != query_length:
                    continue
                for section, comparisons in run["sections"].items():
                    section_cases = section_samples.setdefault(section, {})
                    for comparison in comparisons:
                        case = section_cases.setdefault(comparison["name"], {})
                        for implementation, samples in comparison["samples_us"].items():
                            case.setdefault(implementation, []).extend(samples)
            summary.append(
                {
                    "context": context,
                    "query_length": query_length,
                    "sections": {
                        section: [
                            {
                                "name": name,
                                "medians_us": {
                                    implementation: median(samples)
                                    for implementation, samples in implementations.items()
                                },
                            }
                            for name, implementations in cases.items()
                        ]
                        for section, cases in section_samples.items()
                    },
                }
            )
    return summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nAggregated medians across balanced context/query sweeps:")
    for shape_result in summary:
        print(
            f"context={shape_result['context']} "
            f"query_length={shape_result['query_length']}"
        )
        for section, comparisons in shape_result["sections"].items():
            for comparison in comparisons:
                medians = " ".join(
                    f"{name}={value:.1f} us"
                    for name, value in comparison["medians_us"].items()
                )
                print(f"  {section}/{comparison['name']}: {medians}")


def main() -> None:
    args = parse_args()
    if (
        args.warmup < 0
        or args.iterations <= 0
        or (args.gqa_ratio is not None and args.gqa_ratio <= 0)
    ):
        raise ValueError(
            "gqa-ratio and iterations must be positive; warmup cannot be negative"
        )
    ops = load_implementation(args.solution)
    model_name = shortcut_name_to_full_name(args.model)
    model, _ = load(model_name)
    root = Path(__file__).resolve().parents[1]
    shape_orders = shape_execution_order(
        args.contexts, args.query_lengths, args.context_repeats
    )
    print(
        f"Solution={ops.name} Model={model_name} contexts={args.contexts} "
        f"context_repeats={args.context_repeats} "
        f"query_lengths={args.query_lengths} "
        f"gqa_ratio={args.gqa_ratio or 'model'} "
        f"attention_mask={args.attention_mask} "
        f"MLX={importlib.metadata.version('mlx')} "
        f"mlx-lm={importlib.metadata.version('mlx-lm')}"
    )
    print(
        "Median synchronized latency with rotated implementation order and "
        "forward/reverse context/query order; lower is better."
    )
    selected = set(args.section or SECTIONS)
    runners = {
        "embedding": benchmark_embedding,
        "decode-projections": benchmark_decode_projections,
        "prefill-projections": benchmark_prefill_projections,
        "model-kernels": benchmark_model_kernels,
        "attention": benchmark_attention,
    }
    runs = []
    for repeat_index, shape_order in enumerate(shape_orders):
        for shape_position, (context, query_length) in enumerate(shape_order):
            print(
                f"\ncontext repeat={repeat_index + 1}/{args.context_repeats} "
                f"position={shape_position + 1}/{len(shape_order)} "
                f"tokens={context} query_length={query_length}"
            )
            run_args = argparse.Namespace(
                **vars(args), context=context, query_length=query_length
            )
            sections = {}
            for section in SECTIONS:
                if section in selected:
                    sections[section] = runners[section](run_args, model, ops)
            runs.append(
                {
                    "context_repeat": repeat_index,
                    "shape_position": shape_position,
                    "context": context,
                    "query_length": query_length,
                    "sections": sections,
                }
            )

    summary = summarize_runs(runs, args.contexts, args.query_lengths)
    print_summary(summary)
    if args.json_output is not None:
        payload = {
            "schema_version": 2,
            "metadata": collect_metadata(root, args.model, model_name, model),
            "configuration": {
                "solution": args.solution,
                "sections": [section for section in SECTIONS if section in selected],
                "contexts": args.contexts,
                "query_lengths": args.query_lengths,
                "context_repeats": args.context_repeats,
                "shape_execution_order": [
                    [
                        {"context": context, "query_length": query_length}
                        for context, query_length in order
                    ]
                    for order in shape_orders
                ],
                "warmup": args.warmup,
                "iterations": args.iterations,
                "gqa_ratio": args.gqa_ratio,
                "attention_mask": args.attention_mask,
                "prefill_projections": args.prefill_projection or ["q"],
                "include_split_k": args.include_split_k,
            },
            "runs": runs,
            "summary": summary,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
