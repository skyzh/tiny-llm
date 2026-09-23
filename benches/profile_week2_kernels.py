import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
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


DEFAULT_CASES = (
    "kv-cache:decode:128",
    "capacity-cache:decode:128",
    "quantized-matvec:decode:128",
    "simd-matmul:prefill:128",
    "rmsnorm:decode:128",
    "rope:decode:128",
    "swiglu:decode:128",
    "tiled-prefill:prefill:128",
    "selected:prefill:128",
)
PROMPT_RULE = "synthetic-token-ids"
PREFILL_LOGITS = "all"


@dataclass(frozen=True)
class ProfileCase:
    checkpoint: str
    phase: str
    tokens: int


@dataclass(frozen=True)
class CategoryResult:
    name: str
    median_us: float
    share: float


@dataclass(frozen=True)
class KernelImplementation:
    name: str
    model_type: Any
    checkpoints: tuple[str, ...]
    quantized_weights_type: Any
    grouped_attention: Callable[..., mx.array]
    linear: Callable[..., mx.array]
    silu: Callable[[mx.array], mx.array]
    quantized_linear: Callable[..., mx.array]
    decode_attention: Callable[..., mx.array]
    swiglu: Callable[[mx.array, mx.array], mx.array]
    decode_attention_max_query: int
    decode_attention_max_context: int


def load_implementation(name: str) -> KernelImplementation:
    attention = importlib.import_module(f"{name}.attention")
    basics = importlib.import_module(f"{name}.basics")
    model = importlib.import_module(f"{name}.qwen3_week2")
    quantize = importlib.import_module(f"{name}.quantize")
    kernels = importlib.import_module(f"{name}.week2_kernels")
    return KernelImplementation(
        name=name,
        model_type=model.Qwen3ModelWeek2,
        checkpoints=tuple(model.WEEK2_CHECKPOINTS),
        quantized_weights_type=quantize.QuantizedWeights,
        grouped_attention=attention.scaled_dot_product_attention_grouped,
        linear=basics.linear,
        silu=basics.silu,
        quantized_linear=quantize.quantized_linear,
        decode_attention=kernels.decode_attention_custom,
        swiglu=kernels.swiglu,
        decode_attention_max_query=getattr(model, "DECODE_ATTENTION_MAX_QUERY", 0),
        decode_attention_max_context=getattr(model, "DECODE_ATTENTION_MAX_CONTEXT", 0),
    )


def parse_case(value: str) -> ProfileCase:
    try:
        checkpoint, phase, raw_tokens = value.split(":")
        tokens = int(raw_tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cases use CHECKPOINT:PHASE:TOKENS") from exc
    if phase not in ("decode", "prefill"):
        raise argparse.ArgumentTypeError("phase must be decode or prefill")
    if tokens <= 0:
        raise argparse.ArgumentTypeError("tokens must be positive")
    return ProfileCase(checkpoint, phase, tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute Week 2 time by replaying each real kernel group at its "
            "Qwen model shape and dispatch count."
        )
    )
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--solution",
        choices=("tiny_llm", "tiny_llm_ref"),
        required=True,
        help="implementation to profile; learner evidence uses tiny_llm",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help=(
            "profile CHECKPOINT:PHASE:TOKENS; repeat for more cases "
            "(default: the Week 2 bottleneck progression)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--decision-output", type=Path)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--dominant-category")
    parser.add_argument("--hypothesis")
    parser.add_argument("--observed-effect")
    parser.add_argument("--decision", choices=("keep", "reject", "inconclusive"))
    parser.add_argument("--next-experiment")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup cannot be negative and iterations must be positive")
    if args.case is None:
        args.case = [parse_case(value) for value in DEFAULT_CASES]
    decision_fields = (
        args.baseline_checkpoint,
        args.candidate_checkpoint,
        args.dominant_category,
        args.hypothesis,
        args.observed_effect,
        args.decision,
        args.next_experiment,
    )
    if args.decision_output is not None and not all(decision_fields):
        parser.error("--decision-output requires every decision-record field")
    if args.decision_output is None and any(decision_fields):
        parser.error("decision-record fields require --decision-output")
    return args


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_output_paths(
    json_output: Path | None, decision_output: Path | None
) -> tuple[Path | None, Path | None]:
    normalized = tuple(
        path.expanduser().resolve(strict=False) if path is not None else None
        for path in (json_output, decision_output)
    )
    present = [path for path in normalized if path is not None]
    if len(set(present)) != len(present):
        raise ValueError("output paths must be distinct")
    return normalized


def source_metadata(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def workload_record(
    model: str, case: ProfileCase, warmup: int, iterations: int
) -> dict[str, object]:
    return {
        "model": model,
        "phase": case.phase,
        "tokens": case.tokens,
        "prompt_rule": PROMPT_RULE,
        "prefill_logits": PREFILL_LOGITS,
        "warmup": warmup,
        "iterations": iterations,
    }


def build_decision(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    dominant_category: str,
    hypothesis: str,
    observed_effect: str,
    decision: str,
    next_experiment: str,
) -> dict[str, object]:
    for field in ("source", "solution", "model", "workload_id"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"decision {field} mismatch")
    if decision not in {"keep", "reject", "inconclusive"}:
        raise ValueError("decision must be keep, reject, or inconclusive")
    return {
        "schema_version": 1,
        "source": baseline["source"],
        "solution": baseline["solution"],
        "model": baseline["model"],
        "workload_id": baseline["workload_id"],
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "dominant_category": dominant_category,
        "hypothesis": hypothesis,
        "observed_effect": observed_effect,
        "decision": decision,
        "next_experiment": next_experiment,
    }


def evaluate(outputs: list[mx.array]) -> None:
    mx.eval(*outputs)


def benchmark_groups(
    builders: tuple[tuple[str, Callable[[], list[mx.array]]], ...],
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    orders = [builders[offset:] + builders[:offset] for offset in range(len(builders))]
    for round_index in range(warmup):
        for _, build in orders[round_index % len(orders)]:
            evaluate(build())

    timings = {name: [] for name, _ in builders}
    for round_index in range(iterations):
        order = orders[(warmup + round_index) % len(orders)]
        for name, build in order:
            start = perf_counter()
            evaluate(build())
            timings[name].append(perf_counter() - start)
    return {name: median(samples) * 1_000_000 for name, samples in timings.items()}


def project(implementation: KernelImplementation, x: mx.array, weight: Any) -> mx.array:
    if isinstance(weight, implementation.quantized_weights_type):
        return implementation.quantized_linear(x, weight)
    return implementation.linear(x, weight)


def should_use_decode_attention(
    implementation: KernelImplementation,
    enabled: bool,
    query_length: int,
    context_length: int,
    mask: mx.array | str | None,
) -> bool:
    return (
        enabled
        and query_length <= implementation.decode_attention_max_query
        and context_length <= implementation.decode_attention_max_context
        and not isinstance(mask, mx.array)
    )


class KernelReplay:
    def __init__(
        self,
        implementation: KernelImplementation,
        model: Any,
        phase: str,
        tokens: int,
    ):
        self.implementation = implementation
        self.model = model
        self.phase = phase
        self.rows = 1 if phase == "decode" else tokens
        self.context = tokens
        first = model.layers_inner[0]
        self.hidden_size = first.hidden_size
        self.num_heads = first.num_attention_heads
        self.num_kv_heads = first.self_attn.num_kv_heads
        self.head_dim = first.self_attn.head_dim
        self.intermediate_size = first.mlp.hidden_dim
        self.dtype = model.precision

        self.hidden = mx.random.normal((1, self.rows, self.hidden_size)).astype(
            self.dtype
        )
        self.query = mx.random.normal(
            (1, self.num_heads, self.rows, self.head_dim)
        ).astype(self.dtype)
        self.key = mx.random.normal(
            (1, self.num_kv_heads, self.context, self.head_dim)
        ).astype(self.dtype)
        self.value = mx.random.normal(self.key.shape).astype(self.dtype)
        self.query_rows = self.query.transpose(0, 2, 1, 3)
        self.key_rows = mx.random.normal(
            (1, self.rows, self.num_kv_heads, self.head_dim)
        ).astype(self.dtype)
        self.gate = mx.random.normal((1, self.rows, self.intermediate_size)).astype(
            self.dtype
        )
        self.up = mx.random.normal(self.gate.shape).astype(self.dtype)
        self.tokens = mx.zeros((1, self.rows), dtype=mx.int32)
        evaluate(
            [
                self.hidden,
                self.query,
                self.key,
                self.value,
                self.query_rows,
                self.key_rows,
                self.gate,
                self.up,
                self.tokens,
            ]
        )

    def projections(self) -> list[mx.array]:
        outputs = []
        hidden = self.hidden
        for layer in self.model.layers_inner:
            attention = layer.self_attn
            query = project(self.implementation, hidden, attention.wq)
            key = project(self.implementation, hidden, attention.wk)
            value = project(self.implementation, hidden, attention.wv)
            attention_input = mx.concatenate(
                (key, value, query[..., key.shape[-1] + value.shape[-1] :]),
                axis=-1,
            )
            attention_output = project(
                self.implementation, attention_input, attention.wo
            )
            mlp_input = hidden + attention_output
            gate = project(self.implementation, mlp_input, layer.mlp.w_gate)
            up = project(self.implementation, mlp_input, layer.mlp.w_up)
            mlp_output = project(self.implementation, gate + up, layer.mlp.w_down)
            hidden = mlp_input + mlp_output
            outputs.extend((key, value))
        final_hidden = hidden[:, -1:, :]
        if self.model.w_lm_head is not None:
            outputs.append(
                project(self.implementation, final_hidden, self.model.w_lm_head)
            )
        else:
            outputs.append(self.model.embedding.as_linear(final_hidden))
        return outputs

    def attention(self) -> list[mx.array]:
        outputs = []
        mask = "causal" if self.phase == "prefill" else None
        for layer in self.model.layers_inner:
            attention = layer.self_attn
            if should_use_decode_attention(
                self.implementation,
                attention.use_decode_attention,
                self.rows,
                self.context,
                mask,
            ):
                output = self.implementation.decode_attention(
                    self.query,
                    self.key,
                    self.value,
                    scale=attention.scale,
                    mask=mask,
                )
            else:
                output = self.implementation.grouped_attention(
                    self.query.astype(mx.float32),
                    self.key.astype(mx.float32),
                    self.value.astype(mx.float32),
                    scale=attention.scale,
                    mask=mask,
                ).astype(self.dtype)
            outputs.append(output)
        return outputs

    def pointwise(self) -> list[mx.array]:
        outputs = [self.model.embedding(self.tokens)]
        for layer in self.model.layers_inner:
            attention = layer.self_attn
            outputs.extend(
                (
                    layer.input_layernorm(self.hidden),
                    layer.post_attention_layernorm(self.hidden),
                    attention.q_norm(self.query_rows),
                    attention.k_norm(self.key_rows),
                )
            )
            rope_offset = 0 if attention.use_fast_rope else slice(0, self.rows)
            outputs.extend(
                (
                    attention.rope(self.query_rows, offset=rope_offset),
                    attention.rope(self.key_rows, offset=rope_offset),
                )
            )
            if layer.mlp.use_fast_swiglu:
                outputs.append(self.implementation.swiglu(self.gate, self.up))
            else:
                outputs.append(self.implementation.silu(self.gate) * self.up)
            outputs.extend((self.hidden + self.hidden, self.hidden + self.hidden))
        outputs.append(self.model.norm(self.hidden[:, -1:, :]))
        return outputs

    def cache(self) -> list[mx.array]:
        if self.phase == "prefill":
            return [self.key, self.value]
        previous_key = self.key[:, :, :-1, :]
        previous_value = self.value[:, :, :-1, :]
        new_key = self.key[:, :, -1:, :]
        new_value = self.value[:, :, -1:, :]
        outputs = []
        for _ in self.model.layers_inner:
            outputs.extend(
                (
                    mx.concat((previous_key, new_key), axis=2),
                    mx.concat((previous_value, new_value), axis=2),
                )
            )
        return outputs


def profile_case(
    implementation: KernelImplementation,
    mlx_model: object,
    model_name: str,
    case: ProfileCase,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    model = implementation.model_type(mlx_model, checkpoint=case.checkpoint)
    replay = KernelReplay(implementation, model, case.phase, case.tokens)
    builders = (
        ("projections", replay.projections),
        ("attention", replay.attention),
        ("normalization, position, and activation", replay.pointwise),
    )
    if case.phase == "decode":
        builders += (("KV growth", replay.cache),)
    timings = benchmark_groups(builders, warmup, iterations)
    measured = [(name, timings[name]) for name, _ in builders]
    total = sum(value for _, value in measured)
    categories = [
        CategoryResult(name, value, value / total) for name, value in measured
    ]
    print(f"{case.checkpoint:<18} {case.phase:<7} tokens={case.tokens:<4}")
    for category in categories:
        print(
            f"  {category.name:<40} {category.median_us:>10.1f} us "
            f"{category.share:>6.1%}"
        )
    workload = workload_record(model_name, case, warmup, iterations)
    return {
        "checkpoint": case.checkpoint,
        "workload": workload,
        "workload_id": canonical_hash(workload),
        "attributed_us": total,
        "categories": [asdict(category) for category in categories],
    }


def main() -> None:
    args = parse_args()
    raw_outputs = [path for path in (args.json_output, args.decision_output) if path]
    existing = [path for path in raw_outputs if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    args.json_output, args.decision_output = normalize_output_paths(
        args.json_output, args.decision_output
    )
    outputs = [path for path in (args.json_output, args.decision_output) if path]
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    implementation = load_implementation(args.solution)
    unknown = [
        case.checkpoint
        for case in args.case
        if case.checkpoint not in implementation.checkpoints
    ]
    if unknown:
        raise ValueError(f"unknown Week 2 checkpoint for {args.solution}: {unknown[0]}")
    model_name = shortcut_name_to_full_name(args.model)
    mlx_model, _ = load(model_name)
    print(
        f"Solution={implementation.name} Model={model_name} "
        f"MLX={importlib.metadata.version('mlx')} "
        f"mlx-lm={importlib.metadata.version('mlx-lm')}"
    )
    print(
        "Median synchronized kernel-group replay; shares are normalized across "
        "the measured groups."
    )
    profiles = [
        profile_case(
            implementation,
            mlx_model,
            model_name,
            case,
            args.warmup,
            args.iterations,
        )
        for case in args.case
    ]
    root = Path(__file__).resolve().parents[1]
    source = source_metadata(root)
    software = {
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "python": platform.python_version(),
    }
    host = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": mx.device_info(),
    }
    for profile in profiles:
        profile.update(
            source=source,
            solution=implementation.name,
            model=model_name,
            software=software,
            host=host,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    result = {
        "schema_version": 2,
        "source": source,
        "solution": implementation.name,
        "model": model_name,
        "software": software,
        "host": host,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profiles": profiles,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.decision_output is not None:
        matches = {
            profile["checkpoint"]: profile
            for profile in profiles
            if profile["checkpoint"]
            in {args.baseline_checkpoint, args.candidate_checkpoint}
        }
        missing = {
            args.baseline_checkpoint,
            args.candidate_checkpoint,
        } - matches.keys()
        if missing:
            raise ValueError(
                f"decision checkpoint was not profiled: {sorted(missing)[0]}"
            )
        decision = build_decision(
            matches[args.baseline_checkpoint],
            matches[args.candidate_checkpoint],
            dominant_category=args.dominant_category,
            hypothesis=args.hypothesis,
            observed_effect=args.observed_effect,
            decision=args.decision,
            next_experiment=args.next_experiment,
        )
        args.decision_output.parent.mkdir(parents=True, exist_ok=True)
        args.decision_output.write_text(json.dumps(decision, indent=2) + "\n")


if __name__ == "__main__":
    main()
