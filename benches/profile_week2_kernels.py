import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

from benches import week2_profile_adapter as adapter


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
PROMPT_RULE = "seed-0-independent-operands-and-zero-token-ids"
PREFILL_LOGITS = "last"


ProfileCase = adapter.ProfileCase
load_implementation = adapter.load_implementation


def load_model(name):
    from mlx_lm import load
    return load(name)


def resolve_model(name):
    from model_names import shortcut_name_to_full_name
    return shortcut_name_to_full_name(name)


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
            "Replay public Week 2 operators at Qwen model shapes and API-call counts."
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
    parser.add_argument("--prefill-logits", choices=("last", "all"), default=PREFILL_LOGITS)
    parser.add_argument("--cache-capacity", type=int)
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
    if args.cache_capacity is not None and any(args.cache_capacity < case.tokens for case in args.case):
        parser.error("cache capacity must be at least every case context")
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


def workload_record(model, case, warmup, iterations, *, prefill_logits=None, cache_capacity=None):
    logits = PREFILL_LOGITS if prefill_logits is None else prefill_logits
    if logits not in ("last", "all"):
        raise ValueError("prefill_logits must be last or all")
    bound = case.tokens if cache_capacity is None else cache_capacity
    if type(bound) is not int or bound < case.tokens:
        raise ValueError("cache capacity must be at least context")
    return {
        "model": model, "phase": case.phase, "tokens": case.tokens,
        "query_length": case.tokens if case.phase == "prefill" else 1,
        "position_offset": 0 if case.phase == "prefill" else case.tokens - 1,
        "request_bound": bound, "prompt_rule": PROMPT_RULE, "fixture_seed": 0,
        "prefill_logits": logits, "warmup": warmup, "iterations": iterations,
        "profile_kind": adapter.PROFILE_KIND, "adapter_version": adapter.ADAPTER_VERSION,
        "rope_offset_preparation": adapter.OFFSET_PREPARATION,
    }


def build_decision(baseline, candidate, *, dominant_category, hypothesis,
                   observed_effect, decision, next_experiment):
    for record in (baseline, candidate):
        if record.get("schema_version") != 3 or not record.get("coverage", {}).get("complete"):
            raise ValueError("decision requires complete schema v3 public replay coverage")
    for field in ("source", "solution", "model", "workload_id", "software", "host", "profile_kind"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"decision {field} mismatch")
    if decision not in {"keep", "reject", "inconclusive"}:
        raise ValueError("decision must be keep, reject, or inconclusive")
    return {
        "schema_version": 3, "source": baseline["source"],
        "solution": baseline["solution"], "model": baseline["model"],
        "workload_id": baseline["workload_id"],
        "profile_kind": baseline["profile_kind"],
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "dominant_category": dominant_category, "hypothesis": hypothesis,
        "observed_effect": observed_effect, "decision": decision,
        "next_experiment": next_experiment,
    }


def evaluate(outputs):
    if outputs:
        import mlx.core as mx
        mx.eval(*outputs)


def benchmark_groups(builders, warmup, iterations, *, evaluator=None, clock=None):
    if warmup < 0 or iterations <= 0 or not builders:
        raise ValueError("require groups, nonnegative warmup and positive iterations")
    groups = tuple(item if isinstance(item, adapter.ReplayGroup)
                   else adapter.ReplayGroup(*item) for item in builders)
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("group names must be unique")
    evaluator = evaluate if evaluator is None else evaluator
    clock = perf_counter if clock is None else clock
    orders = [groups[offset:] + groups[:offset] for offset in range(len(groups))]
    timings = {group.name: [] for group in groups}
    for round_index in range(warmup + iterations):
        for group in orders[round_index % len(orders)]:
            outputs = None
            primary = None
            try:
                group.prepare()
                start = clock() if round_index >= warmup else None
                outputs = group.build()
                evaluator(outputs)
                elapsed = clock() - start if start is not None else None
                group.finish()
                if elapsed is not None:
                    timings[group.name].append(elapsed)
            except BaseException as error:
                primary = error
                raise
            finally:
                outputs = None
                try:
                    group.cleanup()
                except Exception as cleanup:
                    if primary is None:
                        raise
                    raise adapter.ReplayCleanupError(primary, cleanup) from primary
    return {name: median(samples) * 1_000_000 for name, samples in timings.items()}


def should_use_decode_attention(implementation, enabled, query_length, context_length, mask):
    return adapter.attention_path(
        rows=query_length, context=context_length, head_dim=1, dtype="float32", mask=mask,
        decode=enabled, max_query=implementation.decode_attention_max_query,
        max_context=implementation.decode_attention_max_context,
    ) == "custom"


def profile_case(implementation, mlx_model, model_name, case, warmup, iterations,
                 *, prefill_logits=None, cache_capacity=None):
    logits = PREFILL_LOGITS if prefill_logits is None else prefill_logits
    workload = workload_record(model_name, case, warmup, iterations,
                               prefill_logits=logits, cache_capacity=cache_capacity)
    common = {"schema_version": 3, "profile_kind": adapter.PROFILE_KIND,
              "checkpoint": case.checkpoint, "workload": workload,
              "workload_id": canonical_hash(workload)}
    try:
        replay = adapter.make_runtime_replay(implementation, mlx_model, case,
                                            prefill_logits=logits, cache_capacity=cache_capacity)
    except adapter.UnsupportedCoverage as error:
        return {**common, "coverage": {"complete": False, "reason": str(error)},
                "categories": []}
    timings = benchmark_groups(replay.groups(), warmup, iterations,
                               evaluator=replay.bindings.evaluate)
    total = sum(timings.values())
    if total <= 0:
        raise ValueError("replay total must be positive")
    return {
        **common, "coverage": {"complete": True},
        "shape": asdict(replay.plan.shape), "effective_features": dict(replay.plan.features),
        "cache": replay.cache.observation,
        "replay_total_us": total,
        "share_denominator": "sum-of-synchronized-public-group-medians",
        "rope_offset_preparation": adapter.OFFSET_PREPARATION,
        "categories": [{"name": name, "median_us": value, "share": value / total}
                       for name, value in timings.items()],
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
    model_name = resolve_model(args.model)
    mlx_model, _ = load_model(model_name)
    print(
        f"Solution={implementation.name} Model={model_name} "
        f"MLX={importlib.metadata.version('mlx')} "
        f"mlx-lm={importlib.metadata.version('mlx-lm')}"
    )
    print("Synchronized public-operator replay; shares are not a model timeline.")
    profiles = [
        profile_case(
            implementation,
            mlx_model,
            model_name,
            case,
            args.warmup,
            args.iterations,
            prefill_logits=args.prefill_logits, cache_capacity=args.cache_capacity,
        )
        for case in args.case
    ]
    for record in profiles:
        print(f"{record['checkpoint']} {record['workload']['phase']} tokens={record['workload']['tokens']}")
        if not record["coverage"]["complete"]:
            print(f"  unsupported: {record['coverage']['reason']}")
        for category in record["categories"]:
            print(f"  {category['name']}: {category['median_us']:.1f} us ({category['share']:.1%} of replay)")
    root = Path(__file__).resolve().parents[1]
    source = source_metadata(root)
    software = {
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "python": platform.python_version(),
    }
    import mlx.core as mx
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
        "schema_version": 3,
        "profile_kind": adapter.PROFILE_KIND,
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
        with args.json_output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
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
        with args.decision_output.open("x") as output:
            output.write(json.dumps(decision, indent=2) + "\n")


if __name__ == "__main__":
    main()
