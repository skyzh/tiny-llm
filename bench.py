import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from time import perf_counter

import mlx.core as mx
from mlx_lm import load
from tqdm.auto import tqdm

from model_names import shortcut_name_to_full_name


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


@dataclass
class BatchRequestState:
    """Per-request state while it moves from prefill into a decode batch."""

    request: BenchRequest
    kv_cache: list
    offset: int = 0
    generated_tokens: int = 0
    next_token: int | None = None
    is_prefill_done: bool = False


@dataclass
class ServingMetrics:
    generated_tokens: int = 0
    decode_tokens: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    peak_active_requests: int = 0
    peak_live_pages: int = 0
    peak_capacity_pages: int = 0
    peak_tail_waste_slots: int = 0
    peak_kv_bytes: int = 0
    reused_page_allocations: int = 0
    storage_growths: int = 0
    copied_pages_on_growth: int = 0
    paged_growth_copy_bytes: int = 0
    dense_growth_copy_bytes: int = 0
    dense_staging_copy_bytes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tiny-llm token throughput with synthetic token IDs."
    )
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--solution", type=str, default="tiny_llm")
    parser.add_argument(
        "--loader",
        type=str,
        default="week2",
        choices=["week1", "week2", "week3"],
    )
    parser.add_argument(
        "--disable-paged-attention",
        action="store_true",
        help="run the Week 3 Day 4 dense-gather compatibility checkpoint",
    )
    parser.add_argument(
        "--week2-checkpoint",
        choices=(
            "kv-cache",
            "quantized-matvec",
            "decode-attention",
            "rmsnorm",
            "rope",
            "swiglu",
            "simd-matmul",
            "split-k",
        ),
        help="run one cumulative Week 2 end-to-end checkpoint",
    )
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--min-output-len", type=int, default=64)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--prefill-logits",
        choices=("all", "last"),
        default="all",
        help=(
            "compute logits for every prompt position for prompt-scoring "
            "comparisons, or only the final row for serving"
        ),
    )
    parser.add_argument(
        "--batch-decode",
        action="store_true",
        help="Run the Week 3 continuous-batching serving benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--prefill-step",
        type=int,
        default=128,
        help="Maximum number of prompt tokens to prefill per scheduler step.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally save benchmark configuration and metrics as JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_seqs <= 0:
        raise ValueError("--num-seqs must be > 0")
    if args.min_input_len <= 0 or args.max_input_len <= 0:
        raise ValueError("input lengths must be > 0")
    if args.min_output_len <= 0 or args.max_output_len <= 0:
        raise ValueError("output lengths must be > 0")
    if args.min_input_len > args.max_input_len:
        raise ValueError("--min-input-len cannot be greater than --max-input-len")
    if args.min_output_len > args.max_output_len:
        raise ValueError("--min-output-len cannot be greater than --max-output-len")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.batch_decode and args.loader == "week1":
        raise ValueError("--batch-decode requires --loader week2 or week3")
    if args.prefill_logits == "last" and args.loader == "week1":
        raise ValueError("--prefill-logits last requires --loader week2 or week3")
    if args.batch_decode and args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.batch_decode and args.prefill_step <= 0:
        raise ValueError("--prefill-step must be > 0")
    if args.batch_decode and args.num_seqs < args.batch_size:
        raise ValueError("--batch-decode requires --num-seqs >= --batch-size")
    if args.week2_checkpoint is not None and args.loader != "week2":
        raise ValueError("--week2-checkpoint requires --loader week2")
    if (
        args.solution != "mlx"
        and args.device != "gpu"
        and (
            args.loader == "week3"
            or (args.loader == "week2" and args.week2_checkpoint != "kv-cache")
        )
    ):
        raise ValueError(
            "The completed Week 2 and Week 3 custom-kernel models are GPU-only; "
            "use the Week 2 kv-cache checkpoint for the readable pre-kernel path"
        )
    if args.disable_paged_attention and args.loader != "week3":
        raise ValueError("--disable-paged-attention requires --loader week3")


def load_solution_modules(solution: str):
    if solution == "mlx":
        return "mlx", None, None, None
    if solution == "tiny_llm":
        from tiny_llm import models
        from tiny_llm.kv_cache import BatchingKvCache, TinyKvFullCache

        return "tiny_llm", models, TinyKvFullCache, BatchingKvCache
    if solution in {"tiny_llm_ref", "ref"}:
        from tiny_llm_ref import models
        from tiny_llm_ref.kv_cache import BatchingKvCache, TinyKvFullCache

        return "tiny_llm_ref", models, TinyKvFullCache, BatchingKvCache
    raise ValueError(f"Solution {solution} not supported for bench")


def random_token_id(rng: Random, low: int, high: int, eos_token_id: int) -> int:
    if low == high:
        return low
    token = rng.randint(low, high)
    if token != eos_token_id:
        return token
    if token == low:
        return low + 1
    return token - 1


def build_requests(
    *,
    rng: Random,
    num_seqs: int,
    vocab_size: int,
    eos_token_id: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
) -> list[BenchRequest]:
    token_low = 256 if vocab_size > 512 else 0
    token_high = vocab_size - 1
    if token_low > token_high:
        token_low = 0
    requests = []
    for _ in range(num_seqs):
        prompt_len = rng.randint(min_input_len, max_input_len)
        max_new_tokens = rng.randint(min_output_len, max_output_len)
        prompt_token_ids = [
            random_token_id(rng, token_low, token_high, eos_token_id)
            for _ in range(prompt_len)
        ]
        requests.append(BenchRequest(prompt_token_ids, max_new_tokens))
    return requests


def sample_next_week1(model, y: mx.array) -> mx.array:
    output_logits = model(y[None, :])
    logits = output_logits[:, -1, :]
    return mx.argmax(logits, axis=-1)


def sample_next_week2(
    model,
    y: mx.array,
    offset: int,
    kv_cache: list,
    logits_to_keep: int | None = 1,
) -> mx.array:
    output_logits = model(y[None, :], offset, kv_cache, logits_to_keep=logits_to_keep)
    logits = output_logits[:, -1, :]
    return mx.argmax(logits, axis=-1)


def sample_next_week2_batched(
    model, y: mx.array, offsets: list[int], kv_cache: list
) -> mx.array:
    output_logits = model(y, offsets, kv_cache, logits_to_keep=1)
    logits = output_logits[:, -1, :]
    return mx.argmax(logits, axis=-1)


def run_one_request_week1(
    model,
    request: BenchRequest,
) -> tuple[int, float, float]:
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    t0 = perf_counter()
    token = sample_next_week1(model, context)
    mx.eval(token)
    prefill_time = perf_counter() - t0

    generated_tokens = 1
    decode_time = 0.0

    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        context = mx.concat([context, token])
        token = sample_next_week1(model, context)
        mx.eval(token)
        decode_time += perf_counter() - t1
        generated_tokens += 1
    return generated_tokens, prefill_time, decode_time


def run_one_request_week2(
    model,
    request: BenchRequest,
    prefill_logits_to_keep: int | None = None,
) -> tuple[int, float, float]:
    kv_cache = model.create_kv_cache()
    try:
        context = mx.array(request.prompt_token_ids, dtype=mx.int32)
        offset = 0

        t0 = perf_counter()
        token = sample_next_week2(
            model,
            context,
            offset,
            kv_cache,
            logits_to_keep=prefill_logits_to_keep,
        )
        mx.eval(token)
        prefill_time = perf_counter() - t0
        offset += context.size

        generated_tokens = 1
        decode_time = 0.0

        for _ in range(request.max_new_tokens - 1):
            t1 = perf_counter()
            token = sample_next_week2(model, token, offset, kv_cache)
            mx.eval(token)
            decode_time += perf_counter() - t1
            offset += 1
            generated_tokens += 1
        return generated_tokens, prefill_time, decode_time
    finally:
        for layer_cache in kv_cache:
            layer_cache.release()


def run_one_request_mlx(
    model,
    request: BenchRequest,
    prefill_logits_to_keep: int | None = None,
) -> tuple[int, float, float]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)

    t0 = perf_counter()
    if prefill_logits_to_keep is None:
        logits = model(context[None, :], cache=cache)
    else:
        hidden = model.model(context[None, :], cache=cache)
        hidden = hidden[:, -prefill_logits_to_keep:, :]
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(hidden)
        else:
            logits = model.lm_head(hidden)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    prefill_time = perf_counter() - t0

    generated_tokens = 1
    decode_time = 0.0
    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        logits = model(token[None, :], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        decode_time += perf_counter() - t1
        generated_tokens += 1
    return generated_tokens, prefill_time, decode_time


def run_batch_requests_serving(
    model,
    requests: list[BenchRequest],
    cache_factory,
    batching_kv_cache_cls,
    *,
    batch_size: int,
    prefill_step: int,
) -> ServingMetrics:
    """Benchmark continuous batching without tokenizer/detokenizer overhead."""

    decode_requests: list[BatchRequestState | None] = [None] * batch_size
    batch_kv_cache = [
        batching_kv_cache_cls(max_active_requests=batch_size)
        for _ in range(model.num_hidden_layers)
    ]
    pending_prefill: BatchRequestState | None = None
    next_request_idx = 0
    metrics = ServingMetrics()

    def live_states() -> list[BatchRequestState]:
        states = [state for state in decode_requests if state is not None]
        if pending_prefill is not None:
            states.append(pending_prefill)
        return states

    def record_cache_state() -> None:
        states = live_states()
        metrics.peak_active_requests = max(metrics.peak_active_requests, len(states))
        live_pages = 0
        tail_waste = 0
        dense_bytes = 0
        for state in states:
            for cache in state.kv_cache:
                page_ids = getattr(cache, "page_ids", None)
                page_lens = getattr(cache, "page_lens", None)
                page_size = getattr(cache, "page_size", None)
                if page_ids is not None:
                    live_pages += len(page_ids)
                    if page_lens and page_size is not None:
                        tail_waste += page_size - page_lens[-1]
                else:
                    key_values = getattr(cache, "key_values", None)
                    if key_values is not None:
                        dense_bytes += key_values[0].nbytes + key_values[1].nbytes

        pools = getattr(model, "page_pools", ())
        capacity_pages = sum(getattr(pool, "capacity", 0) for pool in pools)
        paged_bytes = sum(getattr(pool, "storage_nbytes", 0) for pool in pools)
        dense_bytes += sum(
            getattr(cache, "last_batch_bytes", 0) for cache in batch_kv_cache
        )
        metrics.peak_live_pages = max(metrics.peak_live_pages, live_pages)
        metrics.peak_capacity_pages = max(metrics.peak_capacity_pages, capacity_pages)
        metrics.peak_tail_waste_slots = max(metrics.peak_tail_waste_slots, tail_waste)
        metrics.peak_kv_bytes = max(metrics.peak_kv_bytes, dense_bytes, paged_bytes)

    def record_dense_growth(state: BatchRequestState) -> None:
        metrics.dense_growth_copy_bytes += sum(
            getattr(cache, "growth_copy_bytes", 0) for cache in state.kv_cache
        )

    while True:
        if (
            next_request_idx >= len(requests)
            and pending_prefill is None
            and all(req is None for req in decode_requests)
        ):
            break

        if pending_prefill is None and next_request_idx < len(requests):
            pending_prefill = BatchRequestState(
                request=requests[next_request_idx],
                kv_cache=cache_factory(),
            )
            next_request_idx += 1

        if pending_prefill is not None and not pending_prefill.is_prefill_done:
            remaining = (
                len(pending_prefill.request.prompt_token_ids) - pending_prefill.offset
            )
            chunk_len = min(prefill_step, remaining)
            chunk = pending_prefill.request.prompt_token_ids[
                pending_prefill.offset : pending_prefill.offset + chunk_len
            ]
            t0 = perf_counter()
            token = sample_next_week2(
                model,
                mx.array(chunk, dtype=mx.int32),
                pending_prefill.offset,
                pending_prefill.kv_cache,
            )
            mx.eval(token)
            metrics.prefill_time += perf_counter() - t0
            pending_prefill.offset += chunk_len

            for layer_cache in pending_prefill.kv_cache:
                layer_cache.materialize()

            if pending_prefill.offset == len(pending_prefill.request.prompt_token_ids):
                pending_prefill.is_prefill_done = True
                pending_prefill.generated_tokens = 1
                pending_prefill.next_token = token.item()
                metrics.generated_tokens += 1
            record_cache_state()

        if pending_prefill is not None and pending_prefill.is_prefill_done:
            if (
                pending_prefill.generated_tokens
                >= pending_prefill.request.max_new_tokens
            ):
                record_dense_growth(pending_prefill)
                for layer_cache in pending_prefill.kv_cache:
                    layer_cache.release()
                pending_prefill = None
                continue

            for slot, current in enumerate(decode_requests):
                if current is None:
                    for prefill_cache, batch_cache in zip(
                        pending_prefill.kv_cache, batch_kv_cache
                    ):
                        batch_cache.add_request(prefill_cache, slot)
                    decode_requests[slot] = pending_prefill
                    pending_prefill = None
                    break

        if any(req is not None for req in decode_requests):
            next_tokens = []
            offsets = []
            active_slots = []
            for slot, req in enumerate(decode_requests):
                if req is None:
                    next_tokens.append(0)
                    offsets.append(0)
                    continue
                next_tokens.append(req.next_token)
                offsets.append(req.offset)
                active_slots.append(slot)

            t1 = perf_counter()
            decoded = sample_next_week2_batched(
                model,
                mx.array(next_tokens, dtype=mx.int32).reshape(-1, 1),
                offsets,
                batch_kv_cache,
            )
            mx.eval(decoded)
            metrics.decode_time += perf_counter() - t1
            record_cache_state()

            for slot in active_slots:
                req = decode_requests[slot]
                req.next_token = decoded[slot].item()
                req.offset += 1
                req.generated_tokens += 1
                metrics.generated_tokens += 1
                metrics.decode_tokens += 1
                if req.generated_tokens >= req.request.max_new_tokens:
                    record_dense_growth(req)
                    for layer_cache in batch_kv_cache:
                        layer_cache.remove_request(slot)
                    decode_requests[slot] = None

    pools = getattr(model, "page_pools", ())
    metrics.reused_page_allocations = sum(
        getattr(pool, "reused_page_allocations", 0) for pool in pools
    )
    metrics.storage_growths = sum(getattr(pool, "storage_growths", 0) for pool in pools)
    metrics.copied_pages_on_growth = sum(
        getattr(pool, "copied_pages_on_growth", 0) for pool in pools
    )
    metrics.paged_growth_copy_bytes = sum(
        getattr(pool, "copied_bytes_on_growth", 0) for pool in pools
    )
    metrics.dense_staging_copy_bytes = sum(
        getattr(cache, "staging_copy_bytes", 0) for cache in batch_kv_cache
    )
    return metrics


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = Random(args.seed)
    solution_name, models, kv_cache_cls, batching_kv_cache_cls = load_solution_modules(
        args.solution
    )
    model_name = shortcut_name_to_full_name(args.model)
    effective_prefill_logits = "last" if args.batch_decode else args.prefill_logits
    print(
        f"Solution={solution_name} Loader={args.loader} Device={args.device} "
        f"Model={model_name} "
        f"PagedAttention={not args.disable_paged_attention} "
        f"PrefillLogits={effective_prefill_logits} "
        f"Week2Checkpoint={args.week2_checkpoint}"
    )
    mlx_model, tokenizer = load(model_name)

    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        if solution_name == "mlx":
            if args.week2_checkpoint is not None:
                raise ValueError(
                    "--week2-checkpoint is not supported with --solution mlx"
                )
            if args.batch_decode:
                raise ValueError("--batch-decode is not supported with --solution mlx")
            if args.disable_paged_attention:
                raise ValueError(
                    "--disable-paged-attention is not supported with --solution mlx"
                )
            model = mlx_model
            prefill_logits_to_keep = 1 if args.prefill_logits == "last" else None

            def run_one_request(request: BenchRequest) -> tuple[int, float, float]:
                return run_one_request_mlx(
                    model,
                    request,
                    prefill_logits_to_keep=prefill_logits_to_keep,
                )

        elif args.loader == "week1":
            model = models.dispatch_model(model_name, mlx_model, week=1)

            def run_one_request(request: BenchRequest) -> tuple[int, float, float]:
                return run_one_request_week1(
                    model,
                    request,
                )
        else:
            dispatch_kwargs = {}
            if args.loader == "week3":
                dispatch_kwargs[
                    "enable_paged_attention"
                ] = not args.disable_paged_attention
            elif args.loader == "week2" and args.week2_checkpoint is not None:
                dispatch_kwargs["checkpoint"] = args.week2_checkpoint
            model = models.dispatch_model(
                model_name,
                mlx_model,
                week=int(args.loader.removeprefix("week")),
                **dispatch_kwargs,
            )

            if args.batch_decode:
                cache_factory = (
                    model.create_kv_cache
                    if args.loader == "week3"
                    else lambda: [
                        kv_cache_cls() for _ in range(model.num_hidden_layers)
                    ]
                )

                def run_benchmark(
                    bench_requests: list[BenchRequest],
                ) -> ServingMetrics:
                    return run_batch_requests_serving(
                        model,
                        bench_requests,
                        cache_factory,
                        batching_kv_cache_cls,
                        batch_size=args.batch_size,
                        prefill_step=args.prefill_step,
                    )

            else:
                prefill_logits_to_keep = 1 if args.prefill_logits == "last" else None

                def run_one_request(request: BenchRequest) -> tuple[int, float, float]:
                    return run_one_request_week2(
                        model,
                        request,
                        prefill_logits_to_keep=prefill_logits_to_keep,
                    )

        requests = build_requests(
            rng=rng,
            num_seqs=args.num_seqs,
            vocab_size=mlx_model.args.vocab_size,
            eos_token_id=tokenizer.eos_token_id,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
            min_output_len=args.min_output_len,
            max_output_len=args.max_output_len,
        )

        if args.warmup > 0:
            print(f"Warmup runs: {args.warmup}")
            warmup_iter = range(args.warmup)
            warmup_iter = tqdm(
                warmup_iter,
                total=args.warmup,
                desc="Warmup",
                dynamic_ncols=True,
                leave=False,
            )
            for i in warmup_iter:
                if args.batch_decode:
                    run_benchmark(requests)
                else:
                    run_one_request(requests[i % len(requests)])
            if args.batch_decode:
                mx.synchronize()
                for pool in getattr(model, "page_pools", ()):
                    pool.reset()

        total_prompt_tokens = sum(len(request.prompt_token_ids) for request in requests)
        progress = tqdm(total=len(requests), desc="Bench", dynamic_ncols=True)

        total_generated_tokens = 0
        total_decode_tokens = 0
        total_prefill_time = 0.0
        total_decode_time = 0.0
        serving_metrics = None

        t0 = perf_counter()
        if args.batch_decode:
            serving_metrics = run_benchmark(requests)
            total_generated_tokens = serving_metrics.generated_tokens
            total_decode_tokens = serving_metrics.decode_tokens
            total_prefill_time = serving_metrics.prefill_time
            total_decode_time = serving_metrics.decode_time
            progress.update(len(requests))
            elapsed = perf_counter() - t0
            progress.set_postfix(
                {
                    "out_tok/s": f"{safe_div(total_generated_tokens, elapsed):.1f}",
                    "decode_tok/s": (
                        f"{safe_div(total_decode_tokens, total_decode_time):.1f}"
                    ),
                }
            )
        else:
            for request in requests:
                generated_tokens, prefill_time, decode_time = run_one_request(request)
                total_generated_tokens += generated_tokens
                total_decode_tokens += max(0, generated_tokens - 1)
                total_prefill_time += prefill_time
                total_decode_time += decode_time
                elapsed = perf_counter() - t0
                progress.update(1)
                progress.set_postfix(
                    {
                        "out_tok/s": f"{safe_div(total_generated_tokens, elapsed):.1f}",
                        "decode_tok/s": (
                            f"{safe_div(total_decode_tokens, total_decode_time):.1f}"
                        ),
                    }
                )
        total_time = perf_counter() - t0
        progress.close()

    total_model_tokens = total_prompt_tokens + total_generated_tokens
    print(
        f"Requests: {args.num_seqs}, Prompt tokens: {total_prompt_tokens}, "
        f"Generated tokens: {total_generated_tokens}"
    )
    print(
        f"Time: {total_time:.2f}s, Output throughput: "
        f"{safe_div(total_generated_tokens, total_time):.2f} tok/s"
    )
    print(
        f"Total throughput (prompt+output): "
        f"{safe_div(total_model_tokens, total_time):.2f} tok/s"
    )
    print(
        f"Prefill throughput: "
        f"{safe_div(total_prompt_tokens, total_prefill_time):.2f} tok/s"
    )
    print(
        f"Decode throughput: "
        f"{safe_div(total_decode_tokens, total_decode_time):.2f} tok/s"
    )
    if serving_metrics is not None:
        print(f"Request throughput: {safe_div(args.num_seqs, total_time):.2f} req/s")
        print(f"Peak active requests: {serving_metrics.peak_active_requests}")
        print(f"Peak KV bytes: {serving_metrics.peak_kv_bytes}")
        print(f"Peak live KV pages: {serving_metrics.peak_live_pages}")
        print(f"Peak KV capacity pages: {serving_metrics.peak_capacity_pages}")
        print(f"Peak tail waste slots: {serving_metrics.peak_tail_waste_slots}")
        print(f"Reused page allocations: {serving_metrics.reused_page_allocations}")
        print(f"Page-pool growths: {serving_metrics.storage_growths}")
        print(
            f"Pages copied during pool growth: {serving_metrics.copied_pages_on_growth}"
        )
        print(
            "Dense KV bytes copied during growth: "
            f"{serving_metrics.dense_growth_copy_bytes}"
        )
        print(
            "Dense KV bytes copied into batch tensors: "
            f"{serving_metrics.dense_staging_copy_bytes}"
        )
        print(
            "Paged KV bytes copied during pool growth: "
            f"{serving_metrics.paged_growth_copy_bytes}"
        )

    if args.json_output:
        payload = {
            "configuration": {
                "model": args.model,
                "solution": args.solution,
                "loader": args.loader,
                "num_seqs": args.num_seqs,
                "min_input_len": args.min_input_len,
                "max_input_len": args.max_input_len,
                "min_output_len": args.min_output_len,
                "max_output_len": args.max_output_len,
                "batch_decode": args.batch_decode,
                "batch_size": args.batch_size,
                "prefill_step": args.prefill_step,
                "seed": args.seed,
            },
            "metrics": {
                "elapsed_seconds": total_time,
                "output_tokens_per_second": safe_div(
                    total_generated_tokens, total_time
                ),
                "total_tokens_per_second": safe_div(total_model_tokens, total_time),
                "prefill_tokens_per_second": safe_div(
                    total_prompt_tokens, total_prefill_time
                ),
                "decode_tokens_per_second": safe_div(
                    total_decode_tokens, total_decode_time
                ),
                "requests_per_second": safe_div(args.num_seqs, total_time),
                **(asdict(serving_metrics) if serving_metrics is not None else {}),
            },
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
