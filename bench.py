import argparse
from dataclasses import dataclass
from random import Random
from time import perf_counter

import mlx.core as mx
from mlx_lm import load
from tqdm.auto import tqdm


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tiny-llm token throughput with synthetic token IDs."
    )
    parser.add_argument("--model", type=str, default="qwen2-0.5b")
    parser.add_argument("--solution", type=str, default="tiny_llm")
    parser.add_argument("--loader", type=str, default="week2", choices=["week1", "week2"])
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--min-output-len", type=int, default=64)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
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


def load_solution_modules(solution: str):
    if solution == "tiny_llm":
        from tiny_llm import models
        from tiny_llm.kv_cache import TinyKvFullCache

        return "tiny_llm", models, TinyKvFullCache
    if solution in {"tiny_llm_ref", "ref"}:
        from tiny_llm_ref import models
        from tiny_llm_ref.kv_cache import TinyKvFullCache

        return "tiny_llm_ref", models, TinyKvFullCache
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


def sample_next_week2(model, y: mx.array, offset: int, kv_cache: list) -> mx.array:
    output_logits = model(y[None, :], offset, kv_cache)
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
    kv_cache_cls,
) -> tuple[int, float, float]:
    kv_cache = [kv_cache_cls() for _ in range(model.num_hidden_layers)]
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    offset = 0

    t0 = perf_counter()
    token = sample_next_week2(model, context, offset, kv_cache)
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


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = Random(args.seed)
    solution_name, models, kv_cache_cls = load_solution_modules(args.solution)
    model_name = models.shortcut_name_to_full_name(args.model)
    print(
        f"Solution={solution_name} Loader={args.loader} Device={args.device} "
        f"Model={model_name}"
    )
    mlx_model, tokenizer = load(model_name)

    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        if args.loader == "week1":
            model = models.dispatch_model(model_name, mlx_model, week=1)

            def run_one_request(request: BenchRequest) -> tuple[int, float, float]:
                return run_one_request_week1(
                    model,
                    request,
                )
        else:
            model = models.dispatch_model(
                model_name,
                mlx_model,
                week=2,
            )

            def run_one_request(request: BenchRequest) -> tuple[int, float, float]:
                return run_one_request_week2(
                    model,
                    request,
                    kv_cache_cls,
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
                run_one_request(requests[i % len(requests)])

        total_prompt_tokens = 0
        total_generated_tokens = 0
        total_decode_tokens = 0
        total_prefill_time = 0.0
        total_decode_time = 0.0

        progress = tqdm(total=len(requests), desc="Bench", dynamic_ncols=True)

        t0 = perf_counter()
        for request in requests:
            generated_tokens, prefill_time, decode_time = run_one_request(request)
            total_prompt_tokens += len(request.prompt_token_ids)
            total_generated_tokens += generated_tokens
            total_decode_tokens += max(0, generated_tokens - 1)
            total_prefill_time += prefill_time
            total_decode_time += decode_time
            elapsed = perf_counter() - t0
            progress.update(1)
            progress.set_postfix(
                {
                    "out_tok/s": f"{safe_div(total_generated_tokens, elapsed):.1f}",
                    "decode_tok/s": f"{safe_div(total_decode_tokens, total_decode_time):.1f}",
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


if __name__ == "__main__":
    main()
