from mlx_lm import load
import mlx_lm
import mlx.core as mx
import argparse

import mlx_lm.sample_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlx-community/Qwen3-8B-4bit")
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="tiny_llm")
parser.add_argument("--loader", type=str, default="week1")
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=None)
parser.add_argument("--sampler-top-k", type=int, default=None)
parser.add_argument("--enable-thinking", type=bool, default=False)
parser.add_argument("--enable-flash-attn", type=bool, default=False)

args = parser.parse_args()

use_mlx = False
if args.solution == "tiny_llm":
    print("Using your tiny_llm solution")
    from tiny_llm import (
        Qwen3ModelWeek1,
        Qwen3ModelWeek2,
        simple_generate,
        simple_generate_with_kv_cache,
        sampler,
    )

elif args.solution == "tiny_llm_ref" or args.solution == "ref":
    print("Using tiny_llm_ref solution")
    from tiny_llm_ref import (
        Qwen3ModelWeek1,
        Qwen3ModelWeek2,
        simple_generate,
        simple_generate_with_kv_cache,
        sampler,
    )

elif args.solution == "mlx":
    use_mlx = True
    from mlx_lm.generate import stream_generate

    print("Using the original mlx model")
else:
    raise ValueError(f"Solution {args.solution} not supported")

mlx_model, tokenizer = load(args.model)

with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
    if use_mlx:
        tiny_llm_model = mlx_model
    else:
        if args.loader == "week1":
            print("Using Qwen3ModelWeek1 loader")
            tiny_llm_model = Qwen3ModelWeek1(mlx_model)
        elif args.loader == "week2":
            print("Using Qwen3ModelWeek2 loader")
            tiny_llm_model = Qwen3ModelWeek2(
                mlx_model, enable_flash_attn=args.enable_flash_attn
            )
        else:
            raise ValueError(f"Loader {args.loader} not supported")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    if not use_mlx:
        sampler = sampler.make_sampler(
            args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
        )
        if args.loader == "week1":
            simple_generate(tiny_llm_model, tokenizer, prompt, sampler=sampler)
        elif args.loader == "week2":
            simple_generate_with_kv_cache(tiny_llm_model, tokenizer, prompt)
    else:
        sampler = mlx_lm.sample_utils.make_sampler(
            args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
        )
        for resp in stream_generate(tiny_llm_model, tokenizer, prompt, sampler=sampler):
            print(resp.text, end="", flush=True)
