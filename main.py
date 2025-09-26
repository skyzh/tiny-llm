from mlx_lm import load
import mlx_lm
import mlx.core as mx
import argparse

import mlx_lm.sample_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2-7b")
parser.add_argument("--draft-model", type=str, default=None)
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
parser.add_argument("--enable-thinking", action="store_true")
parser.add_argument("--enable-flash-attn", action="store_true")

args = parser.parse_args()

use_mlx = False
if args.solution == "tiny_llm":
    print("Using your tiny_llm solution")
    from tiny_llm import (
        models,
        simple_generate,
        simple_generate_with_kv_cache,
        sampler,
    )

elif args.solution == "tiny_llm_ref" or args.solution == "ref":
    print("Using tiny_llm_ref solution")
    from tiny_llm_ref import (
        models,
        simple_generate,
        simple_generate_with_kv_cache,
        speculative_generate,
        sampler,
    )

elif args.solution == "mlx":
    use_mlx = True
    from mlx_lm.generate import stream_generate

    print("Using the original mlx model")
else:
    raise ValueError(f"Solution {args.solution} not supported")

args.model = models.shortcut_name_to_full_name(args.model)
mlx_model, tokenizer = load(args.model)

if args.draft_model:
    args.draft_model = models.shortcut_name_to_full_name(args.draft_model)
    draft_mlx_model, draft_tokenizer = load(args.draft_model)
    if args.loader == "week1":
        raise ValueError("Draft model not supported for week1")
else:
    draft_mlx_model = None
    draft_tokenizer = None

with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
    if use_mlx:
        tiny_llm_model = mlx_model
    else:
        if args.loader == "week1":
            print(f"Using week1 loader for {args.model}")
            tiny_llm_model = models.dispatch_model(args.model, mlx_model, week=1)
        elif args.loader == "week2":
            print(
                f"Using week2 loader with flash_attn={args.enable_flash_attn} thinking={args.enable_thinking} for {args.model}"
            )
            tiny_llm_model = models.dispatch_model(
                args.model, mlx_model, week=2, enable_flash_attn=args.enable_flash_attn
            )
            if draft_mlx_model is not None:
                print(f"Using draft model {args.draft_model}")
                draft_tiny_llm_model = models.dispatch_model(
                    args.draft_model, draft_mlx_model, week=2, enable_flash_attn=args.enable_flash_attn
                )
            else:
                draft_tiny_llm_model = None
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
            if draft_tiny_llm_model is not None:
                speculative_generate(draft_tiny_llm_model, tiny_llm_model, draft_tokenizer, tokenizer, prompt)
            else:
                simple_generate_with_kv_cache(tiny_llm_model, tokenizer, prompt)
    else:
        sampler = mlx_lm.sample_utils.make_sampler(
            args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
        )
        for resp in stream_generate(tiny_llm_model, tokenizer, prompt, sampler=sampler):
            print(resp.text, end="", flush=True)
