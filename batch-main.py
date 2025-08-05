from mlx_lm import load
import mlx.core as mx
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct-MLX")

shanghai_wikipedia = """
Shanghai[a] is a direct-administered municipality and the most populous urban area in China. The city is located on the Chinese shoreline on the southern estuary of the Yangtze River, with the Huangpu River flowing through it. The population of the city proper is the second largest in the world after Chongqing, with around 24.87 million inhabitants in 2023, while the urban area is the most populous in China, with 29.87 million residents. As of 2022, the Greater Shanghai metropolitan area was estimated to produce a gross metropolitan product (nominal) of nearly 13 trillion RMB ($1.9 trillion).[13] Shanghai is one of the world's major centers for finance, business and economics, research, science and technology, manufacturing, transportation, tourism, and culture. The Port of Shanghai is the world's busiest container port.
""".strip()

shanghai_wikipedia += "Based on the previous information, "

prompts = [
    shanghai_wikipedia + "Where is Shanghai?",
    shanghai_wikipedia + "How much is the population of Shanghai?",
    shanghai_wikipedia + "What is the GDP of Shanghai?",
    shanghai_wikipedia + "What is the population of Shanghai?",
    shanghai_wikipedia + "What is the second largest city proper in China?",
    shanghai_wikipedia + "What is Shanghai known for?",
    shanghai_wikipedia + "What are the rivers in Shanghai?",
    shanghai_wikipedia + "Shanghai is the major center for what?",
    "What is the capital of France?",
    "Where is New York City?",
    "Where is Tokyo?",
    "What is the capital of China?",
    "Where is Pittsburgh?",
    "Where is Vancouver?",
    "Where is Toronto?",
    "Give me a short introduction to large language model.",
]

# shuffle prompts
random.shuffle(prompts)

parser.add_argument("--solution", type=str, default="tiny_llm")
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--batch-size", type=int, default=5)
parser.add_argument("--prefill-step", type=int, default=128)
args = parser.parse_args()

if args.solution == "tiny_llm":
    print("Using your tiny_llm solution")
    from tiny_llm import Qwen2ModelWeek2, batch_generate

elif args.solution == "tiny_llm_ref" or args.solution == "ref":
    print("Using tiny_llm_ref solution")
    from tiny_llm_ref import Qwen2ModelWeek2, batch_generate

else:
    raise ValueError(f"Solution {args.solution} not supported")

mlx_model, tokenizer = load(args.model)

with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
    tiny_llm_model = Qwen2ModelWeek2(mlx_model)
    encoded_prompts = []
    for idx, prompt in enumerate(prompts):
        print(f"Prompt {idx}: {prompt}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_prompts.append(prompt)
    result = batch_generate(
        tiny_llm_model,
        tokenizer,
        encoded_prompts,
        batch_size=args.batch_size,
        prefill_step=args.prefill_step,
    )
    for prompt_idx, text in result:
        print(f"--- {prompt_idx} ---")
        print(f"Q: {prompts[prompt_idx]}")
        print(f"A: {text}")
