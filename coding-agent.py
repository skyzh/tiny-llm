import argparse
import importlib
import json
import sys
from itertools import cycle
from pathlib import Path
from threading import Event, Thread

SYSTEM = """You are a coding agent. Inspect the workspace before editing it.
Reply with exactly one JSON object and no markdown. Available actions:
{"tool":"list_files","path":"."}
{"tool":"read_file","path":"README.md"}
{"tool":"write_file","path":"hello.py","content":"print('hello')\n"}
When the task is complete: {"final":"brief summary"}
Paths must be relative. Keep changes small and never invent file contents."""
ROOT = Path.cwd().resolve()


def run_with_spinner(label, function, *args):
    if not sys.stdout.isatty():
        return function(*args)
    stopped = Event()

    def animate():
        for frame in cycle("|/-\\"):
            print(f"\r{frame} {label}", end="", flush=True)
            if stopped.wait(0.1):
                break

    thread = Thread(target=animate, daemon=True)
    thread.start()
    try:
        return function(*args)
    finally:
        stopped.set()
        thread.join()
        print(f"\r{' ' * (len(label) + 2)}\r", end="", flush=True)


def generate(model, tokenizer, messages, cache_type, args):
    import mlx.core as mx

    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    if args.loader == "week2":
        caches = [cache_type() for _ in range(model.num_hidden_layers)]
    else:
        caches = model.create_kv_cache()
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    output, offset = [], 0
    try:
        for _ in range(args.max_tokens):
            logits = model(tokens[None], offset, caches)[:, -1, :]
            token = int(mx.argmax(logits, axis=-1).item())
            if token == tokenizer.eos_token_id:
                break
            output.append(token)
            offset += tokens.size
            tokens = mx.array([token])
        return tokenizer.decode(output)
    finally:
        for cache in caches:
            cache.release()


def safe_path(raw):
    path = (ROOT / raw).resolve()
    path.relative_to(ROOT)
    if ".git" in path.relative_to(ROOT).parts:
        raise ValueError(".git is not accessible")
    return path


def run_tool(action):
    try:
        path = safe_path(action.get("path", "."))
        if action["tool"] == "list_files":
            items = sorted(path.iterdir(), key=lambda item: item.name)
            return "\n".join(
                f"{'dir' if item.is_dir() else 'file'} {item.relative_to(ROOT)}"
                for item in items[:100]
                if item.name != ".git"
            )
        if action["tool"] == "read_file":
            return path.read_text()[:12000]
        if action["tool"] == "write_file":
            content = action["content"]
            if len(content) > 12000:
                raise ValueError("content exceeds 12000 characters")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"wrote {path.relative_to(ROOT)}"
        return "error: unknown tool"
    except (KeyError, OSError, ValueError) as error:
        return f"error: {error}"


def parse_action(response):
    try:
        action, _ = json.JSONDecoder().raw_decode(response[response.index("{") :])
        return action
    except (ValueError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="A tiny Week 4 coding agent.")
    parser.add_argument("task", nargs="+", help="coding task for the agent")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--solution",
        choices=["tiny_llm", "tiny_llm_ref", "ref"],
        default="tiny_llm_ref",
    )
    parser.add_argument("--loader", choices=["week2", "week3"], default="week2")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--enable-flash-attn", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    from mlx_lm import load
    import mlx.core as mx

    package = "tiny_llm_ref" if args.solution in {"tiny_llm_ref", "ref"} else "tiny_llm"
    models = importlib.import_module(f"{package}.models")
    cache_type = importlib.import_module(f"{package}.kv_cache").TinyKvFullCache
    print(f"Using {package} with the {args.loader} loader on {args.device}")
    model_name = models.shortcut_name_to_full_name(args.model)
    mlx_model, tokenizer = load(model_name)
    dispatch_args = {"enable_flash_attn": args.enable_flash_attn}
    if args.loader == "week3":
        dispatch_args = {}
        if args.enable_flash_attn:
            print("--enable-flash-attn is only used by the week2 loader; ignoring it")
    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        model = models.dispatch_model(
            model_name, mlx_model, week=int(args.loader[-1]), **dispatch_args
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": " ".join(args.task)},
        ]
        for step in range(args.max_steps):
            response = run_with_spinner(
                "Model is working...",
                generate,
                model,
                tokenizer,
                messages,
                cache_type,
                args,
            )
            print(f"\n[{step + 1}] {response}")
            action = parse_action(response)
            if action and "final" in action:
                return
            if action and "tool" in action:
                print(f"tool call> {json.dumps(action, ensure_ascii=False)}")
                result = run_tool(action)
            else:
                result = "error: reply with one valid JSON action"
            print(f"tool> {result}")
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"Tool result:\n{result}"},
            ]


if __name__ == "__main__":
    main()
