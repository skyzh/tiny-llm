import argparse
import json
from pathlib import Path

SYSTEM = """You are a coding agent. Inspect the workspace before editing it.
Reply with exactly one JSON object and no markdown. Available actions:
{"tool":"list_files","path":"."}
{"tool":"read_file","path":"README.md"}
{"tool":"write_file","path":"hello.py","content":"print('hello')\n"}
When the task is complete: {"final":"brief summary"}
Paths must be relative. Keep changes small and never invent file contents."""
ROOT = Path.cwd().resolve()


def generate(model, tokenizer, messages):
    import mlx.core as mx

    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    caches = model.create_kv_cache()
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    output, offset = [], 0
    try:
        for _ in range(256):
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


def main():
    parser = argparse.ArgumentParser(description="A tiny Week 4 coding agent.")
    parser.add_argument("task", nargs="+", help="coding task for the agent")
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    from mlx_lm import load
    from tiny_llm_ref import models

    model_name = models.shortcut_name_to_full_name(args.model)
    mlx_model, tokenizer = load(model_name)
    model = models.dispatch_model(model_name, mlx_model, week=2)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": " ".join(args.task)},
    ]
    for step in range(args.max_steps):
        response = generate(model, tokenizer, messages)
        print(f"\n[{step + 1}] {response}")
        try:
            action = json.loads(response)
        except json.JSONDecodeError:
            action = None
        if action and "final" in action:
            return
        result = (
            run_tool(action)
            if action and "tool" in action
            else "error: reply with one valid JSON action"
        )
        print(f"tool> {result}")
        messages += [
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Tool result:\n{result}"},
        ]


if __name__ == "__main__":
    main()
