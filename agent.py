import argparse
import importlib
import json
import shlex
import sys
from itertools import cycle
from pathlib import Path
from threading import Event, Thread

from model_names import shortcut_name_to_full_name


def run_with_spinner(label, function, *args):
    """CLI support for Week 4: show progress without changing agent behavior."""

    if not sys.stdout.isatty():
        return function(*args)
    stopped = Event()

    def animate():
        """CLI support for Week 4: redraw a spinner until generation finishes."""

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


def build_parser() -> argparse.ArgumentParser:
    """CLI support for Week 4: define model, budget, and safety policy flags."""

    parser = argparse.ArgumentParser(description="A tiny Week 4 coding agent.")
    parser.add_argument("task", nargs="+", help="coding task for the agent")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--solution",
        choices=["tiny_llm", "tiny_llm_ref", "ref", "mlx"],
        default="tiny_llm_ref",
    )
    parser.add_argument("--loader", choices=["week2", "week3"], default="week2")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=48_000)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="workspace boundary (defaults to the current directory)",
    )
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="allow file writes; existing files still must be read first",
    )
    parser.add_argument(
        "--allow-command",
        action="append",
        default=[],
        metavar="COMMAND",
        help="allow one exact command; repeat the flag to add another",
    )
    return parser


def parse_allowed_commands(values: list[str]) -> tuple[tuple[str, ...], ...]:
    """Week 4, Day 5: convert operator-approved commands into exact argv tuples."""

    commands = []
    for value in values:
        argv = tuple(shlex.split(value))
        if not argv:
            raise ValueError("--allow-command must not be empty")
        commands.append(argv)
    return tuple(commands)


def main():
    """CLI support for Week 4: load a backend and invoke the bounded agent loop."""

    parser = build_parser()
    args = parser.parse_args()
    if args.solution != "mlx" and args.device != "gpu":
        parser.error("The completed Week 2 and Week 3 models require --device gpu")
    try:
        allowed_commands = parse_allowed_commands(args.allow_command)
    except ValueError as error:
        parser.error(str(error))
    if not args.root.exists() or not args.root.is_dir():
        parser.error("--root must be an existing directory")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    import mlx.core as mx
    from mlx_lm import load

    package = (
        "tiny_llm_ref"
        if args.solution in {"tiny_llm_ref", "ref", "mlx"}
        else "tiny_llm"
    )
    agent = importlib.import_module(f"{package}.agent")
    models = None
    if args.solution != "mlx":
        models = importlib.import_module(f"{package}.models")
    policy = agent.ToolPolicy(
        root=args.root,
        allow_writes=args.allow_writes,
        allowed_commands=allowed_commands,
    )
    workspace = agent.Workspace(policy)
    limits = agent.AgentLimits(
        max_steps=args.max_steps,
        max_context_chars=args.max_context_chars,
    )

    if not args.allow_writes:
        print("Safety: read-only mode; pass --allow-writes to permit file changes")
    if allowed_commands:
        print("Safety: only the exact --allow-command values may be executed")
    else:
        print("Safety: command execution is disabled")

    model_name = shortcut_name_to_full_name(args.model)
    mlx_model, tokenizer = load(model_name)
    if args.solution == "mlx":
        model = mlx_model
        print(f"Using the MLX executor on {args.device}; --loader is ignored")
    else:
        assert models is not None
        dispatch_args = {}
        model = models.dispatch_model(
            model_name, mlx_model, week=int(args.loader[-1]), **dispatch_args
        )
        print(f"Using {package} with the {args.loader} loader on {args.device}")

    def generate(messages):
        """Week 4, Day 1: adapt the selected inference backend to the agent API."""

        if args.solution == "mlx":
            from mlx_lm import generate as mlx_generate

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )

            def generate_mlx():
                """Week 4, Day 1: decode with the optimized MLX-LM backend."""

                return mlx_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    verbose=False,
                )

            return run_with_spinner("Model is working...", generate_mlx)

        if args.loader == "week2":
            cache_type = importlib.import_module(f"{package}.kv_cache").TinyKvFullCache

            def cache_factory():
                """Week 4, Day 1: allocate one full cache per decoder layer."""

                return [cache_type() for _ in range(model.num_hidden_layers)]

        else:
            cache_factory = model.create_kv_cache
        return run_with_spinner(
            "Model is working...",
            agent.generate_response,
            model,
            tokenizer,
            messages,
            cache_factory,
            args.max_tokens,
            args.enable_thinking,
        )

    def show_event(event):
        """Week 4, Day 7: print the same trace represented by AgentEvent."""

        print(f"\n[{event.step}] {event.response}")
        if isinstance(event.action, agent.ToolAction):
            action = {"tool": event.action.tool, **event.action.arguments}
            print(f"tool call> {json.dumps(action, ensure_ascii=False)}")
        if event.result is not None:
            print(f"tool> {event.result}")

    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        result = agent.run_agent(
            " ".join(args.task), generate, workspace, limits, show_event
        )
    if result.completed:
        print(f"\nCompleted: {result.final}")
    else:
        print(f"\nStopped: {result.reason}")
    if result.modified_files:
        print("Modified: " + ", ".join(result.modified_files))


if __name__ == "__main__":
    main()
