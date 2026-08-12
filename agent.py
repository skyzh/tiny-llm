"""Run the Week 4 agent loop with a real local MLX model."""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
from pathlib import Path

from model_names import shortcut_name_to_full_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore the current Week 4 agent with a real local model."
    )
    parser.add_argument("task", nargs="+", help="one natural-language goal")
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="pre-created disposable workspace containing no secrets",
    )
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument(
        "--solution",
        choices=["tiny_llm", "tiny_llm_ref", "ref"],
        default="tiny_llm",
        help="course workspace implementation (the model uses local MLX-LM)",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="enable approved write_file and edit_file calls",
    )
    parser.add_argument(
        "--allow-command",
        action="append",
        default=[],
        metavar="COMMAND",
        help="allow one exact command; repeat to allow another",
    )
    parser.add_argument(
        "--receipt-log",
        type=Path,
        metavar="RELATIVE_PATH",
        help="append effect receipts to this workspace-relative JSONL path",
    )
    return parser


def parse_allowed_commands(values: list[str]) -> tuple[tuple[str, ...], ...]:
    commands = []
    for value in values:
        command = tuple(shlex.split(value))
        if not command:
            raise ValueError("--allow-command must not be empty")
        commands.append(command)
    return tuple(commands)


def receipt_path(root: Path, raw: Path | None) -> Path | None:
    if raw is None:
        return None
    if raw.is_absolute() or ".." in raw.parts or not raw.name:
        raise ValueError("--receipt-log must be a workspace-relative file path")
    path = root.joinpath(raw)
    if not path.parent.is_dir():
        raise ValueError("--receipt-log parent directory must exist")
    return path


def confirm_tool(action) -> bool:
    payload = {"tool": action.tool, **action.arguments}
    print("\napproval requested> " + json.dumps(payload, sort_keys=True))
    if action.tool == "run_command":
        print("warning> an allowed command is not confined by the workspace boundary")
    if not sys.stdin.isatty():
        print("approval denied> interactive confirmation requires a TTY")
        return False
    try:
        answer = input("approve this effect? [y/N] ")
    except EOFError:
        print("\napproval denied> no answer received")
        return False
    approved = answer.strip().casefold() in {"y", "yes"}
    print("approval> " + ("yes" if approved else "no"))
    return approved


def show_event(agent, event) -> None:
    print(f"\n[{event.step}] model> {event.response}")
    if isinstance(event.action, agent.ToolAction):
        payload = {"tool": event.action.tool, **event.action.arguments}
        print("action> " + json.dumps(payload, sort_keys=True))
    if event.result is not None:
        print(f"observation> {event.result}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    task = " ".join(args.task).strip()
    if not task:
        parser.error("task must not be empty")
    if args.max_steps <= 0 or args.max_tokens <= 0:
        parser.error("generation limits must be positive")
    if not args.root.is_dir():
        parser.error("--root must be a pre-created directory")
    try:
        commands = parse_allowed_commands(args.allow_command)
        package = (
            "tiny_llm_ref" if args.solution in {"tiny_llm_ref", "ref"} else "tiny_llm"
        )
        agent = importlib.import_module(f"{package}.agent")
        policy = agent.ToolPolicy(
            args.root,
            allow_writes=args.allow_writes,
            allowed_commands=commands,
        )
        log_path = receipt_path(policy.root, args.receipt_log)
        store = agent.ReceiptStore(log_path)
    except (ImportError, ValueError) as error:
        parser.error(str(error))

    workspace = agent.Workspace(policy, confirm_tool, store)
    limits = agent.AgentLimits(max_steps=args.max_steps)
    model_name = shortcut_name_to_full_name(args.model)

    try:
        import mlx.core as mx
        from mlx_lm import load

        mlx_model, tokenizer = load(model_name)
    except Exception as error:
        parser.exit(1, f"error: could not load local model {model_name}: {error}\n")

    def generate(messages):
        from mlx_lm import generate as mlx_generate

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        return mlx_generate(
            mlx_model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )

    print("model> " + model_name)
    print("workspace> " + str(policy.root))
    print("goal> " + task)
    print("tools> " + ", ".join(sorted(workspace.available_tools)))
    if log_path is not None:
        print("receipt log> " + str(log_path))

    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        result = agent.run_agent(
            task,
            generate,
            workspace,
            limits,
            lambda event: show_event(agent, event),
        )

    if result.completed:
        print("\nfinished> " + (result.final or ""))
    else:
        print("\nstopped> " + result.reason)
    print("modified files> " + json.dumps(list(workspace.modified_files)))
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
