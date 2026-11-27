"""Run the Week 4 agent with a local model or a fixed fixture."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from model_names import MODEL_SHORTCUTS, shortcut_name_to_full_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore the Week 4 agent with a local model or a fixed fixture."
    )
    parser.add_argument("task", nargs="*", help="one natural-language goal")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument(
        "--autoreject-fixture",
        action="store_true",
        help="run the fixed reference fixture without a model or shell",
    )
    profiles.add_argument(
        "--inspect-fixture",
        action="store_true",
        help="give a cached Week 2 model read-only access to the fixed sample",
    )
    profiles.add_argument(
        "--live-fixture",
        action="store_true",
        help="pause and branch a cached Week 2 run on the fixed sample",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="pre-created disposable workspace containing no secrets",
    )
    parser.add_argument("--model", default="qwen3-4b")
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


def confirm_tool(action) -> bool:
    payload = {"tool": action.tool, **action.arguments}
    print("\napproval requested> " + json.dumps(payload, sort_keys=True))
    if action.tool == "run_command":
        print("warning> an allowed command is not confined by the workspace boundary")
        print(
            "warning> approved commands may create or change additional filesystem "
            "paths; command receipts record the command and result, not a complete "
            "filesystem diff"
        )
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


def show_file_tool_changes(paths) -> None:
    print("file-tool changes> " + json.dumps(list(paths)))


def cached_model_path(parser, model_name):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(model_name, local_files_only=True)
    except LocalEntryNotFoundError:
        parser.exit(
            1,
            f"error: {model_name} is missing from the local model cache. "
            "Download it during course setup, then retry the fixture.\n",
        )


def load_week_two(parser, models, model_name, model_path, device):
    from mlx_lm import load

    try:
        loaded, tokenizer = load(
            model_path, tokenizer_config={"trust_remote_code": False}
        )
        options = (
            {"checkpoint": "packed-weights", "use_mlx_quantized_linear": True}
            if device == "cpu"
            else {}
        )
        return models.dispatch_model(model_name, loaded, week=2, **options), tokenizer
    except Exception as error:
        parser.exit(
            1, f"error: could not load the Week 2 model {model_name}: {error}\n"
        )


def run_interactive(parser, args, agent, models, task, workspace, log_path):
    import mlx.core as mx

    model_name = shortcut_name_to_full_name(args.model)
    if not model_name.startswith("Qwen/Qwen3"):
        parser.error("--model must be a Qwen/Qwen3 model supported by Week 2")
    model_path = model_name
    if args.inspect_fixture:
        if model_name not in MODEL_SHORTCUTS.values():
            parser.error("--inspect-fixture requires a supported Qwen3 shortcut")
        from week4_fixture import inspection_messages, observed_fixture_read

        model_path = cached_model_path(parser, model_name)

    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        model, tokenizer = load_week_two(
            parser, models, model_name, model_path, args.device
        )

        def generate(messages):
            if args.inspect_fixture:
                messages = inspection_messages(messages)
            return agent.generate_response(
                model,
                tokenizer,
                messages,
                model.create_kv_cache,
                args.max_tokens,
                enable_thinking=args.enable_thinking,
            )

        print("model> " + model_name)
        implementation = "reference" if args.inspect_fixture else "learner"
        print("backend> Week 2 " + implementation)
        print("workspace> " + str(workspace.policy.root))
        print("goal> " + task)
        print("tools> " + ", ".join(sorted(workspace.available_tools)))
        if log_path is not None:
            print("receipt log> " + str(log_path))
        result = agent.run_agent(
            task,
            generate,
            workspace,
            agent.AgentLimits(max_steps=args.max_steps),
            lambda event: show_event(agent, event),
        )

    if args.inspect_fixture:
        if not observed_fixture_read(agent, result):
            print("\nstopped> the required read of app.py was not observed")
            return 1
        print("fixture> observed the expected app.py bytes")
    if result.completed:
        print("\nfinished> " + (result.final or ""))
    else:
        print("\nstopped> " + result.reason)
    show_file_tool_changes(workspace.modified_files)
    return 0 if result.completed else 1


def run_live(parser):
    model_name = shortcut_name_to_full_name("qwen3-4b")
    model_path = cached_model_path(parser, model_name)
    import mlx.core as mx

    from week4_fixture import inspection_backend
    from week4_live_fixture import run_live_fixture

    try:
        agent, models = inspection_backend()
    except ImportError:
        parser.exit(1, "error: complete Setup and run `pdm run build-ext-ref` first.\n")

    with mx.stream(mx.gpu):
        model, tokenizer = load_week_two(parser, models, model_name, model_path, "gpu")
        print("model> " + model_name)
        print("backend> Week 2 reference")
        try:
            run_live_fixture(agent, model, tokenizer)
        except (agent.AgentError, EOFError) as error:
            reason = (
                "the process ended before continuation"
                if isinstance(error, EOFError)
                else str(error)
            )
            parser.exit(1, f"stopped> {reason}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.autoreject_fixture:
        if argv != ["--autoreject-fixture"]:
            parser.error("--autoreject-fixture takes no other arguments")
        from week4_fixture import run_fixture

        print(json.dumps(run_fixture(), sort_keys=True))
        return 0
    if args.live_fixture:
        if argv != ["--live-fixture"]:
            parser.error("--live-fixture takes no other arguments")
        return run_live(parser)

    if args.max_steps <= 0 or args.max_tokens <= 0:
        parser.error("generation limits must be positive")
    task = " ".join(args.task).strip()
    if args.inspect_fixture:
        if (
            task
            or args.root is not None
            or args.allow_writes
            or args.allow_command
            or args.receipt_log
        ):
            parser.error("--inspect-fixture uses only the supplied read-only sample")
    else:
        if not task:
            parser.error("task must not be empty")
        if args.root is None or not args.root.is_dir():
            parser.error("--root must be a pre-created directory")

    if args.inspect_fixture:
        from week4_fixture import inspection_backend, inspection_workspace

        agent, models = inspection_backend()
        task = "Read app.py and describe the greeting in one sentence."
        with inspection_workspace(agent) as workspace:
            return run_interactive(parser, args, agent, models, task, workspace, None)

    from agent_cli_receipts import manual_workspace
    from tiny_llm import agent, models

    try:
        commands = parse_allowed_commands(args.allow_command)
        policy = agent.ToolPolicy(
            args.root, allow_writes=args.allow_writes, allowed_commands=commands
        )
        workspace, log_path = manual_workspace(
            agent, policy, args.receipt_log, confirm_tool
        )
    except ValueError as error:
        parser.error(str(error))

    return run_interactive(parser, args, agent, models, task, workspace, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
