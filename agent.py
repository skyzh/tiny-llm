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
    parser.add_argument("task", nargs="*", help="coding task for a new session")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="accept follow-up messages after each bounded run",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="resume the newest session for this workspace and model",
    )
    resume.add_argument(
        "--session",
        metavar="SESSION_ID",
        help="resume a selected session for this workspace and model",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="do not write a persistent session transcript",
    )
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
    parser.add_argument("--max-context-tokens", type=int, default=32_768)
    parser.add_argument("--reserve-tokens", type=int, default=8_192)
    parser.add_argument("--summary-max-tokens", type=int, default=1_024)
    parser.add_argument("--max-tool-result-tokens", type=int, default=4_096)
    parser.add_argument("--min-recent-turns", type=int, default=2)
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


def validate_session_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject ambiguous session lifecycles before loading a model."""

    if not args.task and not (args.continue_session or args.session):
        parser.error("TASK is required unless --continue or --session is used")
    if args.task and not " ".join(args.task).strip():
        parser.error("TASK must not be empty")
    if args.task and (args.continue_session or args.session):
        parser.error("TASK cannot be combined with --continue or --session")
    if args.no_session and (args.continue_session or args.session):
        parser.error("--no-session cannot be combined with a resume option")
    if args.interactive and not getattr(sys.stdin, "isatty", lambda: False)():
        parser.error("--interactive requires a TTY")


def validate_budget_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject generation settings that can exceed the advertised context limit."""

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_tokens > args.reserve_tokens:
        parser.error("--max-tokens must not exceed --reserve-tokens")
    if args.summary_max_tokens >= args.max_context_tokens:
        parser.error("--summary-max-tokens must be below --max-context-tokens")


def confirm_tool_call(action, root: Path) -> bool:
    """Ask a human to approve one side effect, defaulting safely to no."""

    payload = {"tool": action.tool, **action.arguments}
    print("\nHuman approval required before this tool call:")
    print(f"workspace> {json.dumps(str(root), ensure_ascii=True)}")
    print("tool call> " + json.dumps(payload, ensure_ascii=True, sort_keys=True))
    if action.tool == "run_command":
        print("Warning: commands are not confined by the workspace path boundary.")
    if not getattr(sys.stdin, "isatty", lambda: False)():
        print("Denied: interactive confirmation requires a TTY.")
        return False
    try:
        response = input("Execute this tool call? [y/N] ")
    except EOFError:
        print("\nDenied: no confirmation was received.")
        return False
    approved = response.strip().casefold() in {"y", "yes"}
    if not approved:
        print("Denied.")
    return approved


def show_run_result(result) -> None:
    """Render protocol completion separately from independently validated success."""

    if result.completed:
        final = json.dumps(result.final, ensure_ascii=True)
        print("\nModel finished (task success not independently validated): " + final)
    else:
        print(f"\nStopped: {result.reason}")
    show_side_effect_summary(
        result.modified_files,
        result.command_side_effects_untracked,
        result.uncertain_modified_files,
        result.retained_recovery_files,
        result.command_cleanup_incomplete,
    )


def show_side_effect_summary(
    modified_files,
    command_side_effects_untracked: bool,
    uncertain_modified_files=(),
    retained_recovery_files=(),
    command_cleanup_incomplete: bool = False,
) -> None:
    """Report tracked file mutations and the command-tracking boundary."""

    if modified_files:
        modified = json.dumps(list(modified_files), ensure_ascii=True)
        print("Tracked file-tool changes: " + modified)
    if uncertain_modified_files:
        uncertain = json.dumps(list(uncertain_modified_files), ensure_ascii=True)
        print("Warning: file-tool mutation outcome is uncertain; inspect: " + uncertain)
    if retained_recovery_files:
        retained = json.dumps(list(retained_recovery_files), ensure_ascii=True)
        print("Safety copies retained for manual inspection: " + retained)
    if command_side_effects_untracked:
        print(
            "Warning: one or more commands ran; their side effects are not "
            "included in the tracked file-tool changes."
        )
    if command_cleanup_incomplete:
        print(
            "Warning: command cleanup or output collection was incomplete; "
            "inspect the host for a surviving process."
        )


def show_interrupted_run(workspace) -> None:
    """Disclose known side effects when Ctrl-C prevents an AgentRun result."""

    show_stopped_run(workspace, "interrupted")


def show_stopped_run(workspace, reason: str) -> None:
    """Disclose known side effects when no AgentRun result is available."""

    modified = tuple(
        sorted(
            str(path.relative_to(workspace.policy.root))
            for path in workspace.modified_files
        )
    )
    uncertain = tuple(
        sorted(
            str(path.relative_to(workspace.policy.root))
            for path in workspace.uncertain_modified_files
        )
    )
    retained = tuple(
        sorted(
            str(path.relative_to(workspace.policy.root))
            for path in workspace.retained_recovery_files
        )
    )
    print(f"\nStopped: {reason}")
    show_side_effect_summary(
        modified,
        workspace.command_side_effects_untracked,
        uncertain,
        retained,
        workspace.command_cleanup_incomplete,
    )


def run_and_report(function, workspace):
    """Run the agent and always disclose known side effects before exiting."""

    try:
        result = function()
    except KeyboardInterrupt:
        show_stopped_run(workspace, "interrupted")
        raise SystemExit(130) from None
    except Exception:
        show_stopped_run(workspace, "unexpected error")
        raise
    show_run_result(result)
    if result.reason == "interrupted":
        raise SystemExit(130)
    return result


def main():
    """CLI support for Week 4: load a backend and invoke the bounded agent loop."""

    parser = build_parser()
    args = parser.parse_args()
    validate_session_arguments(parser, args)
    validate_budget_arguments(parser, args)
    if args.solution != "mlx" and args.device != "gpu":
        parser.error("The completed Week 2 and Week 3 models require --device gpu")
    try:
        allowed_commands = parse_allowed_commands(args.allow_command)
    except ValueError as error:
        parser.error(str(error))
    if not args.root.exists() or not args.root.is_dir():
        parser.error("--root must be an existing directory")
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
    try:
        policy = agent.ToolPolicy(
            root=args.root,
            allow_writes=args.allow_writes,
            allowed_commands=allowed_commands,
        )
        context_policy = agent.ContextPolicy(
            max_tokens=args.max_context_tokens,
            reserve_tokens=args.reserve_tokens,
            summary_max_tokens=args.summary_max_tokens,
            max_tool_result_tokens=args.max_tool_result_tokens,
            min_recent_turns=args.min_recent_turns,
        )
    except ValueError as error:
        parser.error(str(error))
    limits = agent.AgentLimits(max_steps=args.max_steps)

    if not args.allow_writes:
        print(
            "Safety: workspace tools are read-only; pass --allow-writes to "
            "permit file changes"
        )
    if allowed_commands:
        print("Safety: only the exact --allow-command values may be executed")
    else:
        print("Safety: command execution is disabled")
    if args.allow_writes or allowed_commands:
        print("Safety: each write, edit, and command requires y/N approval")

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

    backend_id = "|".join(
        (
            model_name,
            f"solution={'mlx' if args.solution == 'mlx' else package}",
            f"loader={'mlx' if args.solution == 'mlx' else args.loader}",
            f"thinking={args.enable_thinking}",
        )
    )
    try:
        if args.no_session:
            session_log = agent.memory_session(policy.root, backend_id)
        else:
            session_store = agent.SessionStore(policy.root, backend_id)
            if args.continue_session:
                session_log = session_store.latest()
            elif args.session:
                session_log = session_store.load(args.session)
            else:
                session_log = session_store.create()
    except ValueError as error:
        parser.error(str(error))
    if session_log.path is None:
        print("Session: ephemeral (--no-session)")
    else:
        print(f"Session transcript: {session_log.session_id} (sensitive local data)")
    cancellation = agent.CancellationToken()
    workspace = agent.Workspace(
        policy,
        confirm_tool=lambda action: confirm_tool_call(action, policy.root),
        session_log=session_log,
        cancellation=cancellation,
    )
    for recovery in workspace.recovery_results:
        if recovery.status == "conflict":
            print(
                "Warning: interrupted mutation conflicts with current bytes: "
                + json.dumps(recovery.path, ensure_ascii=True)
            )

    completed_session = next(
        (
            event.data.get("completed") is True
            for event in reversed(session_log.events)
            if event.type == "run_finished"
        ),
        False,
    )
    pending_steering = bool(session_log.pending_steering())
    if (
        completed_session
        and not pending_steering
        and not args.task
        and not args.interactive
    ):
        parser.error(
            "the selected session completed; provide a follow-up interactively"
        )

    generation_session = None
    if args.solution == "mlx":

        def generate(messages):
            """Adapt the stateless MLX-LM compatibility backend."""

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

    else:
        if args.loader == "week2":
            cache_type = importlib.import_module(f"{package}.kv_cache").TinyKvFullCache

            def cache_factory():
                """Week 4, Day 1: allocate one full cache per decoder layer."""

                return [cache_type() for _ in range(model.num_hidden_layers)]

        else:
            cache_factory = model.create_kv_cache
        generation_session = agent.GenerationSession(
            model,
            tokenizer,
            cache_factory,
            args.max_tokens,
            args.enable_thinking,
            cancellation=cancellation,
        )

        def generate(messages):
            """Generate with Day 4's reusable course-model cache."""

            assert generation_session is not None
            response = run_with_spinner(
                "Model is working...", generation_session, messages
            )
            generate.last_stats = generation_session.last_stats
            return response

    if generation_session is None:

        def encode_messages(messages):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            return tuple(tokenizer.encode(prompt, add_special_tokens=False))

    else:
        encode_messages = generation_session.encode_messages

    context_manager = agent.ContextManager(encode_messages, context_policy)

    def summarize(messages):
        """Generate one summary with state separate from the primary cache."""

        if args.solution == "mlx":
            from mlx_lm import generate as mlx_generate

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )

            return run_with_spinner(
                "Compacting context...",
                lambda: mlx_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.summary_max_tokens,
                    verbose=False,
                ),
            )
        summary_session = agent.GenerationSession(
            model,
            tokenizer,
            cache_factory,
            args.summary_max_tokens,
            args.enable_thinking,
            cancellation=cancellation,
        )
        try:
            return run_with_spinner("Compacting context...", summary_session, messages)
        finally:
            summary_session.close()

    def show_event(event):
        """Week 4, Day 7: print the same trace represented by AgentEvent."""

        print(
            f"\n[{event.step}] model> " + json.dumps(event.response, ensure_ascii=True)
        )
        if isinstance(event.action, agent.ToolAction):
            action = {"tool": event.action.tool, **event.action.arguments}
            print(f"tool call> {json.dumps(action, ensure_ascii=True)}")
        if event.result is not None:
            print(f"tool> {json.dumps(event.result, ensure_ascii=True)}")

    def execute_agent(task):
        """Run generation within the requested MLX stream."""

        with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
            return agent.run_agent(
                task,
                generate,
                workspace,
                limits,
                show_event,
                session=session_log,
                context_manager=context_manager,
                summarize=summarize,
                cancellation=cancellation,
            )

    try:
        pending_task = " ".join(args.task).strip() if args.task else None
        if pending_task is None and completed_session and not pending_steering:
            try:
                pending_task = input("\nfollow-up (blank to exit)> ").strip()
            except EOFError:
                return None
            if not pending_task:
                return None
        result = run_and_report(lambda: execute_agent(pending_task), workspace)
        while args.interactive:
            try:
                follow_up = input("\nfollow-up (blank to exit)> ").strip()
            except EOFError:
                break
            if not follow_up:
                break
            result = run_and_report(lambda: execute_agent(follow_up), workspace)
        return result
    finally:
        if generation_session is not None:
            generation_session.close()


if __name__ == "__main__":
    main()
