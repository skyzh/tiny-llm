"""Pause one guided Week 2 run and compare two confined continuations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from week4_fixture import _APP, _EDIT, _EXPECTED_APP, _NOTES, _VALIDATE, _WRITE
from week4_fixture import _fixture_workspace

_TASK = (
    "Inspect the greeting, change Hello to Welcome, and validate the fixture bytes. "
    "This is a guided fixture. Transcribe the JSON from the latest fixture step. "
    "Return a final response only when that step shows a final action."
)
_DIAGNOSTIC = "Greeting fixture validation: inspect the recorded byte checks.\n"
_BUILD_LOG = _DIAGNOSTIC + "".join(
    f"Fixture check {number:02d} compared the expected greeting bytes.\n"
    for number in range(1, 19)
)
_FILES = (("app.py", _EXPECTED_APP), ("build.log", _BUILD_LOG.encode()))
_PATHS = frozenset({"app.py", "build.log", "NOTES.md"})
_READ_APP = {"tool": "read_file", "path": "app.py"}
_DENIED_WRITE = {**_WRITE, "content": "An unapproved note.\n"}
_BASE_ACTIONS = (_READ_APP, _EDIT, _VALIDATE, _DENIED_WRITE, _READ_APP)
_BRANCHES = {
    "validate-only": "Keep the changed greeting and only validate the existing files.",
    "add-notes": "Keep the changed greeting and write the fixed note before validation.",
}
_MAX_TOKENS = 160


class _GuidedGenerator:
    """Give the model a reproducible next-action prompt for this fixture."""

    def __init__(self, prefix, generate, actions, *, guide_system=False):
        self._prefix = prefix
        self._generate = generate
        self._actions = iter(actions)
        self._guide_system = guide_system

    def __call__(self, messages):
        action = json.dumps(next(self._actions), separators=(",", ":"))
        instruction = (
            "Transcribe this one fixture action. Copy the next line verbatim as "
            f"your entire reply, without markdown or commentary:\n{action}"
        )
        guide = {"role": "user", "content": instruction}
        if self._guide_system:
            guide["content"] = messages[-1]["content"] + "\n\n" + instruction
            messages = [{"role": "system", "content": instruction}]
        return self._generate([*messages, guide])

    def save_checkpoint(self, messages):
        return self._prefix.save_checkpoint(messages)

    def restore_checkpoint(self, checkpoint):
        self._prefix.restore_checkpoint(checkpoint)


class _ObservedModel:
    """Record the input and cache positions of the first model call in a branch."""

    def __init__(self, model):
        self._model = model
        self.num_hidden_layers = model.num_hidden_layers
        self.first_call = None

    def __call__(self, tokens, offset, caches, **options):
        if self.first_call is None:
            self.first_call = {
                "offset": offset,
                "input_tokens": tokens.shape[-1],
                "layer_offsets": [cache.offset for cache in caches],
            }
        return self._model(tokens, offset, caches, **options)


def _show(label, value):
    print(f"{label}> {json.dumps(value, sort_keys=True)}", flush=True)


def _receipts(api, ledger):
    return [
        api.EffectReceipt.from_dict(json.loads(line))
        for line in ledger.read_text(encoding="utf-8").splitlines()
    ]


def _receipt_summary(receipts):
    return [
        {"id": receipt.receipt_id, "tool": receipt.tool, "result": receipt.exit_state}
        for receipt in receipts
    ]


def _start(api, model, tokenizer, root):
    base = root / "base"
    base.mkdir()
    (base / "app.py").write_bytes(_APP)
    (base / "build.log").write_text(_BUILD_LOG, encoding="utf-8")
    ledger = root / "base.jsonl"
    workspace = _fixture_workspace(
        api,
        base,
        api.ReceiptStore(ledger),
        paths=_PATHS,
        allowed_effects=(_EDIT, _VALIDATE),
        expected_files=_FILES,
        detailed_validation=True,
    )
    prefix = api.KvPrefixGenerator(model, tokenizer, _MAX_TOKENS)

    def generate(messages):
        return api.generate_response(
            model, tokenizer, messages, model.create_kv_cache, _MAX_TOKENS
        )

    checkpoint = api.run_to_checkpoint(
        _TASK,
        _GuidedGenerator(prefix, generate, _BASE_ACTIONS, guide_system=True),
        workspace,
        after_tool_calls=len(_BASE_ACTIONS),
        limits=api.AgentLimits(max_steps=len(_BASE_ACTIONS)),
    )
    parsed = [
        json.loads(text) for role, text in checkpoint.messages if role == "assistant"
    ]
    observations = [
        text.removeprefix("Tool result:\n")
        for role, text in checkpoint.messages[2:]
        if role == "user"
    ]
    if (
        parsed != list(_BASE_ACTIONS)
        or observations[0] != _APP.decode()
        or observations[3] != "error: operator denied the tool action"
        or observations[4] != _EXPECTED_APP.decode()
        or workspace.modified_files != ("app.py",)
        or not ledger.exists()
    ):
        raise api.AgentError(
            "the model did not reach the expected fixture observations"
        )
    receipts = _receipts(api, ledger)
    if [(receipt.tool, receipt.exit_state) for receipt in receipts] != [
        ("edit_file", "ok"),
        ("run_command", "ok"),
    ]:
        raise api.AgentError(
            "the completed edit and fixed validation need exact receipts"
        )
    return checkpoint, prefix, receipts


def _run_branch(api, model, base, root, checkpoint, generator, name, steering):
    branch = root / name
    branch.mkdir()
    for path, _ in _FILES:
        (branch / path).write_bytes((base / path).read_bytes())
    ledger = root / f"{name}.jsonl"
    initial_ledger = (root / "base.jsonl").read_bytes()
    ledger.write_bytes(initial_ledger)
    receipts = api.ReceiptStore(ledger)
    add_notes = name == "add-notes"
    files = (*_FILES, ("NOTES.md", _NOTES.encode())) if add_notes else _FILES
    workspace = _fixture_workspace(
        api,
        branch,
        receipts,
        paths=_PATHS,
        expected_files=files,
        allowed_effects=(_WRITE, _VALIDATE) if add_notes else (_VALIDATE,),
    )

    artifact_dir = root / f"{name}-evidence"
    artifact_dir.mkdir()
    store = api.ArtifactStore(artifact_dir)
    range_end = len(_DIAGNOSTIC.encode())
    log_bytes = _BUILD_LOG.encode()
    artifact_id = "artifact-" + hashlib.sha256(log_bytes).hexdigest()
    bounded = api.BoundedEvidenceWorkspace(workspace, store, max_range_bytes=range_end)
    range_action = {
        "tool": "read_file",
        "path": store.range_path(artifact_id, 0, range_end),
    }
    actions = (
        *((_WRITE,) if add_notes else ()),
        _VALIDATE,
        {"tool": "read_file", "path": "build.log"},
        range_action,
        {"final": "fixture validated"},
    )
    effect_receipts = (
        api.ReceiptExpectation(
            "call-1", "edit_file", "ok", "edited app.py", ("app.py",)
        ),
        api.ReceiptExpectation("call-2", "run_command", "ok", "fixture bytes match"),
    )
    if add_notes:
        effect_receipts += (
            api.ReceiptExpectation(
                "call-3", "write_file", "ok", "wrote NOTES.md", ("NOTES.md",)
            ),
        )
    effect_receipts += (
        api.ReceiptExpectation(
            "call-4" if add_notes else "call-3",
            "run_command",
            "ok",
            "fixture bytes match",
        ),
    )
    case = api.EvaluationCase(
        final_contains="fixture validated",
        files=tuple(
            api.FileExpectation(path, content.decode()) for path, content in files
        ),
        results=(
            api.ResultExpectation("run_command", "fixture bytes match"),
            api.ResultExpectation("read_file", '"data":' + json.dumps(_DIAGNOSTIC)),
        ),
        receipts=effect_receipts,
    )
    model.first_call = None
    run = api.resume_with_steering(
        checkpoint,
        steering,
        _GuidedGenerator(generator, generator, actions),
        bounded,
        api.AgentLimits(max_steps=len(actions)),
    )
    report = api.evaluate_run(run, workspace, receipts, case)
    outcome = api.BranchOutcome(name, steering, run, report, generator.reuse)
    prefix = len(checkpoint.model.cached_token_ids)
    call = model.first_call
    if (
        call is None
        or call["offset"] != prefix
        or call["layer_offsets"] != list(checkpoint.model.layer_offsets)
        or outcome.reuse.reused_tokens != prefix
        or outcome.reuse.layer_offsets != checkpoint.model.layer_offsets
    ):
        raise api.AgentError("branch must start at the saved Week 2 cache offsets")
    if not ledger.read_bytes().startswith(initial_ledger):
        raise api.AgentError("branch receipt log lost its base receipts")
    if outcome.report.passed:
        observations = [event.result or "" for event in outcome.run.events]
        externalized = next(
            (
                value
                for value in observations
                if value.startswith("Tool result externalized:\n")
            ),
            "",
        )
        inline_bytes = len(externalized.encode())
        if not 0 < inline_bytes < len(log_bytes) or (
            store.read_range(artifact_id, 0, len(log_bytes)) != log_bytes
        ):
            raise api.AgentError(
                "the bounded observation did not preserve the fixture log"
            )
        _show(
            "context",
            {
                "branch": name,
                "artifact_bytes": len(log_bytes),
                "inline_bytes": inline_bytes,
                "range_bytes": range_end,
                "range_text": _DIAGNOSTIC.strip(),
            },
        )
    _show(
        "branch",
        {
            "name": name,
            "steering": steering,
            "cache": call,
            "reused_tokens": outcome.reuse.reused_tokens,
            "changed": workspace.modified_files,
            "receipts": _receipt_summary(_receipts(api, ledger)),
            "report": asdict(outcome.report),
            "passed": outcome.report.passed,
        },
    )
    return outcome


def run_live_fixture(api, model, tokenizer):
    """Run the supplied local scenario and accept one explicit passing branch."""
    with TemporaryDirectory(prefix="tiny-llm-week4-live-") as temporary:
        root = Path(temporary)
        paused, paused_cache, receipts = _start(api, model, tokenizer, root)
        base_ledger = (root / "base.jsonl").read_bytes()
        status = api.inspect_checkpoint(paused)
        _show(
            "pause",
            {
                "last_action": json.loads(status.last_action),
                "last_observation": status.last_evidence,
                "checkpoint_tokens": paused_cache.reuse.reused_tokens,
                "denied_unapproved_write": True,
                "receipts": _receipt_summary(receipts),
            },
        )
        print("resume> type resume to continue", flush=True)
        if input().strip() != "resume":
            raise api.AgentError(
                "type resume in this process to continue the checkpoint"
            )
        paused.validate()
        messages = [{"role": role, "content": text} for role, text in paused.messages]
        del paused_cache

        def count_tokens(prompt_messages):
            text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            return len(tokenizer.encode(text, add_special_tokens=False))

        compaction = api.compact_completed_interactions(
            messages, receipts[1:], count_tokens, keep_recent=0, result_preview_chars=48
        )
        if compaction.saved_tokens <= 0 or compaction.tokens_before != len(
            paused.model.cached_token_ids
        ):
            raise api.AgentError(
                "the saved validation evidence did not compact as expected"
            )
        _show(
            "context",
            {
                "tokens_before": compaction.tokens_before,
                "tokens_after": compaction.tokens_after,
                "receipt_ids": compaction.receipt_ids,
            },
        )
        compact_messages = list(compaction.messages)
        observed = _ObservedModel(model)
        generator = api.KvPrefixGenerator(observed, tokenizer, _MAX_TOKENS)
        compact_prefix = generator.save_checkpoint(compact_messages)
        if len(compact_prefix.cached_token_ids) != compaction.tokens_after:
            raise api.AgentError("the Week 2 prefix does not match the compacted view")
        checkpoint = api.create_checkpoint(_TASK, compact_messages, compact_prefix)
        base = root / "base"
        outcomes = tuple(
            _run_branch(
                api, observed, base, root, checkpoint, generator.fork(), name, steering
            )
            for name, steering in _BRANCHES.items()
        )
        if (
            (root / "base.jsonl").read_bytes() != base_ledger
            or (base / "app.py").read_bytes() != _EXPECTED_APP
            or (base / "NOTES.md").exists()
            or (root / "validate-only" / "NOTES.md").exists()
        ):
            raise api.AgentError(
                "a continuation changed files or receipts outside its own branch"
            )
        print("select> type validate-only or add-notes", flush=True)
        name = input().strip()
        if name not in _BRANCHES:
            raise api.AgentError("select validate-only or add-notes")
        selected = api.select_branch(outcomes, name)
        _show("selected", {"name": selected.name, "passed": selected.report.passed})
