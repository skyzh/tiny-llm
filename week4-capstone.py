"""Run the deterministic Week 4 coding-agent capstone."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

with redirect_stdout(io.StringIO()):
    import tiny_llm.agent as learner_agent


_TASK = "Set app.py to answer = 2, validate it, and inspect the build evidence."
_DIAGNOSTIC = "ERROR code=E42 dependency mismatch"
_BUILD_LOG = (
    "build start α\n"
    + "compile-unit-ok\n" * 96
    + _DIAGNOSTIC
    + "\n"
    + "link-unit-ok\n" * 96
    + "build end\n"
)


def _action(tool: str, **arguments: object) -> str:
    return json.dumps(
        {"tool": tool, **arguments}, sort_keys=True, separators=(",", ":")
    )


def _validation_command() -> tuple[str, ...]:
    program = (
        "from pathlib import Path; "
        "assert Path('app.py').read_text(encoding='utf-8') == 'answer = 2\\n'; "
        "log = Path('build-source.txt').read_text(encoding='utf-8'); "
        "Path('build.log').write_text(log, encoding='utf-8'); "
        "print('validation passed'); print(log, end='')"
    )
    return (sys.executable, "-c", program)


def _count_characters(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"]) for message in messages)


def _approve(_action: object) -> bool:
    return True


def _deny_extra_edit(agent_api: Any):
    def deny(_action: object):
        return agent_api.ApprovalDecision(False, "keep the requested answer at 2")

    return deny


class _ScriptedCheckpointModel:
    """Small deterministic checkpoint model with explicit prefix accounting."""

    def __init__(self, agent_api: Any, responses: tuple[str, ...]) -> None:
        self._agent_api = agent_api
        self._responses = responses
        self._response_index = 0
        self._checkpoint = None
        self._checkpoint_messages: tuple[tuple[str, str], ...] = ()
        self._restored = False
        self.calls: list[list[dict[str, str]]] = []
        self.reuse = agent_api.PrefixReuse(0, (), 0)

    @staticmethod
    def _tokens(messages: list[dict[str, str]]) -> tuple[int, ...]:
        return tuple(len(message["content"]) for message in messages)

    def __call__(self, messages: list[dict[str, str]]) -> str:
        if self._checkpoint is not None:
            if not self._restored:
                raise self._agent_api.AgentError(
                    "restore the checkpoint before generating"
                )
            prefix = tuple(
                (message["role"], message["content"])
                for message in messages[: len(self._checkpoint_messages)]
            )
            if prefix != self._checkpoint_messages:
                raise self._agent_api.AgentError(
                    "steered messages do not extend the saved prefix"
                )
        self.calls.append([dict(message) for message in messages])
        try:
            response = self._responses[self._response_index]
        except IndexError as error:
            raise self._agent_api.AgentError(
                "scripted model ran out of responses"
            ) from error
        self._response_index += 1
        return response

    def save_checkpoint(self, messages: list[dict[str, str]]):
        if self._checkpoint is not None:
            raise self._agent_api.AgentError("prefix checkpoint was already saved")
        tokens = self._tokens(messages)
        checkpoint = self._agent_api.ModelCheckpoint(
            len(messages),
            self._response_index,
            tokens,
            (len(tokens), len(tokens)),
        )
        self._checkpoint = checkpoint
        self._checkpoint_messages = tuple(
            (message["role"], message["content"]) for message in messages
        )
        return checkpoint

    def restore_checkpoint(self, checkpoint) -> None:
        if checkpoint != self._checkpoint:
            raise self._agent_api.AgentError(
                "model checkpoint does not match the saved prefix"
            )
        self._response_index = checkpoint.response_index
        self._restored = True
        reused = len(checkpoint.cached_token_ids)
        self.reuse = self._agent_api.PrefixReuse(
            reused, checkpoint.layer_offsets, reused
        )

    def fork(self, responses: tuple[str, ...]):
        if self._checkpoint is None:
            raise self._agent_api.AgentError("save a prefix checkpoint before forking")
        branch = _ScriptedCheckpointModel(self._agent_api, responses)
        branch._checkpoint = self._checkpoint
        branch._checkpoint_messages = self._checkpoint_messages
        branch._response_index = self._checkpoint.response_index
        return branch


def _receipt_ids(receipts: Any) -> list[str]:
    ids = []
    for number in range(1, 16):
        receipt = receipts.get(f"call-{number}")
        if receipt is None:
            break
        ids.append(receipt.receipt_id)
    return ids


def _report_branch(outcome: Any) -> dict[str, object]:
    return {
        "name": outcome.name,
        "steering": outcome.steering,
        "evaluation": {
            "passed": outcome.report.passed,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in outcome.report.checks
            ],
        },
        "reused_tokens": outcome.reuse.reused_tokens,
        "layer_offsets": list(outcome.reuse.layer_offsets),
        "avoided_prefill_tokens": outcome.reuse.avoided_prefill_tokens,
    }


def _json_payload(observation: str, prefix: str) -> dict[str, object]:
    if not observation.startswith(prefix):
        raise RuntimeError(f"expected observation prefix: {prefix!r}")
    payload = json.loads(observation.removeprefix(prefix))
    if not isinstance(payload, dict):
        raise RuntimeError("tool observation payload must be an object")
    return payload


def run_capstone(agent_api=learner_agent) -> dict[str, object]:
    """Compose the completed Days 1--9 APIs into one deterministic scenario."""

    command = _validation_command()
    with TemporaryDirectory(prefix="tiny-llm-week4-capstone-") as temporary:
        root = Path(temporary)
        base = root / "base"
        workspace_root = base / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / "app.py").write_text("answer = 1\n", encoding="utf-8")
        (workspace_root / "build-source.txt").write_text(_BUILD_LOG, encoding="utf-8")
        receipts = agent_api.ReceiptStore(base / "receipts.jsonl")
        workspace = agent_api.Workspace(
            agent_api.ToolPolicy(
                workspace_root,
                allow_writes=True,
                allowed_commands=(command,),
            ),
            lambda _action: True,
            receipts,
        )
        base_responses = (
            _action("read_file", path="app.py"),
            _action("edit_file", path="app.py", old="1", new="2"),
            _action("run_command", argv=list(command)),
            _action("read_file", path="build.log"),
        )
        base_model = _ScriptedCheckpointModel(agent_api, base_responses)
        checkpoint = agent_api.run_to_checkpoint(
            _TASK,
            base_model,
            workspace,
            after_tool_calls=4,
            limits=agent_api.AgentLimits(max_steps=4),
        )
        if workspace.modified_files != ("app.py",):
            raise RuntimeError("the base edit must be recorded exactly once")
        base_receipts = [receipts.get("call-1"), receipts.get("call-2")]
        if any(receipt is None for receipt in base_receipts):
            raise RuntimeError("the completed base effects need durable receipts")

        messages = [
            {"role": role, "content": content} for role, content in checkpoint.messages
        ]
        compaction = agent_api.compact_completed_interactions(
            messages,
            base_receipts,
            _count_characters,
            keep_recent=0,
            result_preview_chars=80,
        )
        if compaction.compacted_interactions != 2 or compaction.saved_tokens <= 0:
            raise RuntimeError(
                "capstone compaction must reclaim both completed effects"
            )

        compact_messages = [dict(message) for message in compaction.messages]
        prefix_model = _ScriptedCheckpointModel(agent_api, ())
        compact_model = prefix_model.save_checkpoint(compact_messages)
        compact_checkpoint = agent_api.create_checkpoint(
            _TASK, compact_messages, compact_model
        )
        status = agent_api.inspect_checkpoint(compact_checkpoint)

        branch_roots = {}
        branch_workspaces = {}
        branch_receipts = {}
        initial_receipt_bytes = (base / "receipts.jsonl").read_bytes()
        for name in ("validate-only", "try-extra-edit"):
            branch_root = root / name
            shutil.copytree(base, branch_root)
            branch_roots[name] = branch_root
            receipt_path = branch_root / "receipts.jsonl"
            branch_receipts[name] = agent_api.ReceiptStore(receipt_path)
            if branch_receipts[name].path != branch_root / "receipts.jsonl":
                raise RuntimeError("each branch must own its copied receipt log")
            if name == "validate-only":
                approve = _approve
            else:
                approve = _deny_extra_edit(agent_api)
            branch_workspaces[name] = agent_api.Workspace(
                agent_api.ToolPolicy(
                    branch_root / "workspace",
                    allow_writes=True,
                    allowed_commands=(command,),
                ),
                approve,
                branch_receipts[name],
            )
            if (branch_root / "receipts.jsonl").read_bytes() != initial_receipt_bytes:
                raise RuntimeError("each branch must begin with the copied receipts")

        case = agent_api.EvaluationCase(
            final_contains="validated branch",
            files=(agent_api.FileExpectation("app.py", "answer = 2\n"),),
            results=(agent_api.ResultExpectation("run_command", "validation passed"),),
            receipts=(
                agent_api.ReceiptExpectation(
                    "call-1", "edit_file", "ok", "edited app.py", ("app.py",)
                ),
                agent_api.ReceiptExpectation(
                    "call-2", "run_command", "ok", "validation passed"
                ),
                agent_api.ReceiptExpectation(
                    "call-3", "run_command", "ok", "validation passed"
                ),
            ),
        )
        validate_model = prefix_model.fork(
            (
                _action("run_command", argv=list(command)),
                json.dumps({"final": "validated branch"}, separators=(",", ":")),
            )
        )
        denied_model = prefix_model.fork(
            (
                _action("read_file", path="app.py"),
                _action("edit_file", path="app.py", old="2", new="3"),
                json.dumps({"final": "extra edit denied"}, separators=(",", ":")),
            )
        )
        passing = agent_api.run_branch(
            "validate-only",
            "validate without another edit",
            compact_checkpoint,
            validate_model,
            branch_workspaces["validate-only"],
            branch_receipts["validate-only"],
            case,
        )
        failing = agent_api.run_branch(
            "try-extra-edit",
            "try changing the answer again",
            compact_checkpoint,
            denied_model,
            branch_workspaces["try-extra-edit"],
            branch_receipts["try-extra-edit"],
            case,
        )
        outcomes = (passing, failing)
        if [outcome.name for outcome in outcomes if outcome.report.passed] != [
            "validate-only"
        ]:
            raise RuntimeError("exactly the named validation branch must pass")
        selected = agent_api.select_branch(outcomes, "validate-only")
        if any(
            getattr(event.action, "tool", None) == "edit_file"
            for event in selected.run.events
        ):
            raise RuntimeError("the selected branch must not replay the completed edit")
        selected_receipt_bytes = (
            branch_roots["validate-only"] / "receipts.jsonl"
        ).read_bytes()
        denied_receipt_bytes = (
            branch_roots["try-extra-edit"] / "receipts.jsonl"
        ).read_bytes()
        if not selected_receipt_bytes.startswith(initial_receipt_bytes):
            raise RuntimeError("selected receipts must retain the copied base")
        if denied_receipt_bytes != initial_receipt_bytes:
            raise RuntimeError("the denied branch must not add an effect receipt")
        if any(
            outcome.reuse.reused_tokens <= 0
            or outcome.reuse.reused_tokens != outcome.reuse.avoided_prefill_tokens
            for outcome in outcomes
        ):
            raise RuntimeError("both branches must report the same nonzero reuse fact")

        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        artifacts = agent_api.ArtifactStore(artifact_root)
        bounded = agent_api.BoundedEvidenceWorkspace(
            branch_workspaces[selected.name],
            artifacts,
            max_inline_bytes=512,
            preview_bytes=32,
            max_range_bytes=128,
        )
        build_bytes = _BUILD_LOG.encode("utf-8")
        digest = hashlib.sha256(build_bytes).hexdigest()
        artifact_id = f"artifact-{digest}"
        range_start = build_bytes.index(_DIAGNOSTIC.encode("utf-8"))
        range_end = range_start + len(_DIAGNOSTIC.encode("utf-8"))
        evidence_responses = iter(
            (
                _action("read_file", path="build.log"),
                _action(
                    "read_file",
                    path=artifacts.range_path(artifact_id, range_start, range_end),
                ),
                json.dumps(
                    {"final": "retrieved diagnostic E42"}, separators=(",", ":")
                ),
            )
        )
        evidence_run = agent_api.run_agent(
            "Inspect the selected build evidence and retrieve diagnostic E42.",
            lambda _messages: next(evidence_responses),
            bounded,
            agent_api.AgentLimits(max_steps=3),
        )
        if not evidence_run.completed or len(evidence_run.events) != 3:
            raise RuntimeError("the bounded evidence loop must complete in three steps")
        first_observation = evidence_run.events[0].result
        range_observation = evidence_run.events[1].result
        if not isinstance(first_observation, str) or not isinstance(
            range_observation, str
        ):
            raise RuntimeError("the bounded evidence observations must be strings")
        artifact = _json_payload(first_observation, "Tool result externalized:\n")
        selected_range = _json_payload(range_observation, "Artifact range:\n")
        if (
            selected_range.get("artifact_id") != artifact_id
            or selected_range.get("start") != range_start
            or selected_range.get("end") != range_end
            or selected_range.get("data") != _DIAGNOSTIC
        ):
            raise RuntimeError(
                "the selected artifact range must be the exact E42 bytes"
            )

        app_bytes = (branch_roots[selected.name] / "workspace" / "app.py").read_bytes()
        base_receipt_ids = _receipt_ids(receipts)
        selected_receipt_ids = _receipt_ids(branch_receipts[selected.name])
        return {
            "compaction": {
                "tokens_before": compaction.tokens_before,
                "tokens_after": compaction.tokens_after,
                "saved_tokens": compaction.saved_tokens,
                "receipt_ids": list(compaction.receipt_ids),
                "checkpoint_status": {
                    "task": status.task,
                    "last_action": status.last_action,
                    "last_evidence": status.last_evidence,
                    "next_step": status.next_step,
                },
            },
            "branches": [_report_branch(outcome) for outcome in outcomes],
            "selection": {
                "selected_name": selected.name,
                "app_py_sha256": hashlib.sha256(app_bytes).hexdigest(),
                "modified_files": {
                    "base": list(workspace.modified_files),
                    "selected_branch": list(
                        branch_workspaces[selected.name].modified_files
                    ),
                },
                "base_receipt_ids": base_receipt_ids,
                "selected_receipt_ids": selected_receipt_ids,
            },
            "artifact": {
                "artifact_id": artifact["artifact_id"],
                "sha256": artifact["sha256"],
                "full_byte_count": artifact["byte_count"],
                "model_visible_observation_byte_count": len(
                    first_observation.encode("utf-8")
                ),
                "omitted_interval": artifact["omitted_range"],
                "range_start": selected_range["start"],
                "range_end": selected_range["end"],
                "range_byte_count": selected_range["byte_count"],
                "range_text": selected_range["data"],
            },
        }


def main() -> int:
    """Print the stable capstone record for the completed learner package."""

    print(json.dumps(run_capstone(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
