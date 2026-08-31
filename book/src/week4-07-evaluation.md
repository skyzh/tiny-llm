# Day 7: Evaluate Observable Outcomes

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

The first six days built a small coding-agent loop, connected it to a workspace,
recorded approved effects, and added checkpoint, compaction, and steering
boundaries. The closing question is practical: did one run produce the outcome
the task asked for?

Day 7 answers with a small deterministic evaluation harness. It checks declared
observable facts: the final answer, exact file contents, tool-result evidence,
and named durable receipts. It does not grade hidden reasoning or require one
exact transcript shape.

## The Starter Surface

The final Day 9 scaffold already exposes branching and bounded-evidence
declarations. Leave those future TODO bodies alone. Day 7 owns one module:

| File | Public names | Purpose |
| --- | --- | --- |
| `src/tiny_llm/agent/evaluation.py` | `FileExpectation`, `ResultExpectation`, `ReceiptExpectation`, `EvaluationCase`, `EvaluationCheck`, `EvaluationReport`, `evaluate_run` | Describe required observable facts and produce a stable pass/fail report. |
| `src/tiny_llm/agent/__init__.py` | the names above | Complete the Day 7 exports within the final scaffold. |

Run the cumulative learner checkpoint:

```bash
pdm run test --week 4 --day 7
```

Use this command for the supplied implementation:

```bash
pdm run test-refsol --week 4 --day 7
```

Before you implement the TODOs, all seven Day 7 tasks are expected to fail. The
command force-refreshes and runs the supplied learner tests for Days 1--7
together.

The retired `pdm run evaluate-agent` launcher and its unconsumed static-grader
packages are no longer part of the repository. The supported Day 7 learner
checkpoint is `evaluate_run` through the cumulative test above.

## Task 1: Declare the Outcome

An evaluation case names only the facts that matter for this task:

```python
case = EvaluationCase(
    final_contains="validated",
    files=(FileExpectation("app.py", "answer = 2\n"),),
    results=(
        ResultExpectation("run_command", "validation passed"),
    ),
    receipts=(
        ReceiptExpectation(
            "call-1",
            "edit_file",
            "ok",
            "edited app.py",
            ("app.py",),
        ),
        ReceiptExpectation(
            "call-2",
            "run_command",
            "ok",
            "validation passed",
        ),
    ),
)
```

The two receipt IDs are explicit inputs chosen for this deterministic case. A
different harness could discover or correlate effect records another way; Day
7 does not claim that every evaluation needs fixed call IDs.

Reject an invalid specification before evaluating: required strings cannot be
blank, file paths must be relative and remain inside the workspace, file paths
and receipt IDs must be unique within their groups, and a receipt exit state is
either `ok` or `error`.

## Task 2: Check the Final Answer

Implement:

```python
evaluate_run(run, workspace, receipts, case) -> EvaluationReport
```

The first check requires a completed run whose public final answer contains the
declared substring. This is a small grounding signal, not a prose grader. Do
not inspect hidden reasoning, demand exact wording, or ask another model to
judge the answer.

## Task 3: Check Workspace State

For each `FileExpectation`, resolve the declared path through the existing
`Workspace` boundary, read it as UTF-8, and compare the exact content. Emit one
named check such as `file:app.py`.

A missing file, directory, unreadable file, or content mismatch is observed
evidence that failed. Return a failed check instead of aborting the whole
report. That is different from an invalid case definition, which is rejected.

## Task 4: Match Result Evidence Without Grading a Trace

Each `ResultExpectation` requires at least one public `AgentEvent` with the
declared tool and result substring. Search the events as a set of observable
facts. Do not require an exact number of turns or an exact event order.

This matters because two useful runs may phrase their final answers differently
or place unrelated read-only observations in a different order while producing
the same required outcome.

## Task 5: Check Named Durable Receipts

Use the public `ReceiptStore` passed to `evaluate_run`; do not reach through
private workspace state. For every declared call ID, require the expected tool,
exit state, result substring, and exact changed-artifact tuple.

Absent receipts, tampered persistent logs, mismatched fields, and lookup errors
become failed checks. Evaluation must not append a receipt or rerun an effect.

## Task 6: Produce a Stable Report

Return checks in one deterministic order:

1. final answer;
2. files in case order;
3. results in case order; and
4. receipts in case order.

`EvaluationReport.passed` is true only when every check passes. Its `render()`
method should produce a compact summary:

```text
evaluation: PASS
- final: PASS (required final observed)
- file:app.py: PASS (content matches)
- result:run_command: PASS (required result observed)
- receipt:call-1: PASS (receipt facts match)
- receipt:call-2: PASS (receipt facts match)
```

Stable names and ordering make failures easy to inspect without turning the
test into an exact transcript comparison.

## Task 7: Keep Evaluation Read-Only

The focused scenario asks the existing agent loop to set `answer = 2` in
`app.py`, run the exact configured validation command, and finish. The harness
then checks the final answer, final file bytes, validation result, edit receipt,
and command receipt.

Calling `evaluate_run` must leave the run, workspace bytes, modified-file list,
approval history, and receipt bytes unchanged. It invokes no model, tool, or
approval callback. Independent wrong final, file, result, and receipt facts
each fail their own named check. An alternate final phrase and event order still
pass when the required behavioral evidence is present.

## Checkpoint

You can now turn one coding-agent run into a deterministic report over declared
observable outcomes. This harness samples the facts a particular case names. It
does not prove general task correctness, model quality, security, or production
safety, and it is not a hidden grader, benchmark suite, or LLM-as-judge system.

You now have the evidence needed to compare continuations. Continue with [Day
8: Fork, Steer, and Select](week4-08-fork-steer-select.md) to reuse one real
token/KV prefix, steer two isolated branches, and explicitly choose a passing
outcome without rewinding completed effects.

This evaluator is exercised as a library boundary. After Day 9, the supplied
Week 4 capstone feeds its report into branch selection and then bounded
evidence retrieval.

{{#include copyright.md}}
