# Day 6: Control and Recovery

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

Long-running agents need a way to stop, accept a correction, and recover from a
file mutation that was interrupted at an awkward moment. Day 6 adds those
controls without treating a working directory, a process group, or a model
response as a safety boundary.

> **Implementation status:** The reference implementation, learner API surface,
> and focused tests are executable. Ctrl-C interruption is integrated with the
> CLI. Checkpoint, undo, branch, and steering operations are programmatic APIs;
> the CLI does not yet expose commands for them. The chapter remains WIP even
> though the checkpoint is executable.

## Check the Chapter

Implement the control and recovery APIs under `src/tiny_llm/agent/`, then run:

```bash
pdm run test --week 4 --day 6
```

Use `pdm run test-refsol --week 4 --day 6` for the supplied implementation. The
focused tests use temporary directories, scripted model responses, fake MLX
caches, and a fake command process. They do not load a model or execute a real
subprocess.

## One Cooperative Cancellation Signal

`CancellationToken` is a thread-safe, first-writer-wins signal. The first call
to `cancel(reason)` fixes the reason; later callers cannot replace it. Pass the
same token through the loop, generation session, and workspace:

```python
cancellation = CancellationToken()
workspace = Workspace(
    policy,
    session_log=session,
    cancellation=cancellation,
)
generation = GenerationSession(
    model,
    tokenizer,
    cache_factory,
    max_tokens=512,
    cancellation=cancellation,
)
result = run_agent(
    task,
    generation,
    workspace,
    session=session,
    cancellation=cancellation,
)
```

The implementation observes the signal at explicit safe boundaries:

- before preparing or sending a model request;
- between generated tokens in the course-model decode loop;
- before dispatching a tool;
- after a file intent is durable but before the atomic replace; and
- while polling an allowed command.

A cancellation raises `AgentInterrupted` with the stable reason and the phase
that observed it. `run_agent()` appends an `interrupted` event followed by a
non-completed `run_finished` event. A partially advanced generation cache is
discarded rather than reused. The CLI reports known file and command side
effects and exits with status 130 for an interrupted run.

If a generic backend finishes a complete response just before cancellation is
observed, the loop records that response for cache/audit consistency and then
appends an explicit error observation saying that no action from it was
executed. Resuming the session therefore cannot mistake the discarded action
for work that should be replayed.

Cancellation is cooperative, not preemptive. An MLX kernel already executing on
the device must return before Python can observe the token. The stateless
`--solution mlx` compatibility backend has no per-token course loop, so it may
also observe Ctrl-C only when its backend yields control.

## Durable Next-Turn Steering

`SteeringHandle` first records a non-blank correction as a durable
`steering_queued` audit event:

```python
steering = SteeringHandle(session)
steering.submit("Do not change the schema; fix the parser instead.")
```

A model request already in progress has an immutable message snapshot. After
the active assistant/tool turn closes, the loop materializes the queue entry as
one semantic `user_message` with `kind="steering"` and a source-event ID. The
correction therefore appears exactly once, after the observation from the tool
that did not see it, when the loop rebuilds context for the next request. It
does not cancel the active tool. Interrupt first when the active work must stop
immediately.

If the concurrent response is a final action, queued steering wins: the loop
records that the final was not accepted, delivers the correction, and performs
another bounded model turn.

This checkpoint provides the durable API and next-turn semantics, not a
concurrent terminal-input broker. The current CLI has no way to type a steering
message while generation or a tool is in progress.

## Journal Before Replacing a File

When a `Workspace` has a matching session log, each approved `write_file` or
`edit_file` records a bounded `mutation_intent` before the atomic replace. The
intent contains the normalized path, the before-image and mode when the file
exists, the intended post-write mode, and SHA-256 hashes of the before and
intended after bytes. The log is flushed and synchronized before the workspace
is touched. A `mutation_committed` event follows only after both the current
file hash and mode match the intent.

The final write traverses from an open, identity-checked workspace-root file
descriptor. Every parent is opened with `O_NOFOLLOW`; the target is revalidated
relative to one held parent descriptor. Installation is no-replace. For an
existing file, the implementation first moves the current name to a private
backup, verifies that exact entry, installs the new file only if the destination
is still absent, and preserves the backup after either success or failure. A
writer that opened the old inode before the rename can still append through its
held descriptor, so deleting that backup would risk deleting bytes written
concurrently. These protected `.tiny-llm-agent-*.bak` files are retained
for manual inspection and surfaced through `AgentRun.retained_recovery_files`
and the CLI side-effect summary. The implementation never follows a
parent-directory swap outside the workspace. Reporting is conservative: an
interruption before the backup rename can leave a reserved path in the tuple
that does not yet exist; every reported path that does exist requires manual
inspection.

These checks protect the approved target and its directory lineage from normal
concurrent edits. They are not a security boundary against another process
running as the same user that deliberately discovers and races the random
`.tiny-llm-agent-*` or `.tiny-llm-undo-*` internal names. Portable POSIX Python
does not provide an unlink-if-inode operation; cleanup therefore verifies an
internal entry immediately before unlinking it, and preserves leftovers when
that identity check fails. Use a container or virtual machine when the other
processes sharing the account are not trusted.

If the process stops between those two events, constructing the workspace or
calling `recover_pending()` classifies the current content-and-mode fingerprint:

| Current file state | Recovery classification |
| --- | --- |
| intended after hash and mode | `committed` |
| recorded before hash and mode, including still-missing new file | `not_applied` |
| another hash or mode, unsafe type, or unreadable target | `conflict` |

Recovery only appends the classification. It never writes, removes, or restores
a file. The CLI reports conflicts found at startup. This distinction matters at
the atomic-replace boundary: cancellation may arrive after the replace has
completed but before its commit event is durable. Fingerprint reconciliation
records the completed replace instead of replaying it or guessing. A chmod-only
change does not match either recorded fingerprint and is therefore a conflict.

The before-image is bounded UTF-8 text and the session transcript is sensitive.
Crash recovery is durable only for a persisted session; `--no-session` keeps
the journal in memory and cannot recover it after process exit.

## Checkpoints and Default-No Undo

`MutationJournal.create_checkpoint(name)` records a branch-local event boundary
after resolving pending intents. `plan_undo(checkpoint)` never changes workspace
bytes: it first records classifications for any pending intents, collapses
committed post-checkpoint mutations into one restore per path, expects the latest
exact content-and-mode fingerprint, and carries warnings for command activity
that the file journal cannot reverse.

Undo remains a separate, explicit side effect:

```python
assert workspace.journal is not None
checkpoint = workspace.journal.create_checkpoint("before parser edit")

# ... approved write_file or edit_file calls ...

plan = workspace.journal.plan_undo(checkpoint)

def confirm_undo(_plan):
    try:
        answer = input("Apply this undo plan? y/N ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}

result = workspace.journal.apply_undo(plan, confirm=confirm_undo)
```

Omitting `confirm` denies the undo. An empty response, EOF, interruption, or
anything other than an explicit `y` or `yes` must also deny it.
Before prompting, the implementation preflights the complete plan; any content
or mode conflict, including a chmod-only change, leaves every file unchanged.
It restores the original content and mode of existing files and removes files
that the agent created.

Each path is checked again immediately before its restore or removal. Another
process can still race after the whole-plan preflight: earlier paths may already
be restored before a later path reports a conflict. The durable undo events and
`UndoResult` report that partial outcome honestly. There is no redo operation in
this checkpoint. Each restore/removal has its own write-ahead change ID and
expected result content-and-mode fingerprint, and names the exact
mutation-intent IDs it covers. After a crash, retry reconciles a started change
as applied, not applied, or conflicting before it asks whether to continue; it
never replays an already-applied change or treats a later mutation as part of an
earlier reviewed plan.

Undo uses the same root-anchored directory traversal. A removal first renames
the exact candidate to a private quarantine and verifies its bytes and mode
before making the original path absent. A restore quarantines and verifies the
current entry, then uses a no-replace install. The public-target quarantine or
backup is retained after every successful operation because another process may
still hold a writable descriptor to that inode. Its exact protected path is
chosen and journaled before mutation, then returned in
`UndoResult.retained_recovery_files`, including after crash reconciliation. If
another actor substitutes a file, the operation reports a conflict and restores
or preserves the quarantined bytes instead of deleting or overwriting the
substitute. The tuple conservatively includes every started operation's reserved
evidence path, so an entry may not exist if interruption happened before the
quarantine rename; every entry that does exist requires manual inspection. The
agent-created and initially validated content is bounded, but a
writer holding the old descriptor can keep growing a retained inode. Repeated
copies can therefore consume disk; an operator may inspect and remove the named
`.tiny-llm-undo-*` evidence later.

Commands are not file mutations. A command after the checkpoint adds a warning,
but its filesystem, network, process, or other host effects are not tracked and
cannot be undone by `MutationJournal`. A durable command ID is recorded before
launch, cancellation is checked again before `Popen`, and normal completion is
linked by `command_finished`. On restart, any unmatched launch record is treated
as both side-effecting and cleanup-uncertain.

## Branch Conversation, Not the Workspace

`SessionStore.branch(parent_session_id, at_event_id)` creates a new session at a
closed ancestor boundary. The new session records the parent session and event,
copies the semantic user, assistant, tool, and run events needed to rebuild
context, retains source event IDs for audit, and ends construction with a
durable `branch_completed` marker. An interrupted partial branch is rejected
and ignored by newest-session selection. A parent with an unmatched tool,
mutation, undo change, or command—or a recorded recovery conflict—cannot be
branched until its shared-workspace state is resolved. Branching does not alter
the parent JSONL file.

These operations remain deliberately separate:

- **Compaction** changes which durable history is rendered to the model.
- **Session branching** starts a new conversational lineage at an ancestor.
- **Workspace undo** conflict-checks and restores journaled file mutations.

Branching never restores files, and undo never erases the record of an earlier
attempt. Select and apply an undo plan explicitly if a new conversation branch
also needs older workspace bytes.

## Steering, Branching, and the Live Cache

Steering, compaction, and branching can all change an older portion of the
rendered prompt. They use the same Day 4 cache rule: `GenerationSession`
tokenizes the new messages, finds the token longest common prefix with the live
cache, rewinds every layer to that boundary, and prefills the new suffix. If
layer offsets disagree or rewind fails, it drops all layers and starts cold.

The branch and checkpoint events do not store K/V tensors or promise a cached
token length. A restarted process always rebuilds from semantic events.
Persistent K/V snapshots and branch-specific cache registries are deferred.

## Command and Recovery Boundaries

An allowed command is launched without a shell and in a new process group. With
a cancellation token, the runner polls at bounded intervals and attempts to
kill and reap that group on cancellation or timeout. This is cleanup, not
isolation: a child can create another session, an approved executable can access
paths outside the working directory, and effects that happened before the stop
remain real. Use a disposable workspace even when every call has a `y/N`
prompt.

The Day 6 checkpoint intentionally defers:

- container or virtual-machine isolation;
- concurrent TTY steering;
- persistent K/V cache snapshots;
- redo and transactionally atomic multi-file undo; and
- held-out evaluation and correctness grading.

`AgentRun.completed` still means that the model produced a valid final action.
It does not mean the task is correct or that a validation command passed.

## Exercise

1. Cancel before a model request and verify that the model and cache factory are
   never called.
2. Cancel during fake decoding and verify that partial caches are released.
3. Submit steering during one turn and inspect it once in the next context.
4. Interrupt before and after an atomic replace, then compare the two recovery
   classifications without changing the file during recovery.
5. Create a checkpoint, journal two edits and one new file, review an undo plan,
   deny it by default, and then apply it with explicit confirmation.
6. Change one planned file externally and verify that whole-plan preflight
   leaves the other files untouched.
7. Branch at a closed ancestor and confirm that parent transcript and workspace
   bytes remain unchanged.
8. Cancel the supplied fake command process and inspect its cleanup and
   untracked-side-effect markers.

Day 7 adds deterministic held-out task packages and separates protocol
completion from measured correctness.

{{#include copyright.md}}
