import hashlib
import json
from argparse import Namespace

import pytest

from benches import week2_gpudebug as gpu


ROOT = gpu.ROOT


def test_capture_help_exposes_only_current_week2_checkpoint_names():
    assert gpu.KNOWN_CHECKPOINTS[-3:] == (
        "shared-input-qkv",
        "shared-input-gate-up-swiglu",
        "io-aware-dense-attention",
    )
    assert "decode-attention" not in gpu.KNOWN_CHECKPOINTS
    assert "split-k" not in gpu.KNOWN_CHECKPOINTS
    assert "long-context-attention" not in gpu.KNOWN_CHECKPOINTS
    assert "fused-gate-up" not in gpu.KNOWN_CHECKPOINTS


def _identity() -> dict:
    workload = gpu.workload_record("swiglu", "decode", 128)
    return {
        "source": {"tree": "tree"},
        "solution": "tiny_llm",
        "checkpoint": "swiglu",
        "phase": "decode",
        "workload": workload,
        "workload_id": gpu.canonical_hash(workload),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("solution", "tiny_llm_ref"),
        ("checkpoint", "simd-matmul"),
        ("phase", "prefill"),
        ("tokens", 32),
        ("tree", "other-tree"),
        ("workload_id", "0" * 64),
    ),
)
def test_evidence_identity_rejects_every_mismatch(field, value):
    reference = _identity()
    candidate = json.loads(json.dumps(reference))
    if field == "tokens":
        candidate["workload"]["tokens"] = value
    elif field == "tree":
        candidate["source"]["tree"] = value
    else:
        candidate[field] = value

    with pytest.raises(ValueError, match="mismatch"):
        gpu.validate_evidence_identity(reference, candidate)


def test_package_manifest_is_sorted_and_changes_with_package_identity(tmp_path):
    package = tmp_path / "trace.gputrace"
    (package / "nested").mkdir(parents=True)
    (package / "z.bin").write_bytes(b"z")
    (package / "nested/a.bin").write_bytes(b"a")

    first, first_bytes, first_count = gpu.package_manifest(package)
    second, second_bytes, second_count = gpu.package_manifest(package)
    assert first == second
    assert (first_bytes, first_count) == (second_bytes, second_count) == (2, 2)
    assert first.splitlines() == sorted(first.splitlines(), key=lambda line: line[66:])

    identities = {hashlib.sha256(first.encode()).hexdigest()}
    (package / "z.bin").write_bytes(b"changed")
    identities.add(
        hashlib.sha256(gpu.package_manifest(package)[0].encode()).hexdigest()
    )
    (package / "z.bin").rename(package / "renamed.bin")
    identities.add(
        hashlib.sha256(gpu.package_manifest(package)[0].encode()).hexdigest()
    )
    (package / "renamed.bin").unlink()
    identities.add(
        hashlib.sha256(gpu.package_manifest(package)[0].encode()).hexdigest()
    )
    (package / "added.bin").write_bytes(b"added")
    identities.add(
        hashlib.sha256(gpu.package_manifest(package)[0].encode()).hexdigest()
    )
    assert len(identities) == 5


def test_package_manifest_rejects_symlink_escape(tmp_path):
    package = tmp_path / "trace.gputrace"
    package.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    (package / "escape").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        gpu.package_manifest(package)


def test_parser_keeps_objects_around_banners_and_malformed_fragments():
    objects = gpu.iter_json_objects(
        'banner\n{broken\n{"timeline": [1], "unknown": true}\n'
        'noise {"counters": {"cycles": 7}}{"shaders": ["x"]}'
    )
    summary = gpu.summarize_gpudebug(objects, [])
    assert len(objects) == 3
    assert summary["timeline"]["available"]
    assert summary["shader_ranking"]["available"]
    assert summary["counters"]["available"]
    assert not summary["commands"]["available"]


def test_parser_and_missing_trees_fail_closed():
    with pytest.raises(ValueError, match="no usable JSON object"):
        gpu.summarize_gpudebug(gpu.iter_json_objects("banner {broken"), [])

    summary = gpu.summarize_gpudebug([{"status": "ok"}], [])
    for name in ("timeline", "shader_ranking", "counters", "commands"):
        assert summary[name]["available"] is False
        assert "reason" in summary[name]
    assert all(
        counter["available"] is False for counter in summary["counter_fields"].values()
    )


def test_reduce_needs_no_gpudebug_binary_and_refuses_overwrite(tmp_path, monkeypatch):
    manifest = tmp_path / "trace.sha256"
    manifest.write_text(f"{'a' * 64}  file\n")
    identity = _identity()
    metadata = {
        "schema_version": 1,
        "artifact_kind": "week2-metal-capture",
        **identity,
        "model": {"requested": "qwen3-4b", "resolved": "Qwen/Qwen3-4B-MLX-4bit"},
        "trace_package": {
            "basename": "trace.gputrace",
            "bytes": 1,
            "files": 1,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
    }
    metadata_path = tmp_path / "capture.json"
    metadata_path.write_text(json.dumps(metadata))
    profile_path = tmp_path / "profile.jsonl"
    profile_path.write_text('{"status": "ok"}\n')
    output = tmp_path / "reduced.json"
    args = Namespace(
        capture_metadata=metadata_path,
        manifest=manifest,
        profile_jsonl=profile_path,
        commands_jsonl=None,
        output=output,
    )
    monkeypatch.setenv("PATH", "")
    gpu.reduce(args)
    result = json.loads(output.read_text())
    assert result["gpudebug"]["counters"]["available"] is False

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gpu.reduce(args)


def test_reduce_rejects_workload_and_manifest_identity_mismatch(tmp_path):
    manifest = tmp_path / "trace.sha256"
    manifest.write_text(f"{'a' * 64}  file\n")
    identity = _identity()
    metadata = {
        **identity,
        "model": {},
        "trace_package": {"manifest_sha256": "0" * 64},
    }
    metadata["workload"]["tokens"] = 32
    metadata_path = tmp_path / "capture.json"
    metadata_path.write_text(json.dumps(metadata))
    profile_path = tmp_path / "profile.jsonl"
    profile_path.write_text("{}")
    args = Namespace(
        capture_metadata=metadata_path,
        manifest=manifest,
        profile_jsonl=profile_path,
        commands_jsonl=None,
        output=tmp_path / "out.json",
    )
    with pytest.raises(ValueError, match="workload identity mismatch"):
        gpu.reduce(args)

    metadata["workload_id"] = gpu.canonical_hash(metadata["workload"])
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        gpu.reduce(args)


def test_manifest_verification_rejects_traversal_even_with_matching_hash(tmp_path):
    manifest = tmp_path / "trace.sha256"
    manifest.write_text(f"{'a' * 64}  ../outside\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="traversal"):
        gpu.verify_manifest(manifest, digest)


def test_capture_checks_destinations_before_platform_or_model_work(tmp_path):
    output = tmp_path / "existing"
    output.write_text("preserve")
    args = Namespace(trace=output, metadata=tmp_path / "m", manifest=tmp_path / "h")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gpu.capture(args)


@pytest.mark.parametrize("roles", ((0, 1), (0, 2), (1, 2)))
@pytest.mark.parametrize("alias_kind", ("exact", "lexical", "symlink-parent"))
def test_capture_rejects_aliases_between_every_output_role(tmp_path, roles, alias_kind):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    outputs = [
        real_parent / "trace.gputrace",
        real_parent / "capture.json",
        real_parent / "trace.sha256",
    ]
    source, alias_role = roles
    if alias_kind == "exact":
        outputs[alias_role] = outputs[source]
    elif alias_kind == "lexical":
        outputs[alias_role] = (
            outputs[source].parent / "unused" / ".." / outputs[source].name
        )
    else:
        linked_parent = tmp_path / f"linked-{source}-{alias_role}"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        outputs[alias_role] = linked_parent / outputs[source].name

    args = Namespace(trace=outputs[0], metadata=outputs[1], manifest=outputs[2])
    with pytest.raises(ValueError, match="distinct"):
        gpu.capture(args)


@pytest.mark.parametrize("file_role", (1, 2))
@pytest.mark.parametrize("direction", ("inside-trace", "trace-inside"))
def test_capture_rejects_trace_package_overlap(tmp_path, file_role, direction):
    outputs = [
        tmp_path / "trace.gputrace",
        tmp_path / "capture.json",
        tmp_path / "trace.sha256",
    ]
    if direction == "inside-trace":
        outputs[file_role] = outputs[0] / outputs[file_role].name
    else:
        outputs[0] = outputs[file_role] / outputs[0].name

    args = Namespace(trace=outputs[0], metadata=outputs[1], manifest=outputs[2])
    with pytest.raises(ValueError, match="must not overlap"):
        gpu.capture(args)


def test_capture_distinct_outputs_preserve_roles_and_package_identity(tmp_path):
    trace = tmp_path / "trace.gputrace"
    metadata = tmp_path / "capture.json"
    manifest = tmp_path / "trace.sha256"
    assert gpu.normalize_capture_outputs(trace, metadata, manifest) == (
        trace.resolve(),
        metadata.resolve(),
        manifest.resolve(),
    )

    trace.mkdir()
    (trace / "capture.bin").write_bytes(b"capture")
    before = gpu.package_manifest(trace)
    manifest.write_text(before[0])
    metadata.write_text("{}\n")
    assert gpu.package_manifest(trace) == before


def test_checked_result_is_compact_bounded_and_explicit_about_missing_trees():
    path = (
        ROOT
        / "benchmark_results/m4-pro-qwen3-4b-week2-gpudebug-macos27-mlx-0.32.0.json"
    )
    payload = json.loads(path.read_text())
    assert payload["provenance"] == {
        "task": 525,
        "report_sha256": "ecfce490a9a0f31e6f114b170d5c5efffe933dbb4747d87f2542804c1a561fa6",
        "compact_bundle_sha256": "a06043a651c51eacf1b9dc2ecfaea56e174a7110d12013e78b7c2e7166487d62",
    }
    assert len(payload["decisions"]) == 5
    assert len(payload["captures"]) == 8
    split = payload["captures"][-1]
    assert split["checkpoint"] == "split-k"
    assert split["gpudebug"]["counters"]["available"] is False
    assert "no occupancy conclusion" in split["gpudebug"]["counters"]["reason"]
    encoded = path.read_text()
    assert ".gputrace" not in encoded
    assert "/Users/" not in encoded
    assert "capture command" not in encoded.lower()
