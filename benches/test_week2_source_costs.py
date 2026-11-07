"""Scalar/CLI contracts only: these tests must not load Apple or MLX code."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benches import week2_source_costs as costs


def scalar_profile():
    row = {
        "file": "/kernel.metal",
        "line": 1,
        "function": "kernel",
        "value0": 3.0,
        "value1": 1.0,
        "num_instructions": 2,
        "source": "DO_NOT_EXPORT_SOURCE",
    }
    return {
        "binaries": [
            {
                "index": 0,
                "value0": 6.0,
                "value1": 4.0,
                "instruction_cost_sum": 10.0,
                "unmapped_cost": 6.0,
                "rows": [row, dict(row), dict(row)],
                "leaf_rows": [row],
            }
        ]
    }


class SourceCostsTests(unittest.TestCase):
    def setUp(self):
        self.owned = tempfile.TemporaryDirectory(prefix="tiny-llm-source-costs-")
        self.addCleanup(self.owned.cleanup)
        self.root = Path(self.owned.name)
        self.capture = self.root / "saved.gputrace"
        session = self.capture / "one.gpuprofiler_raw"
        session.mkdir(parents=True)
        self.stream = session / "streamData"
        self.stream.write_bytes(b"scalar test placeholder, not a real capture")

    def test_existing_and_dangling_output_refuse_before_private_code(self):
        for dangling in (False, True):
            output = self.root / ("dangling" if dangling else "existing")
            if dangling:
                output.symlink_to(self.root / "absent")
            else:
                output.mkdir()
            with mock.patch.object(
                costs, "check_compatibility", side_effect=AssertionError("not reached")
            ):
                with self.assertRaises(FileExistsError):
                    costs.main([str(self.capture), "--out", str(output)])

    def test_input_output_overlap_and_ambiguous_session_refuse(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            costs.checked_paths(self.capture, self.capture / "result")
        other = self.capture / "two.gpuprofiler_raw"
        other.mkdir()
        (other / "streamData").write_bytes(b"another placeholder")
        with self.assertRaisesRegex(ValueError, "one embedded"):
            costs.checked_paths(self.capture, self.root / "result")

    def test_manifest_identity_and_escape(self):
        sidecar = self.root / "source.txt"
        sidecar.write_text("private source line\n")
        raw = sidecar.read_bytes()
        entry = {
            "path": sidecar.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest = self.root / "sources-manifest.json"
        manifest.write_text(json.dumps({"/kernel.metal": entry}))
        verified = costs.verify_sources(manifest)
        self.assertEqual(verified["files"]["/kernel.metal"]["lines"], 1)
        self.assertNotIn("private source line", json.dumps(verified))
        sidecar.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            costs.verify_sources(manifest)
        nested = self.root / "nested"
        nested.mkdir()
        escaped = nested / "manifest.json"
        escaped.write_text(
            json.dumps({"/kernel.metal": {**entry, "path": "../source.txt"}})
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            costs.verify_sources(escaped)

    def test_missing_manifest_is_unavailable_not_verified_zero(self):
        value = costs.verify_sources(self.root / "missing.json")
        self.assertFalse(value["available"])
        self.assertIn("reason", value)
        self.assertNotIn("files", value)

    def test_overlap_rows_are_separate_and_source_text_is_excluded(self):
        rows = costs.cost_rows(
            scalar_profile(), {"files": {"/kernel.metal": {"lines": 1}}}
        )
        self.assertEqual([r["mode"] for r in rows].count("inclusive"), 3)
        self.assertEqual([r["mode"] for r in rows].count("outer-callsite"), 1)
        self.assertEqual(rows[0]["percent_of_binary_weighted_cost"], 40)
        self.assertTrue(all(r["location"] == "source-line" for r in rows))
        self.assertNotIn("DO_NOT_EXPORT_SOURCE", json.dumps(rows))
        unknown = costs.cost_rows(scalar_profile(), {"available": False})
        self.assertTrue(all(r["location"] == "unverified-location" for r in unknown))

    def test_conservation_and_invalid_costs_refuse(self):
        variants = []
        bad = scalar_profile()
        bad["binaries"][0]["instruction_cost_sum"] = 9
        variants.append(bad)
        bad = scalar_profile()
        bad["binaries"][0]["unmapped_cost"] = 5
        variants.append(bad)
        for value in (float("nan"), float("inf"), -1):
            bad = scalar_profile()
            bad["binaries"][0]["value0"] = value
            variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                costs.cost_rows(bad, {})

    def test_compatibility_refuses_mismatch_before_decoder_import(self):
        with (
            mock.patch.object(sys, "version_info", (3, 12, 13)),
            mock.patch.object(costs.platform, "system", return_value="Darwin"),
            mock.patch.object(costs.platform, "machine", return_value="arm64"),
            mock.patch.object(
                costs.platform, "mac_ver", return_value=("27.0.1", (), "")
            ),
        ):
            binary = self.root / "framework"
            binary.write_bytes(b"not the pinned framework")
            with mock.patch.object(costs, "FRAMEWORK", binary):
                with self.assertRaisesRegex(RuntimeError, "pinned Xcode"):
                    costs.check_compatibility()

    def test_real_cli_writes_code_free_outputs_with_injected_decoder(self):
        output = self.root / "result"

        def decoder(stream, target):
            self.assertEqual(stream, self.stream.resolve(strict=True))
            (target / "line-cost-data.json").write_text(json.dumps(scalar_profile()))
            (target / "decoded-profile-types.json").write_text(
                json.dumps({"pipeline_timings": []})
            )

        with (
            mock.patch.object(costs, "check_compatibility"),
            mock.patch.object(costs, "decode_existing", side_effect=decoder),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            costs.main([str(self.capture), "--out", str(output)])
        summary = json.loads((output / "summary.json").read_text())
        self.assertFalse(summary["new_capture_or_GPU_replay"])
        self.assertFalse(summary["source_text_exported"])
        self.assertFalse(summary["sources"]["available"])
        self.assertNotIn(
            "DO_NOT_EXPORT_SOURCE", (output / "source-costs.json").read_text()
        )
        self.assertNotIn(
            "DO_NOT_EXPORT_SOURCE", (output / "source-costs.csv").read_text()
        )

    def test_both_help_paths_are_backend_free(self):
        from benches import profile_week2_kernels as profile

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as source_exit:
                costs.main(["--help"])
            with mock.patch.object(sys, "argv", ["profile-week2-kernels", "--help"]):
                with self.assertRaises(SystemExit) as replay_exit:
                    profile.main()
        self.assertEqual(source_exit.exception.code, 0)
        self.assertEqual(replay_exit.exception.code, 0)


class CaptureContractTests(unittest.TestCase):
    def test_capture_cli_exposes_flags_and_refuses_small_capacity_early(self):
        from benches import week2_gpudebug as gpu

        argv = [
            "capture",
            "--solution",
            "tiny_llm",
            "--model",
            "qwen3-0.6b",
            "--checkpoint",
            "capacity-cache",
            "--phase",
            "decode",
            "--tokens",
            "128",
            "--trace",
            "saved.gputrace",
            "--metadata",
            "capture.json",
            "--manifest",
            "manifest.sha256",
            "--prefill-logits",
            "last",
            "--cache-capacity",
            "256",
        ]
        args = gpu.build_parser().parse_args(argv)
        self.assertEqual((args.prefill_logits, args.cache_capacity), ("last", 256))
        with (
            mock.patch.object(sys, "argv", ["capture-week2", *argv[:-1], "127"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                gpu.main()
        self.assertEqual(error.exception.code, 2)

    def test_capture_identity_covers_capacity_and_logits(self):
        from benches import week2_gpudebug as gpu

        record = gpu.workload_record(
            "capacity-cache", "decode", 128, prefill_logits="last", cache_capacity=256
        )
        self.assertEqual(record["request_bound"], 256)
        self.assertEqual(record["cache_capacity"], 256)
        self.assertEqual(record["prefill_logits"], "last")
        self.assertIsNone(
            gpu.workload_record("kv-cache", "decode", 128)["cache_capacity"]
        )
        self.assertEqual(
            gpu.workload_record("capacity-cache", "decode", 128)["prefill_logits"],
            "all",
        )
        for kwargs in ({"cache_capacity": 127}, {"prefill_logits": "unsupported"}):
            with self.assertRaises(ValueError):
                gpu.workload_record("capacity-cache", "decode", 128, **kwargs)

    def record_forward(self, *, phase="decode", trace="saved.gputrace", fail=False):
        from benches import week2_gpudebug as gpu

        events = []

        class Input:
            shape = (1, 128)
            kind = "full"

            def __getitem__(self, index):
                part = Input()
                part.kind = "prefix" if index[1].stop == -1 else "last"
                return part

        class Cache:
            def __init__(self, index):
                self.index = index

            def release(self):
                events.append(("release", self.index))
                if fail and self.index == 0:
                    raise OSError("release0")

        class Model:
            def create_kv_cache(self, *, capacity):
                events.append(("create", capacity))
                return [Cache(0), Cache(1)]

            def __call__(self, query, offset, cache, *, logits_to_keep):
                events.append(("forward", query.kind, offset, logits_to_keep))
                if fail and query.kind == "last":
                    raise ValueError("forward")
                return query.kind

        class Metal:
            def start_capture(self, path):
                events.append(("start", path))

            def stop_capture(self):
                events.append(("stop",))
                if fail:
                    raise RuntimeError("stop")

        try:
            gpu.capture_forward(
                Model(),
                Input(),
                phase=phase,
                capacity=256,
                logits_to_keep=1,
                evaluate=lambda value: events.append(("eval", value)),
                metal=Metal(),
                trace=trace,
            )
        except Exception as error:
            return events, error
        return events, None

    def test_real_forward_seeds_outside_capture_then_releases_all(self):
        events, error = self.record_forward()
        self.assertIsNone(error)
        self.assertEqual(
            events,
            [
                ("create", 256),
                ("forward", "prefix", 0, 1),
                ("eval", "prefix"),
                ("start", "saved.gputrace"),
                ("forward", "last", 127, 1),
                ("eval", "last"),
                ("stop",),
                ("release", 0),
                ("release", 1),
            ],
        )

    def test_warmup_uses_same_public_api_without_capture(self):
        events, error = self.record_forward(phase="prefill", trace=None)
        self.assertIsNone(error)
        self.assertEqual(
            events,
            [
                ("create", 256),
                ("forward", "full", 0, 1),
                ("eval", "full"),
                ("release", 0),
                ("release", 1),
            ],
        )

    def test_forward_stop_and_release_failures_all_remain_observable(self):
        from benches.week2_profile_adapter import ReplayCleanupError

        events, error = self.record_forward(fail=True)
        self.assertIsInstance(error, ReplayCleanupError)
        self.assertIsInstance(error.primary, ReplayCleanupError)
        self.assertIsInstance(error.primary.primary, ValueError)
        self.assertIsInstance(error.primary.cleanup, RuntimeError)
        self.assertEqual([str(e) for e in error.cleanup.causes], ["release0"])
        self.assertEqual(events[-3:], [("stop",), ("release", 0), ("release", 1)])


if __name__ == "__main__":
    unittest.main()
