"""Focused backend-free checks for request bounds and allocation reporting."""

import ast
import contextlib
import io
import json
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import patch

from benches import bench_course_progression as progression
from benches.bench_request_metrics import (
    PEAK_MLX_SCOPE,
    REQUEST_ELAPSED_SCOPE,
    measure_request,
    request_cache_capacity,
)


class TextOutput:
    def __init__(self):
        self.text = None

    def exists(self):
        return False

    def is_symlink(self):
        return False

    def write_text(self, text):
        self.text = text


class Progress:
    def update(self, count):
        pass

    def set_postfix(self, values):
        pass

    def close(self):
        pass


def worker_definitions():
    # Compile the real request/main definitions, excluding all module imports.
    # Inert public inputs below exercise their wiring without loading a backend.
    source = Path(__file__).with_name("bench.py")
    tree = ast.parse(source.read_text())
    names = {"BenchRequest", "run_one_request_week2", "main", "safe_div"}
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {"dataclass": dataclass, "__name__": "request_worker_witness"}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


class RequestMetricsTests(unittest.TestCase):
    def test_selected_checkpoint_capacity_and_other_routes(self):
        request = SimpleNamespace(prompt_token_ids=list(range(128)), max_new_tokens=129)
        for checkpoint in (
            None,
            "selected",
            "capacity-cache",
            "quantized-matvec",
            "simd-matmul",
            "rmsnorm",
            "rope",
            "swiglu",
            "tiled-prefill",
        ):
            with self.subTest(checkpoint=checkpoint):
                self.assertEqual(
                    request_cache_capacity("week2", checkpoint, request), 257
                )
        for loader, checkpoint in (
            ("week2", "kv-cache"),
            ("week1", None),
            ("week3", None),
        ):
            self.assertIsNone(request_cache_capacity(loader, checkpoint, request))

    def test_real_request_forwards_capacity_and_releases_on_failure(self):
        namespace = worker_definitions()
        events = []
        cache = SimpleNamespace(release=lambda: events.append("release"))
        model = SimpleNamespace(
            create_kv_cache=lambda **kw: events.append(kw) or [cache]
        )
        namespace.update(
            mx=SimpleNamespace(
                array=lambda *a, **kw: SimpleNamespace(size=128), int32="int32"
            ),
            perf_counter=lambda: 0,
            sample_next_week2=lambda *a, **kw: (_ for _ in ()).throw(
                ValueError("fixture")
            ),
        )
        request = namespace["BenchRequest"]([1] * 128, 129)
        for capacity, expected in ((257, {"capacity": 257}), (None, {})):
            events.clear()
            with self.assertRaisesRegex(ValueError, "fixture"):
                namespace["run_one_request_week2"](
                    model, request, cache_capacity=capacity
                )
            self.assertEqual(events, [expected, "release"])

    def test_real_main_routes_capacity_and_saves_worker_allocation(self):
        for solution, loader, checkpoint, expected in (
            ("ref", "week2", "capacity-cache", 257),
            ("tiny_llm", "week2", "kv-cache", None),
            ("ref", "week2", None, 257),
            ("ref", "week3", None, None),
            ("ref", "week1", None, "other"),
            ("mlx", "week2", None, "other"),
        ):
            with self.subTest(solution=solution, loader=loader, checkpoint=checkpoint):
                namespace = worker_definitions()
                output = TextOutput()
                args = SimpleNamespace(
                    solution=solution,
                    loader=loader,
                    week2_checkpoint=checkpoint,
                    model="fixture",
                    device="gpu",
                    prefill_logits="all" if loader == "week1" else "last",
                    seed=0,
                    batch_decode=False,
                    disable_paged_attention=False,
                    week3_inherit_course_projections=False,
                    num_seqs=1,
                    warmup=0,
                    min_input_len=128,
                    max_input_len=128,
                    min_output_len=129,
                    max_output_len=129,
                    json_output=output,
                    batch_size=1,
                    prefill_step=128,
                )
                calls = []

                def run(model, request, **kwargs):
                    calls.append(kwargs)
                    return 129, 0.25, 0.75

                memory = SimpleNamespace(
                    gpu="gpu",
                    cpu="cpu",
                    stream=lambda device: contextlib.nullcontext(),
                    synchronize=lambda: None,
                    reset_peak_memory=lambda: None,
                    get_peak_memory=lambda: 123456,
                )
                ticks = iter([1.0, 2.0])
                model = SimpleNamespace(args=SimpleNamespace(vocab_size=1024))
                models = SimpleNamespace(dispatch_model=lambda *a, **kw: model)
                namespace.update(
                    parse_args=lambda: args,
                    validate_args=lambda args: None,
                    Random=Random,
                    load_solution_modules=lambda solution: (
                        solution,
                        models,
                        None,
                        None,
                    ),
                    shortcut_name_to_full_name=lambda name: name,
                    load=lambda name: (model, SimpleNamespace(eos_token_id=0)),
                    mx=memory,
                    run_one_request_week1=run,
                    run_one_request_week2=run,
                    run_one_request_mlx=run,
                    build_requests=lambda **kw: [
                        namespace["BenchRequest"]([1] * 128, 129)
                    ],
                    tqdm=lambda **kw: Progress(),
                    perf_counter=lambda: 0,
                    measure_request=lambda run, request, **kw: measure_request(
                        run, request, memory=kw["memory"], clock=lambda: next(ticks)
                    ),
                    request_cache_capacity=request_cache_capacity,
                    PEAK_MLX_SCOPE=PEAK_MLX_SCOPE,
                    REQUEST_ELAPSED_SCOPE=REQUEST_ELAPSED_SCOPE,
                    json=json,
                    asdict=asdict,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    namespace["main"]()
                if expected == "other":
                    self.assertNotIn("cache_capacity", calls[0])
                else:
                    self.assertEqual(calls[0]["cache_capacity"], expected)
                payload = json.loads(output.text)
                self.assertEqual(payload["metrics"]["request_peak_mlx_bytes"], [123456])
                self.assertEqual(payload["metrics"]["peak_mlx_bytes"], 123456)
                self.assertEqual(payload["metrics"]["elapsed_seconds"], 1.0)
                self.assertEqual(payload["metrics"]["peak_mlx_scope"], PEAK_MLX_SCOPE)

    def test_measurement_keeps_reset_and_query_outside_request_clock(self):
        events = []

        def clock():
            events.append("clock")
            return len(events)

        memory = SimpleNamespace(
            synchronize=lambda: events.append("sync"),
            reset_peak_memory=lambda: events.append("reset"),
            get_peak_memory=lambda: events.append("peak") or 1024,
        )
        result, elapsed, peak = measure_request(
            lambda request: events.append("request") or (129, 1, 2),
            None,
            memory=memory,
            clock=clock,
        )
        self.assertEqual(
            events, ["sync", "reset", "clock", "request", "clock", "sync", "peak"]
        )
        self.assertEqual((result, elapsed, peak), ((129, 1, 2), 2, 1024))

    def test_request_failure_is_not_a_zero_peak_sample(self):
        memory = SimpleNamespace(
            synchronize=lambda: None,
            reset_peak_memory=lambda: None,
            get_peak_memory=lambda: self.fail("query after failed request"),
        )
        with self.assertRaisesRegex(ValueError, "request failed"):
            measure_request(
                lambda request: (_ for _ in ()).throw(ValueError("request failed")),
                None,
                memory=memory,
                clock=lambda: 0,
            )

    def test_parser_retains_bytes_and_legacy_missing(self):
        rates = "Prefill throughput: 4.00 tok/s\nDecode throughput: 2.00 tok/s\nOutput throughput: 3.00 tok/s\n"
        measured = progression.parse_throughput(
            rates + "Peak MLX allocation bytes: 123456\n"
        )
        legacy = progression.parse_throughput(rates)
        self.assertEqual(
            json.loads(json.dumps(asdict(measured)))["peak_mlx_bytes"], 123456
        )
        self.assertIsNone(json.loads(json.dumps(asdict(legacy)))["peak_mlx_bytes"])
        self.assertIsNone(
            progression.median_throughput([measured, legacy]).peak_mlx_bytes
        )
        zero = progression.parse_throughput(rates + "Peak MLX allocation bytes: 0\n")
        self.assertEqual(zero.peak_mlx_bytes, 0)

    def test_progression_serializes_each_sample_and_median(self):
        output = TextOutput()
        args = SimpleNamespace(
            variant=["mlx"],
            model="fixture",
            solution="tiny_llm",
            suite="week2",
            device="gpu",
            input_len=128,
            output_len=129,
            seed=0,
            warmup=2,
            repeats=2,
            prefill_logits="last",
            offline=True,
            cooldown_seconds=0,
            json_output=output,
        )
        samples = [
            progression.Throughput(4, 2, 3, 100),
            progression.Throughput(4, 2, 3, 202),
        ]
        stdout = io.StringIO()
        with (
            patch.object(progression, "parse_args", return_value=args),
            patch.object(
                progression,
                "collect_host_metadata",
                return_value={
                    "platform": "fixture",
                    "machine": "fixture",
                    "mlx_version": "unrun",
                },
            ),
            patch.object(
                progression, "collect_source_metadata", return_value={"fixture": True}
            ),
            patch.object(progression, "run_variant", side_effect=samples),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            progression.main()
        payload = json.loads(output.text)
        result = payload["results"]["mlx"]
        self.assertEqual([s["peak_mlx_bytes"] for s in result["samples"]], [100, 202])
        self.assertEqual(result["median"]["peak_mlx_bytes"], 151)
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(
            payload["workload"]["request_elapsed_scope"], REQUEST_ELAPSED_SCOPE
        )
        self.assertEqual(payload["peak_mlx_scope"], PEAK_MLX_SCOPE)
        self.assertIn("Peak MLX MiB", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
