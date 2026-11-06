"""CPU recording witnesses: stdlib fixtures, no MLX or course implementation."""
from dataclasses import replace
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from benches import profile_week2_kernels as profile
from benches import week2_profile_adapter as adapter


def features(**updates):
    result = dict.fromkeys(adapter.PUBLIC_FEATURES, False)
    result.update(updates)
    return result


class Fixture:
    def __init__(self, shape, role, dtype="bfloat16"):
        self.shape, self.role, self.dtype = shape, role, dtype
        self.nbytes = 2
        for extent in shape:
            self.nbytes *= extent


class Recording:
    def __init__(self, fail_update=False, fail_seed=False, fail_release=False):
        self.events, self.caches = [], []
        self.fail_update, self.fail_seed = fail_update, fail_seed
        self.fail_release = fail_release

    def array(self, shape, role, dtype):
        return Fixture(shape, role, dtype)

    def project(self, layer, role, value):
        self.events.append(("project", layer, role, value.shape))
        return value

    def pointwise(self, layer, role, *values, offset=0):
        self.events.append(("pointwise", layer, role, offset, values[0].shape))
        return values[0]

    def attention(self, query, key, value, *, scale, mask):
        self.events.append(("attention", query.shape, key.shape, scale, mask))
        return query

    def evaluate(self, outputs):
        self.events.append(("evaluate", len(outputs)))

    def cache_factory(self, capacity):
        owner = self

        class Cache:
            def __init__(self):
                self.capacity, self.offset = capacity, 0
                self.released = False
                for name in adapter.COUNTERS:
                    setattr(self, name, 0)

            def update_and_fetch(self, key, value, mask_length=None, mask=None):
                length = key.shape[2]
                owner.events.append(("update", self.offset, length, self.capacity, mask))
                if (owner.fail_update and length == 1) or (owner.fail_seed and length > 1):
                    raise RuntimeError("injected update failure")
                if self.capacity is not None and self.offset + length > self.capacity:
                    raise ValueError("overflow")
                if self.capacity is None and self.offset:
                    for name in ("logical_copy_bytes", "physical_growth_copy_bytes", "growth_copy_bytes"):
                        setattr(self, name, getattr(self, name) + 2 * self.offset)
                if self.capacity is not None:
                    self.slice_write_bytes += key.nbytes + value.nbytes
                self.offset += length
                shape = (key.shape[0], key.shape[1], self.offset, key.shape[3])
                return Fixture(shape, "cached-key"), Fixture(shape, "cached-value"), self.offset, mask

            def release(self):
                self.released = True
                owner.events.append(("release", self.offset))
                if owner.fail_release:
                    raise RuntimeError("injected release failure")

        cache = Cache()
        self.caches.append(cache)
        self.events.append(("new-cache", capacity))
        return cache

    def bindings(self):
        return adapter.PublicBindings(self.array, self.project, self.pointwise,
                                      self.attention, self.cache_factory, self.evaluate)


def plan(phase="decode", tokens=8, **kwargs):
    return adapter.ReplayPlan(adapter.ProfileCase("capacity-cache", phase, tokens),
                              adapter.ModelShape(2, 6, 10, 2, 1, 4, 23),
                              features(bounded_kv_capacity=True), **kwargs)


def decision_record():
    return {"schema_version": 3, "coverage": {"complete": True}, "source": {"tree": "tree"},
            "solution": "tiny_llm", "model": "model", "workload_id": "workload",
            "software": {"mlx": "version"}, "host": {"machine": "machine"},
            "profile_kind": adapter.PROFILE_KIND, "checkpoint": "capacity-cache"}


def decision(left, right):
    return profile.build_decision(left, right, dominant_category="projections",
                                  hypothesis="test", observed_effect="test", decision="inconclusive",
                                  next_experiment="test")


class PublicContractTests(unittest.TestCase):
    def test_public_replay_needs_no_model_graph_and_preserves_role_shapes(self):
        recording = Recording()
        replay = adapter.PublicReplay(plan(), recording.bindings())
        replay.projections()
        calls = [x for x in recording.events if x[0] == "project"]
        self.assertEqual(len(calls), 15)
        self.assertEqual(calls[3], ("project", 0, "o", (1, 1, 8)))
        self.assertEqual(calls[6], ("project", 0, "down", (1, 1, 10)))
        self.assertEqual(calls[-1], ("project", None, "head", (1, 1, 6)))
        self.assertFalse(hasattr(replay, "model"))

    def test_all_and_last_logits_change_both_head_and_final_norm_rows(self):
        for mode, expected in (("last", 1), ("all", 8)):
            recording = Recording()
            replay = adapter.PublicReplay(plan("prefill", prefill_logits=mode), recording.bindings())
            replay.projections()
            replay.pointwise()
            self.assertIn(("project", None, "head", (1, expected, 6)), recording.events)
            self.assertIn(("pointwise", None, "final_norm", 0, (1, expected, 6)), recording.events)

    def test_decode_positions_attention_scale_and_mask(self):
        recording = Recording()
        replay = adapter.PublicReplay(plan(), recording.bindings())
        replay.pointwise()
        replay.attention()
        rope = [x for x in recording.events if x[:3] == ("pointwise", 0, "rope")]
        self.assertEqual([x[3] for x in rope], [7, 7])
        attention = [x for x in recording.events if x[0] == "attention"]
        self.assertEqual(len(attention), 2)
        self.assertEqual(attention[0][1:], ((1, 2, 1, 4), (1, 1, 8, 4), 0.5, None))

    def test_public_operator_choices_and_simd_options_use_injected_callables(self):
        calls = []
        readable = lambda *args: calls.append("student-readable")
        optimized = lambda *args: calls.append("student-optimized")
        for flag in ("quantized_weights", "fast_rms_norm", "fast_rope", "fast_swiglu"):
            for enabled, expected in ((False, "student-readable"), (True, "student-optimized")):
                selected = adapter.choose_operator(features(**{flag: enabled}), flag, readable, optimized)
                selected()
                self.assertEqual(calls[-1], expected)
        self.assertEqual(adapter.packed_weight_options(features(simdgroup_matmul=True)),
                         {"use_simdgroup_matmul": True})
        self.assertEqual(adapter.packed_weight_options(features()), {"use_simdgroup_matmul": False})

    def test_runtime_registry_namespaces_are_injected_without_importing_a_backend(self):
        for solution in ("tiny_llm", "tiny_llm_ref"):
            imports = []
            def injected(name):
                imports.append(name)
                if name.endswith("qwen3_week2"):
                    return SimpleNamespace(WEEK2_CHECKPOINTS=("kv-cache",),
                                           WEEK2_CHECKPOINT_FEATURES={"kv-cache": features()})
                return SimpleNamespace(dense_prefill_attention_mma=object())
            with patch.object(adapter.importlib, "import_module", side_effect=injected):
                implementation = adapter.load_implementation(solution)
            self.assertEqual(implementation.name, solution)
            self.assertEqual(len(imports), 9)
            self.assertTrue(all(name.startswith(solution + ".") for name in imports))
            self.assertEqual(implementation.checkpoints, ("kv-cache",))

    def test_plan_copies_features_instead_of_observing_later_mutation(self):
        flags = features(bounded_kv_capacity=True)
        selected = replace(plan(), features=flags)
        flags["bounded_kv_capacity"] = False
        self.assertEqual(selected.capacity, 8)
        with self.assertRaises(TypeError):
            selected.features["bounded_kv_capacity"] = False

    def test_attention_boundaries_and_readable_fallback(self):
        common = dict(context=128, head_dim=128, dtype="bfloat16", mask="causal", tiled=True)
        self.assertEqual(adapter.attention_path(rows=8, **common), "readable")
        self.assertEqual(adapter.attention_path(rows=9, **common), "tiled")
        self.assertEqual(adapter.attention_path(rows=9, **{**common, "head_dim": 64}), "readable")
        self.assertEqual(adapter.attention_path(rows=9, **{**common, "dtype": "float32"}), "readable")
        self.assertEqual(adapter.attention_path(rows=1, context=256, head_dim=4, dtype="float32",
                                              mask=None, decode=True, max_query=2, max_context=256), "custom")
        self.assertEqual(adapter.attention_path(rows=1, context=257, head_dim=4, dtype="float32",
                                              mask=None, decode=True, max_query=2, max_context=256), "readable")

    def test_unsupported_reference_selected_and_unknown_features_refuse_plan(self):
        with self.assertRaises(adapter.UnsupportedCoverage):
            replace(plan(), solution="tiny_llm_ref", case=adapter.ProfileCase("selected", "prefill", 8))
        with self.assertRaises(adapter.UnsupportedCoverage):
            replace(plan(), features=features(bounded_kv_capacity=True, unknown=True))
        self.assertTrue(replace(plan(), case=adapter.ProfileCase("selected", "prefill", 8)))

    def test_capacity_and_invalid_shape_refuse_before_fixture_preparation(self):
        with self.assertRaises(ValueError):
            plan(cache_capacity=7)
        with self.assertRaises(ValueError):
            adapter.ModelShape(2, 6, 10, 3, 2, 4, 23)
        with self.assertRaises(ValueError):
            adapter.ProfileCase("capacity-cache", "decode", 0)


class LifecycleTests(unittest.TestCase):
    def run_cache(self, replay, recording, iterations=3):
        group = replay.groups()[-1]
        ticks = iter(range(iterations * 2))

        def clock():
            recording.events.append(("clock",))
            return next(ticks)

        profile.benchmark_groups((group,), 0, iterations, evaluator=recording.evaluate, clock=clock)

    def test_decode_uses_real_cache_method_same_prefix_each_sample(self):
        recording = Recording()
        replay = adapter.PublicReplay(plan(cache_capacity=12), recording.bindings())
        recording.events.clear()
        self.run_cache(replay, recording)
        updates = [x for x in recording.events if x[0] == "update"]
        self.assertEqual([(x[1], x[2]) for x in updates], [(0, 7), (0, 7), (7, 1), (7, 1)] * 3)
        self.assertTrue(all(x[3] == 12 for x in updates))
        first_clock = recording.events.index(("clock",))
        self.assertEqual([x[2] for x in recording.events[:first_clock] if x[0] == "update"], [7, 7])
        self.assertEqual(recording.events[first_clock + 3][0], "evaluate")
        self.assertTrue(all(cache.released for cache in recording.caches))
        self.assertEqual(replay.cache.caches, [])
        self.assertEqual(replay.cache.observation["before_offset"], 7)
        self.assertEqual(replay.cache.observation["after_offset"], 8)
        self.assertEqual(replay.cache.observation["counter_deltas"]["logical_copy_bytes"], 0)

    def test_dense_and_bounded_select_different_capacity_and_counter_paths(self):
        for bounded in (False, True):
            recording = Recording()
            replay = adapter.PublicReplay(replace(plan(), features=features(bounded_kv_capacity=bounded)), recording.bindings())
            self.run_cache(replay, recording, 1)
            self.assertTrue(all(cache.capacity == (8 if bounded else None) for cache in recording.caches))
            delta = replay.cache.observation["counter_deltas"]
            self.assertEqual(delta["logical_copy_bytes"] == 0, bounded)
            self.assertEqual(delta["slice_write_bytes"] > 0, bounded)

    def test_prefill_has_no_seed_update_and_includes_first_write(self):
        recording = Recording()
        replay = adapter.PublicReplay(plan("prefill"), recording.bindings())
        recording.events.clear()
        self.run_cache(replay, recording, 1)
        updates = [x for x in recording.events if x[0] == "update"]
        self.assertEqual([(x[1], x[2], x[4]) for x in updates], [(0, 8, "causal")] * 2)
        self.assertLess(recording.events.index(("clock",)), recording.events.index(updates[0]))
        self.assertTrue(replay.cache.observation["first_allocation_in_update"])

    def test_failures_during_prepare_or_update_dispose_every_created_cache(self):
        for flags in ({"fail_update": True}, {"fail_seed": True}):
            recording = Recording(**flags)
            replay = adapter.PublicReplay(plan(), recording.bindings())
            with self.assertRaisesRegex(RuntimeError, "injected update"):
                self.run_cache(replay, recording, 1)
            self.assertTrue(all(cache.released for cache in recording.caches))
            self.assertEqual(replay.cache.caches, [])

    def test_wrong_prefix_output_is_refused_and_disposed(self):
        recording = Recording()
        original = recording.cache_factory
        def factory(capacity):
            cache = original(capacity)
            call = cache.update_and_fetch
            def wrong(*args, **kwargs):
                key, value, offset, mask = call(*args, **kwargs)
                return Fixture((1, 1, offset + 1, 4), "unused-tail"), value, offset, mask
            cache.update_and_fetch = wrong
            return cache
        bindings = replace(recording.bindings(), cache_factory=factory)
        replay = adapter.PublicReplay(plan(), bindings)
        with self.assertRaisesRegex(RuntimeError, "logical prefix"):
            self.run_cache(replay, recording, 1)
        self.assertTrue(all(cache.released for cache in recording.caches))

    def test_cleanup_attempts_all_releases_when_a_release_fails(self):
        recording = Recording(fail_release=True)
        replay = adapter.PublicReplay(plan(), recording.bindings())
        with self.assertRaises(ExceptionGroup):
            self.run_cache(replay, recording, 1)
        self.assertTrue(all(cache.released for cache in recording.caches))
        self.assertEqual(replay.cache.caches, [])

    def test_synchronization_is_inside_clock_and_cleanup_outside(self):
        events = []
        group = adapter.ReplayGroup("test", lambda: events.append("build") or ["output"],
                                    lambda: events.append("prepare"), lambda: events.append("finish"),
                                    lambda: events.append("cleanup"))
        values = iter((1, 3))
        def clock():
            events.append("clock")
            return next(values)
        result = profile.benchmark_groups((group,), 0, 1,
                    evaluator=lambda _: events.append("evaluate"), clock=clock)
        self.assertEqual(events, ["prepare", "clock", "build", "evaluate", "clock", "finish", "cleanup"])
        self.assertEqual(result, {"test": 2_000_000})


class IdentityAndRefusalTests(unittest.TestCase):
    def test_workload_fields_and_mechanism_treatment(self):
        first = profile.ProfileCase("kv-cache", "decode", 8)
        second = profile.ProfileCase("capacity-cache", "decode", 8)
        a = profile.workload_record("model", first, 4, 12, cache_capacity=12)
        self.assertEqual(a, profile.workload_record("model", second, 4, 12, cache_capacity=12))
        self.assertEqual(a["request_bound"], 12)
        self.assertEqual(a["position_offset"], 7)
        for field in a:
            changed = {**a, field: "different"}
            self.assertNotEqual(profile.canonical_hash(a), profile.canonical_hash(changed), field)
        self.assertNotEqual(profile.canonical_hash(a), profile.canonical_hash(
            profile.workload_record("model", second, 4, 12, cache_capacity=13)))
        self.assertNotEqual(profile.canonical_hash(a), profile.canonical_hash(
            profile.workload_record("model", second, 4, 12, cache_capacity=12, prefill_logits="all")))

    def test_decisions_refuse_old_incomplete_and_mismatched_records(self):
        baseline = decision_record()
        self.assertEqual(decision(baseline, {**baseline, "checkpoint": "quantized-matvec"})["schema_version"], 3)
        for changed in ({"schema_version": 2}, {"coverage": {"complete": False}},
                        {"workload_id": "other"}, {"host": {}}, {"software": {}}):
            with self.assertRaises(ValueError):
                decision(baseline, {**baseline, **changed})

    def test_incomplete_case_has_no_numeric_total_or_shares(self):
        with patch.object(adapter, "make_runtime_replay", side_effect=adapter.UnsupportedCoverage("missing coverage")):
            record = profile.profile_case(SimpleNamespace(), object(), "model",
                                          profile.ProfileCase("selected", "prefill", 8), 4, 12)
        self.assertFalse(record["coverage"]["complete"])
        self.assertEqual(record["categories"], [])
        self.assertNotIn("replay_total_us", record)
        self.assertNotIn("share_denominator", record)

    def test_existing_output_refuses_before_runtime_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("preserve")
            with patch.object(profile, "parse_args", return_value=SimpleNamespace(json_output=path, decision_output=None)), \
                 patch.object(profile, "load_implementation", side_effect=AssertionError("runtime entered")):
                with self.assertRaises(FileExistsError):
                    profile.main()
            self.assertEqual(path.read_text(), "preserve")

    def test_alias_output_refuses_before_runtime_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            alias = path.parent / "unused" / ".." / path.name
            with patch.object(profile, "parse_args", return_value=SimpleNamespace(json_output=path, decision_output=alias)), \
                 patch.object(profile, "load_implementation", side_effect=AssertionError("runtime entered")):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    profile.main()

    def test_balanced_group_rotation_remains_exact(self):
        calls = []
        builders = tuple((name, lambda name=name: calls.append(name) or []) for name in "abcd")
        ticks = iter(range(32))
        profile.benchmark_groups(builders, 0, 4, evaluator=lambda _: None, clock=lambda: next(ticks))
        self.assertEqual("".join(calls), "abcdbcdacdabdabc")
