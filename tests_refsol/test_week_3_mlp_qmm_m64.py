"""Reference-only M64 compatibility: rounding, lifetime and fallback behavior."""

import json
import gc
import hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
import pytest
from extensions_ref.tiny_llm_ext_ref import _ext

META = json.loads(Path(__file__).with_name("mlp_qmm_m64_cases.json").read_text())
CASES = META["cases"]
OBSERVER = None


def emit(row):
    if OBSERVER is not None:
        OBSERVER(row)


def fingerprints(roots):
    return [
        hashlib.sha256(np.asarray(v.view(mx.uint8)).tobytes()).hexdigest()
        for v in roots
    ]


def prepare(row):
    K, N = (2560, 9728) if row["domain"] == 0 else (9728, 2560)
    B, L = 1, 256
    dt = mx.bfloat16
    variant = row.get("variant", "")
    offset = row["offset"]
    if variant.startswith("m") and variant[1:].isdigit():
        L = int(variant[1:])
    if variant == "batch":
        B, L = 2, 128
    if variant == "f16":
        dt = mx.float16
    if variant == "f32":
        dt = mx.float32
    if variant == "smallK":
        K = 128
    if variant == "smallN":
        N = 128
    rng = np.random.default_rng(row["seed"])
    a = (rng.standard_normal((B, L, K)) * 0.1).astype(np.float32)
    w = rng.integers(0, 2**32 - 1, size=(N, K // 8), dtype=np.uint32)
    s = np.full((N, K // 128), 0.01, np.float32)
    b = np.full(s.shape, -0.05, np.float32)
    if row["kind"] == "eligible":
        if row["family"] == 1:
            a[:, :, 1::2] *= -1
        elif row["family"] == 2:
            a.fill(0)
            a[:, :, ::2] = -0.0
        elif row["family"] == 3:
            w.fill(0x76543210)
            b.fill(0.125)
    if row["kind"] == "rounding":
        a.fill(0)
        a[:, :, 0] = 1
        a[:, :, 1] = 2**-8
        w.fill(0)
        w[:, 0] = 0x11
        s.fill(1)
        b.fill(0)
        family = row["family"]
        base = row["placement"] * 64
        if family == 1:
            a[:, :, 0] = 1 + 2**-7
        elif family == 2:
            a *= -1
        elif family == 3:
            a[:, :, 0] = 2**-126
            a[:, :, 1] = 2**-133
        elif family == 4:
            a[0, base, 0] = np.inf
            a[0, base + 1, 0] = -np.inf
        elif family == 5:
            a[0, base, 0] = np.nan
            a[0, base + 1, 0] = np.inf
            a[0, base + 2, 0] = -np.inf
            a[0, base + 3, 0] = -0.0
    if variant == "alias":
        w.fill(0x3F003F00)
    roots = []
    values = []
    for data, dtype in [(a, dt), (w, mx.uint32), (s, dt), (b, dt)]:
        flat = mx.array(data).astype(dtype).reshape(-1)
        root = (
            mx.concatenate([mx.zeros((offset,), dtype=dtype), flat]) if offset else flat
        )
        roots.append(root)
        values.append(root[offset : offset + flat.size].reshape(data.shape))
    mx.eval(*roots, *values)
    if variant == "strided":
        values[0] = mx.contiguous(values[0].transpose(0, 2, 1)).transpose(0, 2, 1)
        mx.eval(values[0])
        roots.append(values[0])
    if variant == "alias":
        values[0] = (
            values[1].view(mx.bfloat16).reshape(-1)[: B * L * K].reshape(B, L, K)
        )
        mx.eval(values[0])
    if variant == "lazy_weights":
        values[1] = mx.contiguous(values[1])
    if variant == "unknown_lazy":
        values[0] = -values[0]
    if row["kind"] == "owner" and row["mode"] == 2:
        # A known deferred native producer, with no eager materialization of its output.
        if row["domain"] == 0:
            norm_weight = mx.ones((K,), dtype=dt)
            mx.eval(norm_weight)
            roots.append(norm_weight)
            values[0] = _ext.rms_norm(values[0], norm_weight, 1e-6)
        else:
            up = mx.ones((B, L, K), dtype=dt)
            mx.eval(up)
            roots.append(up)
            values[0] = _ext.swiglu(values[0], up)
    logical = (
        sum(v.nbytes for v in roots) + sum(v.nbytes for v in values) + 4 * B * L * N * 2
    )
    assert logical < 256 * 1024**2, "QUALITY_LOGICAL_RAM_BOUND"
    return values, roots, logical


def original(v, group=128, bits=4):
    a, w, s, b = v
    return mx.quantized_matmul(
        a, w, scales=s, biases=b, transpose=True, group_size=group, bits=bits
    )


def composed(v, group=128, bits=4):
    a, w, s, b = v
    out = _ext._try_mlp_qmm_m64(a, w, s, b, group, bits)
    return (original(v, group, bits) if out is None else out), out is not None


def run(row):
    v, roots, logical = prepare(row)
    group = 128
    bits = 4
    if row["kind"] == "reject":
        q = row["variant"]
        if q == "packed_shape":
            v[1] = v[1][:, :-1]
        elif q == "scale_shape":
            v[2] = v[2][:, :-1]
        elif q == "bias_shape":
            v[3] = v[3][:, :-1]
        elif q == "group":
            group = 7
        elif q == "bits":
            bits = 3
        elif q == "weight_dtype":
            v[1] = v[1].astype(mx.int32)
        elif q == "scale_dtype":
            v[2] = v[2].astype(mx.int32)
            v[3] = v[3].astype(mx.int32)
        elif q == "inner_shape":
            v[0] = v[0][:, :, :-1]
        error = None
        try:
            original(v, group, bits)
        except Exception as e:
            error = type(e)
        assert error is not None, "INVALID_FIXTURE_REJECTS"
        with pytest.raises(error):
            composed(v, group, bits)
        result = dict(
            stage="compared",
            id=row["id"],
            domain=row["domain"],
            status="PASS",
            rejection=error.__name__,
            no_forward=True,
        )
        emit(result)
        return result
    before = fingerprints(roots)
    route_before = list(_ext._mlp_qmm_m64_route_counts())
    ref = original(v)
    out, selected = composed(v)
    emit(
        dict(
            stage="prepared",
            id=row["id"],
            domain=row["domain"],
            selected=selected,
            original_before_hashes=before,
            logical_fixture_bytes=logical,
            route_before=route_before,
        )
    )
    release = row["kind"] == "owner" and row["mode"] == 3
    if row["kind"] == "owner":
        if release:
            mx.eval(ref)
            mx.synchronize()
        if row["mode"] in (1, 3):
            v = [mx.zeros((1,), dtype=mx.bfloat16) for _ in v]
        if release:
            roots = None
        gc.collect()
    if row["kind"] == "owner" and row["mode"] == 2:
        mx.eval(out)
        mx.synchronize()
        mx.eval(ref)
    else:
        mx.eval(ref, out)
    mx.synchronize()
    delta = [
        b - a for a, b in zip(route_before, list(_ext._mlp_qmm_m64_route_counts()))
    ]
    exact = bool(mx.array_equal(ref.view(mx.uint8), out.view(mx.uint8)).item())
    after = None if release else fingerprints(roots)
    pipeline = list(_ext._mlp_qmm_m64_pipeline_state())
    sentinel = None
    if row["kind"] == "rounding" and row["family"] in (0, 5):
        sentinel = bool(
            mx.all(out[0, (row["placement"] * 64 + 8) % 256, :] == 1).item()
        )
    result = dict(
        stage="compared",
        id=row["id"],
        kind=row["kind"],
        domain=row["domain"],
        bit_exact=exact,
        shape=list(out.shape),
        selected=selected,
        route_delta=delta,
        pipeline_state=pipeline,
        original_before_hashes=before,
        original_after_hashes=after,
        original_after_observed=not release,
        caller_release_output_witness=release,
        finite=bool(mx.all(mx.isfinite(out)).item()),
        rounding_sentinel_one=sentinel,
        logical_fixture_bytes=logical,
        already_computed_only=True,
    )
    emit(result)
    assert exact, "EXACT_M64_REFERENCE_BITS"
    assert after is None or before == after, "ORIGINAL_INPUT_BYTES_CHANGED"
    assert selected == (row["kind"] != "fallback"), "ADMITTED_OR_FALLBACK_ROUTE"
    assert delta[2] == int(selected) and delta[4 + row["domain"]] == int(selected), (
        "PER_DOMAIN_EVALUATED_ROUTE"
    )
    if selected:
        assert (
            pipeline[0] >= 128
            and pipeline[1] <= pipeline[2]
            and 7808 <= pipeline[2]
            and pipeline[4] == 0
        ), "COMPILED_PIPELINE_LIMITS"
    if sentinel is not None:
        assert sentinel, "BF16_PROJECTION_ROUNDING"
    return result


@pytest.mark.parametrize("row", CASES, ids=[r["id"] for r in CASES])
def test_mlp_qmm_m64(row):
    run(row)


def group(name):
    return [
        run(next(r for r in CASES if r["id"] == rid))
        for rid in META["smoke_groups"][name]
    ]


def smoke_arithmetic():
    return group("smoke_arithmetic")


def smoke_boundary():
    return group("smoke_boundary")


def smoke_owners():
    return group("smoke_owners")
