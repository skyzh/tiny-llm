"""Week 3 Day 1 continuous-batching tests."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

if Path(__file__).parent.name == "tests_refsol":
    import tiny_llm_ref.batch as batch_runtime
    import tiny_llm_ref.models as model_dispatch
else:
    import tiny_llm.batch as batch_runtime
    import tiny_llm.models as model_dispatch

from .tiny_llm_base import *
from .utils import *


def rope_helper(stream: mx.Stream, traditional: bool, precision: mx.Dtype):
    BATCH_SIZE = 16
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 14
    SEQ_LEN = 9
    BASE = 10000
    with mx.stream(stream):
        for _ in range(100):
            user_layer = FastRoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = mx.random.uniform(
                shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=precision
            )

            input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN, size=BATCH_SIZE)
            input_pos_mx = mx.array(input_pos, dtype=mx.int32)
            input_pos_user = input_pos.tolist()

            reference_output = mx.fast.rope(
                x.transpose(0, 2, 1, 3),
                dims=HEAD_DIM,
                traditional=traditional,
                base=BASE,
                scale=1.0,
                offset=input_pos_mx,
            ).transpose(0, 2, 1, 3)
            user_output = user_layer(x, input_pos_user)
            assert_allclose(
                user_output,
                reference_output,
                precision,
                atol=5e-6 if precision == mx.float32 else 1e-3,
            )


@pytest.mark.parametrize("traditional", [False, True], ids=["default", "traditional"])
def test_task_1_rope_multiple_offsets(traditional: bool):
    rope_helper(mx.gpu, traditional, mx.bfloat16)


def test_task_1_rectangular_causal_mask_keeps_the_prefix_visible():
    mask = causal_mask(L=2, S=5, dtype=mx.float32)

    expected = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0, -mx.inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    assert_allclose(mask, expected, mx.float32)


def test_task_2_batching_kv_cache():
    cache = BatchingKvCache(max_active_requests=3)

    slot0 = TinyKvFullCache()
    slot0.update_and_fetch(
        mx.array([[[[10.0]]]], dtype=mx.float32),
        mx.array([[[[110.0]]]], dtype=mx.float32),
    )

    slot2 = TinyKvFullCache()
    slot2.update_and_fetch(
        mx.array([[[[20.0], [21.0]]]], dtype=mx.float32),
        mx.array([[[[120.0], [121.0]]]], dtype=mx.float32),
    )

    cache.add_request(slot0, 0)
    cache.add_request(slot2, 2)

    keys = mx.array(
        [
            [[[12.0], [13.0]]],
            [[[0.0], [0.0]]],
            [[[22.0], [23.0]]],
        ],
        dtype=mx.float32,
    )
    values = mx.array(
        [
            [[[112.0], [113.0]]],
            [[[0.0], [0.0]]],
            [[[122.0], [123.0]]],
        ],
        dtype=mx.float32,
    )

    batched_keys, batched_values, seq_len, mask = cache.update_and_fetch(
        keys, values, mask_length=2
    )

    expected_keys = mx.array(
        [
            [[[0.0], [10.0], [12.0], [13.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[20.0], [21.0], [22.0], [23.0]]],
        ],
        dtype=mx.float32,
    )
    expected_values = mx.array(
        [
            [[[0.0], [110.0], [112.0], [113.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[120.0], [121.0], [122.0], [123.0]]],
        ],
        dtype=mx.float32,
    )
    expected_mask = mx.array(
        [
            [[[-mx.inf, 0.0, 0.0, -mx.inf], [-mx.inf, 0.0, 0.0, 0.0]]],
            [
                [
                    [-mx.inf, -mx.inf, -mx.inf, -mx.inf],
                    [-mx.inf, -mx.inf, -mx.inf, -mx.inf],
                ]
            ],
            [[[0.0, 0.0, 0.0, -mx.inf], [0.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=mx.float32,
    ).reshape(3, 1, 2, 4)

    assert seq_len is None
    assert_allclose(batched_keys, expected_keys, mx.float32)
    assert_allclose(batched_values, expected_values, mx.float32)
    assert_allclose(mask, expected_mask, mx.float32)

    cache.remove_request(0)
    replacement = TinyKvFullCache()
    replacement.update_and_fetch(
        mx.array([[[[30.0]]]], dtype=mx.float32),
        mx.array([[[[130.0]]]], dtype=mx.float32),
    )
    cache.add_request(replacement, 0)

    reused_keys, reused_values, seq_len, reused_mask = cache.update_and_fetch(
        mx.array([[[[31.0]]], [[[0.0]]], [[[24.0]]]], dtype=mx.float32),
        mx.array([[[[131.0]]], [[[0.0]]], [[[124.0]]]], dtype=mx.float32),
        mask_length=1,
    )
    expected_reused_keys = mx.array(
        [
            [[[0.0], [0.0], [0.0], [30.0], [31.0]]],
            [[[0.0], [0.0], [0.0], [0.0], [0.0]]],
            [[[20.0], [21.0], [22.0], [23.0], [24.0]]],
        ],
        dtype=mx.float32,
    )
    expected_reused_values = mx.array(
        [
            [[[0.0], [0.0], [0.0], [130.0], [131.0]]],
            [[[0.0], [0.0], [0.0], [0.0], [0.0]]],
            [[[120.0], [121.0], [122.0], [123.0], [124.0]]],
        ],
        dtype=mx.float32,
    )
    expected_reused_mask = mx.array(
        [
            [[[-mx.inf, -mx.inf, -mx.inf, 0.0, 0.0]]],
            [[[-mx.inf, -mx.inf, -mx.inf, -mx.inf, -mx.inf]]],
            [[[0.0, 0.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=mx.float32,
    )
    assert seq_len is None
    assert_allclose(reused_keys, expected_reused_keys, mx.float32)
    assert_allclose(reused_values, expected_reused_values, mx.float32)
    assert_allclose(reused_mask, expected_reused_mask, mx.float32)


def helper_test_task_3(
    model_name: str,
    seq_len: int,
    iters: int = 1,
):
    """Tests for continuous batching of decode requests."""
    requests = 4
    max_seq_len = seq_len

    mlx_model, tokenizer = load(model_name)
    model = model_dispatch.dispatch_week3_batch_model(model_name, mlx_model)
    for _ in range(iters):
        cache = [
            BatchingKvCache(requests, max_seq_len)
            for _ in range(model.num_hidden_layers)
        ]
        # Start each request at a staggered token index.
        staggered_start = [seq_len * i // requests for i in range(requests)]
        inputs = (
            mx.arange(requests * seq_len, dtype=mx.int32).reshape(requests, seq_len)
            % tokenizer.vocab_size
        )
        ref_outputs = mlx_model(inputs)
        for offset in range(seq_len + staggered_start[-1]):
            seq_idx = [offset - start for start in staggered_start]

            # Requests join at the staggered start, and leave when they reach seq_len.
            for request_id, sidx in enumerate(seq_idx):
                if sidx == 0:
                    for c in cache:
                        c.add_request(TinyKvFullCache(), request_id)
                elif sidx == seq_len:
                    for c in cache:
                        c.remove_request(request_id)

            next_tokens = []
            next_offsets = []
            for request_id, sidx in enumerate(seq_idx):
                if 0 <= sidx < seq_len:
                    next_tokens.append(inputs[request_id, sidx].item())
                    next_offsets.append(sidx)
                else:
                    next_tokens.append(0)
                    next_offsets.append(0)

            user_out = model(
                inputs=mx.array(next_tokens, dtype=mx.int32).reshape(-1, 1),
                offset=mx.array(next_offsets, dtype=mx.int32),
                cache=cache,
            )

            for request_id, sidx in enumerate(seq_idx):
                if 0 <= sidx < seq_len:
                    user_out_r = user_out[request_id, 0, :]
                    ref_out_r = ref_outputs[request_id, sidx, :]
                    user_out_r = user_out_r - mx.logsumexp(user_out_r, keepdims=True)
                    ref_out_r = ref_out_r - mx.logsumexp(ref_out_r, keepdims=True)
                    assert_allclose(
                        user_out_r,
                        ref_out_r,
                        precision=mx.bfloat16,
                        rtol=0.1,
                        atol=2.0,
                    )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_3_qwen3_0_6b():
    helper_test_task_3("Qwen/Qwen3-0.6B-MLX-4bit", seq_len=3)


@pytest.mark.skipif(not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found")
def test_task_3_qwen3_4b():
    helper_test_task_3(
        "Qwen/Qwen3-4B-MLX-4bit",
        seq_len=3,
    )


def _tiny_batch_forward(model):
    batch_size = 3
    active_slots = (0, 2)
    cache = [
        BatchingKvCache(batch_size, max_seq_len=8)
        for _ in range(model.num_hidden_layers)
    ]
    for slot in active_slots:
        request_cache = model.create_kv_cache()
        for layer_cache, batch_cache in zip(request_cache, cache):
            batch_cache.add_request(layer_cache, slot)

    inputs = mx.array([[5], [0], [7]], dtype=mx.int32)
    offsets = mx.array([0, 0, 3], dtype=mx.int32)
    return model(inputs=inputs, offset=offsets, cache=cache, logits_to_keep=1)


def test_task_3_week3_batch_factory_returns_numerically_valid_model():
    mx.random.seed(7)
    mlx_model = tiny_qwen3_mlx_model()
    factory_model = model_dispatch.dispatch_week3_batch_model("qwen3-0.6b", mlx_model)
    equivalent_model = Qwen3ModelWeek2(
        mlx_model,
        use_mlx_quantized_linear=True,
    )

    factory_logits = _tiny_batch_forward(factory_model)
    expected_logits = _tiny_batch_forward(equivalent_model)
    active = mx.array([0, 2], dtype=mx.int32)

    assert factory_logits.shape == (3, 1, 128)
    assert factory_logits.dtype == mx.bfloat16
    assert np.isfinite(np.array(factory_logits[active].astype(mx.float32))).all()
    assert_allclose(
        factory_logits[active], expected_logits[active], precision=mx.bfloat16
    )


def test_task_3_mlx_projection_selector_matches_public_equation_with_bias():
    mx.random.seed(11)
    dense_weight = mx.random.normal((4, 32), dtype=mx.float32)
    packed, scales, biases = mx.quantize(dense_weight, group_size=32, bits=4)
    weights = QuantizedWeights(
        scales=scales,
        biases=biases,
        group_size=32,
        bits=4,
        weight=packed,
        use_mlx_quantized_linear=True,
    )
    x = mx.random.normal((2, 32), dtype=mx.float32)
    output_bias = mx.array([0.5, -1.0, 1.5, -2.0], dtype=mx.float32)
    expected = (
        mx.quantized_matmul(
            x,
            packed,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=32,
            bits=4,
        )
        + output_bias
    )

    assert_allclose(mlx_quantized_linear(x, weights, output_bias), expected, mx.float32)
    assert_allclose(quantized_linear(x, weights, output_bias), expected, mx.float32)


class SchedulerFakeDetokenizer:
    def __init__(self, _):
        self.text = ""

    def add_token(self, token):
        self.text += str(token)


class SchedulerFakeTokenizer:
    eos_token_id = 99
    _tokenizer = object()
    detokenizer = SchedulerFakeDetokenizer(_tokenizer)

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        return {
            "long": [10],
            "short": [20],
            "later": [30],
            "immediate-eos": [40],
            "two-token-prompt": [50, 51],
        }[prompt]


class SchedulerFakeModel:
    num_hidden_layers = 1

    def create_kv_cache(self):
        return [TinyKvFullCache()]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        assert logits_to_keep == 1
        batch_cache = isinstance(cache[0], BatchingKvCache)
        sequence_length = 1 if batch_cache else inputs.shape[1]
        keys = mx.zeros((inputs.shape[0], 1, sequence_length, 1), dtype=mx.float32)
        if batch_cache:
            cache[0].update_and_fetch(keys, keys, mask_length=sequence_length)
        else:
            cache[0].update_and_fetch(keys, keys)

        next_by_token = {
            0: 99,
            1: 2,
            2: 3,
            3: 4,
            4: 99,
            5: 99,
            6: 99,
            7: 99,
            10: 1,
            20: 5,
            30: 6,
            40: 99,
            51: 7,
        }
        logits = mx.zeros((inputs.shape[0], 1, 100), dtype=mx.float32)
        for row, token in enumerate(inputs[:, -1].tolist()):
            logits[row, 0, next_by_token[token]] = 1
        return logits


def test_task_4_request_prefills_the_complete_prompt_in_one_call():
    request = batch_runtime.Request(
        SchedulerFakeModel(),
        SchedulerFakeTokenizer(),
        "two-token-prompt",
        prefill_max_step=2,
        prompt_idx=7,
    )

    request.try_prefill()

    assert request.is_prefill_done
    assert request.offset == 2
    assert request.next_token == 7
    assert request.prompt_idx == 7


def test_task_4_batch_generate_handles_immediate_eos(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    result = batch_runtime.batch_generate(
        SchedulerFakeModel(),
        SchedulerFakeTokenizer(),
        ["immediate-eos"],
        max_seq_len=2,
        batch_size=1,
        prefill_step=2,
    )

    assert result == [(0, "")]


@pytest.mark.parametrize(
    ("prompt", "max_seq_len", "expected"),
    [
        ("long", 1, [(0, "")]),
        ("long", 2, [(0, "1")]),
    ],
)
def test_task_4_batch_generate_stops_before_crossing_max_seq_len(
    monkeypatch, prompt, max_seq_len, expected
):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    result = batch_runtime.batch_generate(
        SchedulerFakeModel(),
        SchedulerFakeTokenizer(),
        [prompt],
        max_seq_len=max_seq_len,
        batch_size=1,
        prefill_step=max_seq_len,
    )

    assert result == expected


def test_task_4_batch_generate_rejects_an_oversized_prompt(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    with pytest.raises(ValueError):
        batch_runtime.batch_generate(
            SchedulerFakeModel(),
            SchedulerFakeTokenizer(),
            ["two-token-prompt"],
            max_seq_len=1,
            batch_size=1,
            prefill_step=1,
        )


def test_task_4_batch_generate_reuses_capacity_and_returns_completion_order(
    monkeypatch,
):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    model = SchedulerFakeModel()
    result = batch_runtime.batch_generate(
        model,
        SchedulerFakeTokenizer(),
        ["long", "short", "later"],
        batch_size=2,
    )

    assert result == [(1, "5"), (2, "6"), (0, "1234")]


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen3_1_7b():
    helper_test_task_3(
        "Qwen/Qwen3-1.7B-MLX-4bit",
        seq_len=3,
    )
