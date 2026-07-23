"""Week 3 Day 1 continuous-batching tests."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

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


def test_task_2_batching_kv_cache():
    cache = BatchingKvCache(max_active_requests=3, max_seq_len=8)

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


def helper_test_task_3(
    model_name: str,
    seq_len: int,
    iters: int = 1,
):
    """Tests for continuous batching of decode requests."""
    requests = 4
    max_seq_len = seq_len

    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model)
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


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen3_1_7b():
    helper_test_task_3(
        "Qwen/Qwen3-1.7B-MLX-4bit",
        seq_len=3,
    )
