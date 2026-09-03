import pytest
from .utils import *
from .tiny_llm_base import Qwen3ModelWeek1, Embedding, dequantize_linear, qwen3_week1
from mlx_lm import load


REQUIRED_MODEL = "Qwen/Qwen3-0.6B-MLX-4bit"
REQUIRED_MODEL_DOWNLOAD = f"hf download {REQUIRED_MODEL}"


def require_default_model():
    if not qwen3_0_6b_model_exists():
        pytest.fail(
            f"The default Day 5 model is required. Run `{REQUIRED_MODEL_DOWNLOAD}` "
            "and rerun this checkpoint."
        )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_1_transformer_block(
    stream: mx.Stream, precision: mx.Dtype, mask: str | None
):
    with mx.stream(stream):
        from mlx_lm.models import qwen3

        BATCH_SIZE = 1
        SEQ_LEN = 10
        NUM_ATTENTION_HEAD = 4
        NUM_KV_HEADS = 2
        HIDDEN_SIZE = 32
        HEAD_DIM = 12
        INTERMEDIATE_SIZE = HIDDEN_SIZE * 4

        args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=1,
            intermediate_size=INTERMEDIATE_SIZE,
            num_attention_heads=NUM_ATTENTION_HEAD,
            num_key_value_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rms_norm_eps=1e-6,
            vocab_size=1000,
            max_position_embeddings=128,
            rope_theta=10000,
            tie_word_embeddings=True,
        )

        mlx_transformer_block = qwen3.TransformerBlock(args)

        mlx_attention = mlx_transformer_block.self_attn
        wq = mlx_attention.q_proj.weight
        wk = mlx_attention.k_proj.weight
        wv = mlx_attention.v_proj.weight
        wo = mlx_attention.o_proj.weight
        q_norm = mlx_attention.q_norm.weight
        k_norm = mlx_attention.k_norm.weight

        mlx_mlp = mlx_transformer_block.mlp
        w_gate = mlx_mlp.gate_proj.weight
        w_up = mlx_mlp.up_proj.weight
        w_down = mlx_mlp.down_proj.weight

        w_input_layernorm = mlx_transformer_block.input_layernorm.weight
        w_post_attention_layernorm = (
            mlx_transformer_block.post_attention_layernorm.weight
        )

        user_transformer_block = qwen3_week1.Qwen3TransformerBlock(
            num_attention_heads=NUM_ATTENTION_HEAD,
            num_kv_heads=NUM_KV_HEADS,
            hidden_size=HIDDEN_SIZE,
            head_dim=HEAD_DIM,
            intermediate_size=INTERMEDIATE_SIZE,
            rms_norm_eps=1e-6,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            w_input_layernorm=w_input_layernorm,
            w_post_attention_layernorm=w_post_attention_layernorm,
        )

        mx.random.seed(42)
        x = mx.random.uniform(shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=precision)

        user_output = user_transformer_block(x, mask=mask)
        mlx_output = mlx_transformer_block(x, mask=mask, cache=None)

        assert_allclose(
            user_output, mlx_output, precision=precision, rtol=1e-1, atol=1e-1
        )
        assert user_output.dtype == mlx_output.dtype


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "leading_shape",
    [(), (5,), (2, 3), (2, 1, 3)],
    ids=["zero_leading", "one_leading", "two_leading", "three_leading"],
)
def test_task_2_embedding_lookup_without_download(
    stream: mx.Stream,
    leading_shape: tuple[int, ...],
):
    vocab_size = 7
    embedding_dim = 5
    weight = mx.array(
        np.linspace(-0.75, 0.875, vocab_size * embedding_dim).reshape(
            vocab_size, embedding_dim
        )
    ).astype(mx.bfloat16)
    token_ids = (
        mx.arange(int(np.prod(leading_shape)), dtype=mx.int32).reshape(leading_shape)
        * 3
        + 1
    ) % vocab_size

    with mx.stream(stream):
        output = Embedding(vocab_size, embedding_dim, weight)(token_ids)
        expected = weight[token_ids, :]

    assert output.shape == (*leading_shape, embedding_dim)
    assert output.dtype == mx.bfloat16
    assert_allclose(output, expected, precision=mx.bfloat16)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "leading_shape",
    [(), (5,), (2, 3), (2, 1, 3)],
    ids=["zero_leading", "one_leading", "two_leading", "three_leading"],
)
def test_task_2_embedding_as_linear_without_download(
    stream: mx.Stream,
    leading_shape: tuple[int, ...],
):
    vocab_size = 7
    embedding_dim = 5
    weight = mx.array(
        np.linspace(-0.625, 0.75, vocab_size * embedding_dim).reshape(
            vocab_size, embedding_dim
        )
    ).astype(mx.bfloat16)
    hidden = mx.array(
        np.linspace(
            -0.5,
            0.625,
            int(np.prod(leading_shape)) * embedding_dim,
        ).reshape(*leading_shape, embedding_dim)
    ).astype(mx.bfloat16)

    with mx.stream(stream):
        output = Embedding(vocab_size, embedding_dim, weight).as_linear(hidden)
        expected = mx.matmul(hidden, weight.T)

    assert output.shape == (*leading_shape, vocab_size)
    assert output.dtype == mx.bfloat16
    assert_allclose(output, expected, precision=mx.bfloat16)


def helper_test_task_3(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek1(mlx_model)
    for iteration in range(iters):
        input = (mx.arange(10, dtype=mx.int32) + iteration * 10).reshape(
            1, 10
        ) % tokenizer.vocab_size
        user_output = model(input)
        ref_output = mlx_model(input)
        user_output = user_output - mx.logsumexp(user_output, axis=-1, keepdims=True)
        ref_output = ref_output - mx.logsumexp(ref_output, axis=-1, keepdims=True)
        assert_allclose(
            user_output, ref_output, precision=mx.bfloat16, rtol=0.1, atol=2.5
        )


def test_task_2_embedding_call():
    require_default_model()
    mlx_model, _ = load(REQUIRED_MODEL)
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.bfloat16),
    )
    for _ in range(50):
        input = mx.random.randint(low=0, high=mlx_model.args.vocab_size, shape=(1, 10))
        user_output = embedding(input)
        ref_output = mlx_model.model.embed_tokens(input)
        assert_allclose(user_output, ref_output, precision=mx.bfloat16)


def test_task_2_embedding_as_linear():
    require_default_model()
    mlx_model, _ = load(REQUIRED_MODEL)
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.bfloat16),
    )
    for _ in range(50):
        input = mx.random.uniform(shape=(1, 10, mlx_model.args.hidden_size))
        user_output = embedding.as_linear(input)
        ref_output = mlx_model.model.embed_tokens.as_linear(input)
        assert_allclose(user_output, ref_output, precision=mx.bfloat16, atol=1e-1)


def test_task_3_qwen3_0_6b():
    require_default_model()
    helper_test_task_3(REQUIRED_MODEL, 5)


@pytest.mark.skipif(not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found")
def test_task_3_qwen3_4b():
    helper_test_task_3("Qwen/Qwen3-4B-MLX-4bit", 1)


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen3_1_7b():
    helper_test_task_3("Qwen/Qwen3-1.7B-MLX-4bit", 3)
