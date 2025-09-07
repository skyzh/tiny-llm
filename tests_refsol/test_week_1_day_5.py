import pytest
from .utils import *
from .tiny_llm_base import Qwen2ModelWeek1, Embedding, dequantize_linear, qwen2_week1
from mlx_lm import load

@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_1_transformer_block(
    stream: mx.Stream, precision: mx.Dtype, mask: str | None
):
    with mx.stream(stream):
        from mlx_lm.models import qwen2

        BATCH_SIZE = 1
        SEQ_LEN = 10
        NUM_ATTENTION_HEAD = 4
        NUM_KV_HEADS = 2
        HIDDEN_SIZE = 32
        INTERMEDIATE_SIZE = HIDDEN_SIZE * 4
        
        args = qwen2.ModelArgs(
            model_type="qwen2",
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=1,
            intermediate_size=INTERMEDIATE_SIZE,
            num_attention_heads=NUM_ATTENTION_HEAD,
            num_key_value_heads=NUM_KV_HEADS,
            rms_norm_eps=1e-6,
            vocab_size=1000,
        )

        mlx_transformer_block = qwen2.TransformerBlock(args)

        mlx_attention = mlx_transformer_block.self_attn
        wq = mlx_attention.q_proj.weight
        wk = mlx_attention.k_proj.weight
        wv = mlx_attention.v_proj.weight
        wo = mlx_attention.o_proj.weight
        bq = mlx_attention.q_proj.bias
        bk = mlx_attention.k_proj.bias
        bv = mlx_attention.v_proj.bias

        mlx_mlp = mlx_transformer_block.mlp
        w_gate = mlx_mlp.gate_proj.weight
        w_up = mlx_mlp.up_proj.weight
        w_down = mlx_mlp.down_proj.weight

        w_input_layernorm = mlx_transformer_block.input_layernorm.weight
        w_post_attention_layernorm = mlx_transformer_block.post_attention_layernorm.weight
        
        user_transformer_block = qwen2_week1.Qwen2TransformerBlock(
            num_attention_heads=NUM_ATTENTION_HEAD,
            num_kv_heads=NUM_KV_HEADS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            rms_norm_eps=1e-6,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            w_input_layernorm=w_input_layernorm,
            w_post_attention_layernorm=w_post_attention_layernorm
        )

        mx.random.seed(42)
        x = mx.random.uniform(
            shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=precision
        )

        user_output = user_transformer_block(x, mask=mask)
        mlx_output = mlx_transformer_block(x, mask=mask, cache=None)
        
        assert_allclose(user_output, mlx_output, precision=precision, rtol=1e-1)

@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_utils_qwen_2_05b():
    pass


@pytest.mark.skipif(
    not qwen_2_7b_model_exists(), reason="Qwen2-7B-Instruct-MLX model not found"
)
def test_utils_qwen_2_7b():
    pass


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct-MLX model not found"
)
def test_utils_qwen_2_15b():
    pass


def helper_test_task_3(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    model = Qwen2ModelWeek1(mlx_model)
    for _ in range(iters):
        input = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=(1, 10))
        user_output = model(input)
        user_output = user_output - mx.logsumexp(user_output, keepdims=True)
        ref_output = mlx_model(input)
        ref_output = ref_output - mx.logsumexp(ref_output, keepdims=True)
        assert_allclose(user_output, ref_output, precision=mx.float16, rtol=1e-1)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_2_embedding_call():
    mlx_model, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
    )
    for _ in range(50):
        input = mx.random.randint(low=0, high=mlx_model.args.vocab_size, shape=(1, 10))
        user_output = embedding(input)
        ref_output = mlx_model.model.embed_tokens(input)
        assert_allclose(user_output, ref_output, precision=mx.float16)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_2_embedding_as_linear():
    mlx_model, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
    )
    for _ in range(50):
        input = mx.random.uniform(shape=(1, 10, mlx_model.args.hidden_size))
        user_output = embedding.as_linear(input)
        ref_output = mlx_model.model.embed_tokens.as_linear(input)
        assert_allclose(user_output, ref_output, precision=mx.float16, atol=1e-1)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct-MLX", 5)


@pytest.mark.skipif(
    not qwen_2_7b_model_exists(), reason="Qwen2-7B-Instruct-MLX model not found"
)
def test_task_3_qwen_2_7b():
    helper_test_task_3("Qwen/Qwen2-7B-Instruct-MLX", 1)


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct-MLX model not found"
)
def test_task_3_qwen_2_15b():
    helper_test_task_3("Qwen/Qwen2-1.5B-Instruct-MLX", 3)
