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


def test_task_1_attention_composes_learner_rmsnorm_for_q_and_k(monkeypatch):
    hidden_size = 32
    num_attention_heads = 4
    num_kv_heads = 2
    head_dim = 12
    rms_norm_eps = 1e-6
    q_norm = mx.arange(head_dim).astype(mx.bfloat16)
    k_norm = (mx.arange(head_dim) + 1).astype(mx.bfloat16)
    norm_inits = []

    class RecordingRMSNorm:
        def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
            self.dim = dim
            self.weight = weight
            self.eps = eps
            norm_inits.append(self)

    monkeypatch.setattr(qwen3_week1, "RMSNorm", RecordingRMSNorm)

    attention = qwen3_week1.Qwen3MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        wq=mx.zeros((num_attention_heads * head_dim, hidden_size)),
        wk=mx.zeros((num_kv_heads * head_dim, hidden_size)),
        wv=mx.zeros((num_kv_heads * head_dim, hidden_size)),
        wo=mx.zeros((hidden_size, num_attention_heads * head_dim)),
        q_norm=q_norm,
        k_norm=k_norm,
        rms_norm_eps=rms_norm_eps,
    )

    assert norm_inits == [attention.q_norm, attention.k_norm]
    assert [norm.dim for norm in norm_inits] == [head_dim, head_dim]
    assert [norm.eps for norm in norm_inits] == [rms_norm_eps, rms_norm_eps]
    assert norm_inits[0].weight is q_norm
    assert norm_inits[1].weight is k_norm


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


@pytest.mark.parametrize("tie_word_embeddings", [True, False], ids=["tied", "untied"])
def test_task_3_model_assembly_without_download(monkeypatch, tie_word_embeddings: bool):
    num_hidden_layers = 3
    hidden_size = 6
    vocab_size = 11
    num_attention_heads = 3
    num_key_value_heads = 1
    head_dim = 4
    intermediate_size = 7
    rms_norm_eps = 3e-4
    max_position_embeddings = 37
    rope_theta = 12_345

    def quantized(name: str):
        return SimpleNamespace(name=name)

    def norm_weight(size: int, value: float):
        return mx.full((size,), value, dtype=mx.bfloat16)

    layers = []
    for layer_index in range(num_hidden_layers):
        prefix = f"layer_{layer_index}"
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=quantized(f"{prefix}.q_proj"),
                    k_proj=quantized(f"{prefix}.k_proj"),
                    v_proj=quantized(f"{prefix}.v_proj"),
                    o_proj=quantized(f"{prefix}.o_proj"),
                    q_norm=SimpleNamespace(
                        weight=norm_weight(head_dim, layer_index + 1)
                    ),
                    k_norm=SimpleNamespace(
                        weight=norm_weight(head_dim, layer_index + 2)
                    ),
                ),
                mlp=SimpleNamespace(
                    gate_proj=quantized(f"{prefix}.gate_proj"),
                    up_proj=quantized(f"{prefix}.up_proj"),
                    down_proj=quantized(f"{prefix}.down_proj"),
                ),
                input_layernorm=SimpleNamespace(
                    weight=norm_weight(hidden_size, layer_index + 3)
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=norm_weight(hidden_size, layer_index + 4)
                ),
            )
        )

    mlx_model = SimpleNamespace(
        args=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
        ),
        model=SimpleNamespace(
            embed_tokens=quantized("embed_tokens"),
            layers=layers,
            norm=SimpleNamespace(weight=norm_weight(hidden_size, 9)),
        ),
        lm_head=quantized("lm_head"),
    )

    dequantized = []
    block_inits = []
    calls = []

    def fake_dequantize_linear(layer):
        dequantized.append(layer.name)
        return mx.full((1,), len(dequantized), dtype=mx.bfloat16)

    class RecordingEmbedding:
        def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            assert weight.dtype == mx.bfloat16
            calls.append(("embedding_init", vocab_size, embedding_dim))

        def __call__(self, token_ids: mx.array) -> mx.array:
            calls.append(("embedding", tuple(token_ids.shape)))
            return mx.ones((*token_ids.shape, self.embedding_dim), dtype=mx.bfloat16)

        def as_linear(self, hidden: mx.array) -> mx.array:
            calls.append(("tied_projection", tuple(hidden.shape)))
            return mx.full((*hidden.shape[:-1], self.vocab_size), 13, dtype=mx.bfloat16)

    class RecordingBlock:
        def __init__(
            self,
            num_attention_heads: int,
            num_kv_heads: int,
            hidden_size: int,
            head_dim: int,
            intermediate_size: int,
            rms_norm_eps: float,
            wq: mx.array,
            wk: mx.array,
            wv: mx.array,
            wo: mx.array,
            q_norm: mx.array,
            k_norm: mx.array,
            w_gate: mx.array,
            w_up: mx.array,
            w_down: mx.array,
            w_input_layernorm: mx.array,
            w_post_attention_layernorm: mx.array,
            max_seq_len: int = 32768,
            theta: int = 1000000,
        ):
            self.layer_index = len(block_inits)
            block_inits.append(
                {
                    "num_attention_heads": num_attention_heads,
                    "num_kv_heads": num_kv_heads,
                    "hidden_size": hidden_size,
                    "head_dim": head_dim,
                    "intermediate_size": intermediate_size,
                    "rms_norm_eps": rms_norm_eps,
                    "max_seq_len": max_seq_len,
                    "theta": theta,
                    "weights": (
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm,
                        k_norm,
                        w_gate,
                        w_up,
                        w_down,
                        w_input_layernorm,
                        w_post_attention_layernorm,
                    ),
                }
            )

        def __call__(self, hidden: mx.array, mask=None) -> mx.array:
            calls.append(("block", self.layer_index, mask, tuple(hidden.shape)))
            return (hidden.astype(mx.float32) + self.layer_index + 1).astype(
                mx.bfloat16
            )

    class RecordingNorm:
        def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
            assert dim == hidden_size
            assert weight.dtype == mx.bfloat16
            assert eps == rms_norm_eps

        def __call__(self, hidden: mx.array) -> mx.array:
            calls.append(("final_norm", tuple(hidden.shape)))
            return (hidden.astype(mx.float32) + 7).astype(mx.bfloat16)

    def recording_linear(hidden: mx.array, weight: mx.array) -> mx.array:
        assert weight.dtype == mx.bfloat16
        calls.append(("untied_projection", tuple(hidden.shape)))
        return mx.full((*hidden.shape[:-1], vocab_size), 17, dtype=mx.bfloat16)

    monkeypatch.setattr(qwen3_week1, "dequantize_linear", fake_dequantize_linear)
    monkeypatch.setattr(qwen3_week1, "Embedding", RecordingEmbedding)
    monkeypatch.setattr(qwen3_week1, "Qwen3TransformerBlock", RecordingBlock)
    monkeypatch.setattr(qwen3_week1, "RMSNorm", RecordingNorm)
    monkeypatch.setattr(qwen3_week1, "linear", recording_linear)

    model = Qwen3ModelWeek1(mlx_model)
    token_ids = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    output = model(token_ids)

    expected_dequantized = ["embed_tokens"]
    for layer_index in range(num_hidden_layers):
        prefix = f"layer_{layer_index}"
        expected_dequantized.extend(
            [
                f"{prefix}.q_proj",
                f"{prefix}.k_proj",
                f"{prefix}.v_proj",
                f"{prefix}.o_proj",
                f"{prefix}.gate_proj",
                f"{prefix}.up_proj",
                f"{prefix}.down_proj",
            ]
        )
    if not tie_word_embeddings:
        expected_dequantized.append("lm_head")
    assert dequantized == expected_dequantized

    assert len(block_inits) == num_hidden_layers
    for block_init in block_inits:
        assert block_init["num_attention_heads"] == num_attention_heads
        assert block_init["num_kv_heads"] == num_key_value_heads
        assert block_init["hidden_size"] == hidden_size
        assert block_init["head_dim"] == head_dim
        assert block_init["intermediate_size"] == intermediate_size
        assert block_init["rms_norm_eps"] == rms_norm_eps
        assert block_init["max_seq_len"] == max_position_embeddings
        assert block_init["theta"] == rope_theta
        assert all(weight.dtype == mx.bfloat16 for weight in block_init["weights"])

    expected_calls = [
        ("embedding_init", vocab_size, hidden_size),
        ("embedding", (2, 3)),
        ("block", 0, "causal", (2, 3, hidden_size)),
        ("block", 1, "causal", (2, 3, hidden_size)),
        ("block", 2, "causal", (2, 3, hidden_size)),
        ("final_norm", (2, 3, hidden_size)),
        (
            "tied_projection" if tie_word_embeddings else "untied_projection",
            (2, 3, hidden_size),
        ),
    ]
    assert calls == expected_calls
    assert output.shape == (2, 3, vocab_size)
    assert output.dtype == mx.bfloat16
    assert np.all(
        np.array(output.astype(mx.float32)) == (13 if tie_word_embeddings else 17)
    )


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
