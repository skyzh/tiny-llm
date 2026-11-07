"""Supplied public-operator replay contract; importing this module needs only stdlib.

The recording interface accepts opaque fixtures and injected public callables.
Only load_implementation/make_runtime_replay import course or backend modules.
No learner model representation is part of this interface.
"""

from dataclasses import asdict, dataclass
import importlib
from typing import Any, Callable, Mapping
from types import MappingProxyType


PROFILE_KIND = "public-operator-replay"
ADAPTER_VERSION = 2
OFFSET_PREPARATION = "once-per-pointwise-sample-inside-timing"
PUBLIC_FEATURES = (
    "bounded_kv_capacity", "quantized_weights", "fast_rms_norm", "fast_rope",
    "fast_swiglu", "simdgroup_matmul", "tiled_prefill_attention",
)
COUNTERS = (
    "logical_copy_bytes", "physical_growth_copy_bytes", "slice_write_bytes",
    "growth_copy_bytes",
)


class UnsupportedCoverage(ValueError):
    """The requested checkpoint cannot have a complete public replay."""


class CleanupError(RuntimeError):
    """Ordered release causes, compatible with the declared Python 3.10 floor."""
    def __init__(self, causes):
        self.causes = tuple(causes)
        details = "; ".join(
            f"[{index}] {type(cause).__name__}: {cause}"
            for index, cause in enumerate(self.causes, 1)
        )
        super().__init__(f"cache cleanup failures: {details}")


class ReplayCleanupError(RuntimeError):
    """Retain the primary operation failure and the complete cleanup failure."""
    def __init__(self, primary, cleanup):
        self.primary = primary
        self.cleanup = cleanup
        super().__init__(
            f"operation failed ({type(primary).__name__}: {primary}); {cleanup}"
        )


@dataclass(frozen=True)
class ProfileCase:
    checkpoint: str
    phase: str
    tokens: int

    def __post_init__(self):
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be decode or prefill")
        if type(self.tokens) is not int or self.tokens <= 0:
            raise ValueError("tokens must be positive")


@dataclass(frozen=True)
class ModelShape:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    dtype: str = "bfloat16"

    def __post_init__(self):
        for name in ("layers", "hidden", "intermediate", "heads", "kv_heads", "head_dim", "vocab"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")


@dataclass(frozen=True)
class ReplayPlan:
    case: ProfileCase
    shape: ModelShape
    features: Mapping[str, bool]
    solution: str = "tiny_llm"
    prefill_logits: str = "last"
    cache_capacity: int | None = None

    def __post_init__(self):
        if self.prefill_logits not in ("last", "all"):
            raise ValueError("prefill_logits must be last or all")
        if self.cache_capacity is not None and (
            type(self.cache_capacity) is not int or self.cache_capacity < self.case.tokens
        ):
            raise ValueError("cache capacity must be at least the context length")
        if self.solution not in ("tiny_llm", "tiny_llm_ref"):
            raise ValueError("unknown solution")
        for key in PUBLIC_FEATURES:
            if type(self.features.get(key)) is not bool:
                raise ValueError(f"missing boolean feature {key}")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        extras = [key for key, enabled in self.features.items() if key not in PUBLIC_FEATURES and enabled]
        if extras or (self.solution == "tiny_llm_ref" and self.case.checkpoint == "selected"):
            raise UnsupportedCoverage("reference-only selected scheduling has no complete public replay")

    @property
    def rows(self):
        return self.case.tokens if self.case.phase == "prefill" else 1

    @property
    def prefix(self):
        return 0 if self.case.phase == "prefill" else self.case.tokens - 1

    @property
    def bound(self):
        return self.case.tokens if self.cache_capacity is None else self.cache_capacity

    @property
    def capacity(self):
        return self.bound if self.features["bounded_kv_capacity"] else None

    @property
    def head_rows(self):
        return self.rows if self.prefill_logits == "all" else 1

    @property
    def mask(self):
        return "causal" if self.case.phase == "prefill" else None


def attention_path(*, rows, context, head_dim, dtype, mask, tiled=False,
                   tiled_min=9, decode=False, max_query=0, max_context=0):
    if tiled and rows >= tiled_min and head_dim == 128 and dtype == "bfloat16":
        return "tiled"
    if (decode and rows <= max_query and context <= max_context
            and (mask is None or isinstance(mask, str))):
        return "custom"
    return "readable"


def choose_operator(features, flag, readable, optimized):
    """Pure selection seam shared by recording witnesses and runtime bindings."""
    return optimized if features[flag] else readable


def packed_weight_options(features):
    return {"use_simdgroup_matmul": features["simdgroup_matmul"]}


def _identity_offset(offset, batch_size):
    return offset


@dataclass(frozen=True)
class PublicBindings:
    """Callables bind existing solution operators, never a model object graph."""
    array: Callable[[tuple, str, str], Any]
    project: Callable[[int | None, str, Any], Any]
    pointwise: Callable[..., Any]
    attention: Callable[..., Any]
    cache_factory: Callable[[int | None], Any]
    evaluate: Callable[[list], None]
    prepare_pointwise: Callable[[int, int], Any] = _identity_offset


def _noop():
    return None


@dataclass(frozen=True)
class ReplayGroup:
    name: str
    build: Callable[[], list]
    prepare: Callable[[], None] = _noop
    finish: Callable[[], None] = _noop
    cleanup: Callable[[], None] = _noop


class CacheReplay:
    def __init__(self, plan, bindings, prefix_key, prefix_value, new_key, new_value):
        self.plan, self.bindings = plan, bindings
        self.prefix_key, self.prefix_value = prefix_key, prefix_value
        self.new_key, self.new_value = new_key, new_value
        self.caches = []
        self.before = []
        self.observation = None

    @staticmethod
    def outputs(cache, key, value, mask=None):
        result = cache.update_and_fetch(key, value, mask_length=key.shape[2], mask=mask)
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("implement TinyKvFullCache.update_and_fetch for this checkpoint")
        expected_shape = (key.shape[0], key.shape[1], cache.offset, key.shape[3])
        if (getattr(result[0], "shape", None) != expected_shape
                or getattr(result[1], "shape", None) != expected_shape
                or getattr(result[0], "dtype", None) != key.dtype
                or getattr(result[1], "dtype", None) != value.dtype
                or result[2] != cache.offset or result[3] != mask):
            raise RuntimeError("cache update must return the logical prefix, offset, dtype and mask")
        return [result[0], result[1]]

    def prepare(self):
        if self.caches:
            raise RuntimeError("previous cache sample was not disposed")
        prepared = []
        try:
            for _ in range(self.plan.shape.layers):
                cache = self.bindings.cache_factory(self.plan.capacity)
                self.caches.append(cache)
                if self.plan.prefix:
                    prepared.extend(self.outputs(cache, self.prefix_key, self.prefix_value))
            self.bindings.evaluate(prepared)
        finally:
            prepared.clear()  # do not retain setup's logical-prefix views
        self.before = []
        for cache in self.caches:
            if cache.offset != self.plan.prefix or cache.capacity != self.plan.capacity:
                raise RuntimeError("cache preparation did not preserve the selected prefix/capacity")
            self.before.append({name: getattr(cache, name) for name in COUNTERS})

    def build(self):
        outputs = []
        for cache in self.caches:
            outputs.extend(self.outputs(cache, self.new_key, self.new_value, self.plan.mask))
        return outputs

    def finish(self):
        totals = dict.fromkeys(COUNTERS, 0)
        for cache, before in zip(self.caches, self.before, strict=True):
            if cache.offset != self.plan.case.tokens or cache.capacity != self.plan.capacity:
                raise RuntimeError("cache replay changed the wrong logical length/capacity")
            for name in COUNTERS:
                totals[name] += getattr(cache, name) - before[name]
        self.observation = {
            "before_offset": self.plan.prefix, "after_offset": self.plan.case.tokens,
            "capacity": self.plan.capacity, "request_bound": self.plan.bound,
            "first_update_from_empty": self.plan.prefix == 0,
            "counter_kind": "nominal-python-accounting", "counter_deltas": totals,
        }

    def cleanup(self):
        caches, self.caches = self.caches, []
        self.before = []
        errors = []
        for cache in caches:
            try:
                cache.release()
            except Exception as error:
                errors.append(error)
        if errors:
            raise CleanupError(errors) from errors[0]


class PublicReplay:
    def __init__(self, plan: ReplayPlan, bindings: PublicBindings):
        self.plan, self.bindings = plan, bindings
        s, rows = plan.shape, plan.rows
        inputs = []

        def fixture(role, shape, dtype=None):
            value = bindings.array(shape, role, dtype or s.dtype)
            inputs.append(value)
            return value

        self.hidden = fixture("hidden", (1, rows, s.hidden))
        self.attention_input = fixture("attention-output", (1, rows, s.heads * s.head_dim))
        self.mlp_hidden = fixture("mlp-output", (1, rows, s.intermediate))
        self.final_hidden = fixture("head-input", (1, plan.head_rows, s.hidden))
        self.query = fixture("query", (1, s.heads, rows, s.head_dim))
        self.key = fixture("key", (1, s.kv_heads, plan.case.tokens, s.head_dim))
        self.value = fixture("value", (1, s.kv_heads, plan.case.tokens, s.head_dim))
        self.query_rows = fixture("query-rows", (1, rows, s.heads, s.head_dim))
        self.key_rows = fixture("key-rows", (1, rows, s.kv_heads, s.head_dim))
        self.gate = fixture("gate", (1, rows, s.intermediate))
        self.up = fixture("up", (1, rows, s.intermediate))
        self.tokens = fixture("token-ids", (1, rows), "int32")
        prefix_shape = (1, s.kv_heads, plan.prefix, s.head_dim)
        new_shape = (1, s.kv_heads, rows, s.head_dim)
        self.cache = CacheReplay(
            plan, bindings, fixture("prefix-key", prefix_shape),
            fixture("prefix-value", prefix_shape), fixture("new-key", new_shape),
            fixture("new-value", new_shape),
        )
        bindings.evaluate(inputs)

    def projections(self):
        out = []
        for layer in range(self.plan.shape.layers):
            for role, value in (
                ("q", self.hidden), ("k", self.hidden), ("v", self.hidden),
                ("o", self.attention_input), ("gate", self.hidden),
                ("up", self.hidden), ("down", self.mlp_hidden),
            ):
                out.append(self.bindings.project(layer, role, value))
        out.append(self.bindings.project(None, "head", self.final_hidden))
        return out

    def attention(self):
        return [self.bindings.attention(
            self.query, self.key, self.value, scale=self.plan.shape.head_dim ** -0.5,
            mask=self.plan.mask,
        ) for _ in range(self.plan.shape.layers)]

    def pointwise(self):
        """Normalize offsets once per sample, inside this timed group."""
        call = self.bindings.pointwise
        out = [call(None, "embedding", self.tokens)]
        rope_offset = self.bindings.prepare_pointwise(self.plan.prefix, self.tokens.shape[0])
        for layer in range(self.plan.shape.layers):
            for role, value in (("input_norm", self.hidden), ("post_norm", self.hidden),
                                ("q_norm", self.query_rows), ("k_norm", self.key_rows)):
                out.append(call(layer, role, value))
            out.extend((
                call(layer, "rope", self.query_rows, offset=rope_offset),
                call(layer, "rope", self.key_rows, offset=rope_offset),
                call(layer, "swiglu", self.gate, self.up),
                call(layer, "residual", self.hidden, self.hidden),
                call(layer, "residual", self.hidden, self.hidden),
            ))
        out.append(call(None, "final_norm", self.final_hidden))
        return out

    def groups(self):
        return (
            ReplayGroup("projections", self.projections),
            ReplayGroup("attention", self.attention),
            ReplayGroup("normalization, position, and activation", self.pointwise),
            ReplayGroup("KV cache update", self.cache.build, self.cache.prepare,
                        self.cache.finish, self.cache.cleanup),
        )


@dataclass(frozen=True)
class RuntimeImplementation:
    name: str
    checkpoints: tuple
    features: Mapping
    modules: Any
    tiled_attention: Callable
    decode_attention_max_query: int = 0
    decode_attention_max_context: int = 0


def load_implementation(name):
    """Runtime boundary: may import MLX via the selected solution's modules."""
    if name not in ("tiny_llm", "tiny_llm_ref"):
        raise ValueError("unknown solution")
    modules = {part: importlib.import_module(f"{name}.{part}") for part in (
        "attention", "basics", "embedding", "layer_norm", "positional_encoding",
        "qwen3_week2", "quantize", "week2_kernels", "kv_cache",
    )}
    model = modules["qwen3_week2"]
    return RuntimeImplementation(
        name, tuple(model.WEEK2_CHECKPOINTS), model.WEEK2_CHECKPOINT_FEATURES,
        modules, modules["week2_kernels"].dense_prefill_attention_mma,
        getattr(model, "DECODE_ATTENTION_MAX_QUERY", 0),
        getattr(model, "DECODE_ATTENTION_MAX_CONTEXT", 0),
    )


def make_runtime_replay(implementation, source, case, *, prefill_logits="last",
                        cache_capacity=None, backend=None):
    """Normalize the loaded Qwen input; never construct the learner model."""
    args = source.args
    shape = ModelShape(args.num_hidden_layers, args.hidden_size, args.intermediate_size,
                       args.num_attention_heads, args.num_key_value_heads,
                       args.head_dim, args.vocab_size)
    features = asdict(implementation.features[case.checkpoint])
    plan = ReplayPlan(case, shape, features, implementation.name, prefill_logits, cache_capacity)
    if backend is None:
        import mlx.core as backend
    mx = backend

    modules = implementation.modules
    quantize, kernels = modules["quantize"], modules["week2_kernels"]
    packed = features["quantized_weights"]
    prepared_weights = []

    def weight(layer):
        if getattr(layer, "bits", None) != 4 or getattr(layer, "group_size", None) != 128:
            raise UnsupportedCoverage("public replay requires W4/group-128 Qwen input weights")
        if packed:
            value = quantize.QuantizedWeights.from_mlx_layer(
                layer, **packed_weight_options(features),
            )
            prepared_weights.extend((value.weight, value.scales, value.biases))
        else:
            value = quantize.dequantize_linear(layer).astype(mx.bfloat16)
            prepared_weights.append(value)
        return value

    weights, norms = {}, {}
    norm_type = choose_operator(features, "fast_rms_norm", modules["layer_norm"].RMSNorm, kernels.FastRMSNorm)
    for i, layer in enumerate(source.model.layers):
        for role, original in (
            ("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
            ("v", layer.self_attn.v_proj), ("o", layer.self_attn.o_proj),
            ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj),
            ("down", layer.mlp.down_proj),
        ):
            weights[i, role] = weight(original)
        for role, width, original in (
            ("input_norm", shape.hidden, layer.input_layernorm.weight),
            ("post_norm", shape.hidden, layer.post_attention_layernorm.weight),
            ("q_norm", shape.head_dim, layer.self_attn.q_norm.weight),
            ("k_norm", shape.head_dim, layer.self_attn.k_norm.weight),
        ):
            norms[i, role] = norm_type(width, original, eps=args.rms_norm_eps)
            prepared_weights.append(original)
    if len(source.model.layers) != shape.layers:
        raise UnsupportedCoverage("Qwen layer count disagrees with configuration")
    embedded = weight(source.model.embed_tokens)
    embedding_type = choose_operator(features, "quantized_weights", modules["embedding"].Embedding, modules["embedding"].QuantizedEmbedding)
    embedding = embedding_type(shape.vocab, shape.hidden, embedded)
    if not args.tie_word_embeddings:
        weights[None, "head"] = weight(source.lm_head)
    norms[None, "final_norm"] = norm_type(shape.hidden, source.model.norm.weight, eps=args.rms_norm_eps)
    prepared_weights.append(source.model.norm.weight)
    rope_type = choose_operator(features, "fast_rope", modules["positional_encoding"].RoPE, kernels.FastRoPE)
    rope = rope_type(shape.head_dim, args.max_position_embeddings, args.rope_theta)
    mx.eval(*[item for item in prepared_weights if item is not None])
    mx.random.seed(0)

    def array(dims, role, dtype):
        if dtype == "int32":
            return mx.zeros(dims, dtype=mx.int32)
        return mx.random.normal(dims).astype(mx.bfloat16)

    def project(layer, role, value):
        if role == "head" and args.tie_word_embeddings:
            result = embedding.as_linear(value)
        else:
            linear = choose_operator(features, "quantized_weights", modules["basics"].linear, quantize.quantized_linear)
            result = linear(value, weights[layer, role])
        if result is None:
            raise RuntimeError(f"implement the selected public projection ({role})")
        return result

    def pointwise(layer, role, *values, offset=0):
        if role == "embedding":
            result = embedding(values[0])
        elif role == "rope":
            result = rope(values[0], offset=offset)
        elif role == "swiglu":
            activation = choose_operator(
                features, "fast_swiglu",
                lambda gate, up: modules["basics"].silu(gate) * up, kernels.swiglu,
            )
            result = activation(*values)
        elif role == "residual":
            result = values[0] + values[1]
        else:
            result = norms[layer, role](values[0])
        if result is None:
            raise RuntimeError(f"implement the selected public operator ({role})")
        return result

    def prepare_pointwise(offset, batch_size):
        # Called by pointwise() for every warmup/formal sample, within its clock.
        if features["fast_rope"]:
            return mx.full((batch_size,), offset, dtype=mx.int32)
        return slice(offset, offset + plan.rows)

    route = attention_path(rows=plan.rows, context=case.tokens, head_dim=shape.head_dim,
                           dtype=shape.dtype, mask=plan.mask,
                           tiled=features["tiled_prefill_attention"])

    def attention(query, key, value, *, scale, mask):
        if route == "tiled":
            result = implementation.tiled_attention(query, key, value, scale=scale, mask=mask)
        else:
            result = modules["attention"].scaled_dot_product_attention_grouped(
                query.astype(mx.float32), key.astype(mx.float32), value.astype(mx.float32),
                scale=scale, mask=mask,
            )
            if result is not None:
                result = result.astype(mx.bfloat16)
        if result is None:
            raise RuntimeError("implement the selected public attention operator")
        return result

    bindings = PublicBindings(array, project, pointwise, attention,
                              lambda capacity: modules["kv_cache"].TinyKvFullCache(capacity=capacity),
                              lambda outputs: mx.eval(*outputs) if outputs else None,
                              prepare_pointwise)
    return PublicReplay(plan, bindings)
