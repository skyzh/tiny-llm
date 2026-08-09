import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INTERFACES = {
    "quantized_matmul": ("Week 2, Day 3", "quantized_matmul.cpp"),
    "rms_norm": ("Week 2, Day 4", "week2_kernels.cpp"),
    "rope": ("Week 2, Day 4", "week2_kernels.cpp"),
    "swiglu": ("Week 2, Day 4", "week2_kernels.cpp"),
    "decode_attention": ("Week 2, Day 5", "week2_kernels.cpp"),
    "paged_cache_update": ("Week 3, Day 3", "paged_attention.cpp"),
    "quantized_embedding": ("Week 3, Day 4", "quantized_matmul.cpp"),
    "paged_attention": ("Week 3, Day 4", "paged_attention.cpp"),
}

PRIMITIVE_CLASSES = {
    "QuantizedMatmul": ("Week 2, Day 3", "quantized_matmul.cpp"),
    "Week2RMSNorm": ("Week 2, Day 4", "week2_kernels.cpp"),
    "Week2RoPE": ("Week 2, Day 4", "week2_kernels.cpp"),
    "Week2SwiGLU": ("Week 2, Day 4", "week2_kernels.cpp"),
    "Week2DecodeAttention": ("Week 2, Day 5", "week2_kernels.cpp"),
    "PagedCacheUpdate": ("Week 3, Day 3", "paged_attention.cpp"),
    "QuantizedEmbedding": ("Week 3, Day 4", "quantized_matmul.cpp"),
    "PagedAttention": ("Week 3, Day 4", "paged_attention.cpp"),
}

METAL_CHECKPOINTS = {
    "quantized_matmul.metal": {
        "quantized_matmul_vanilla_w4a16_g128": "Week 2, Day 3",
        "quantized_matvec_x4_fast_w4a16_g128": "Week 2, Day 3",
        "quantized_matmul_simdgroup_w4a16_g128": "Week 2, Day 6",
        "quantized_matmul_simdgroup_splitk_w4a16_g128": "Week 2, Day 7",
        "quantized_matmul_splitk_reduce": "Week 2, Day 7",
        "quantized_embedding_w4a16_g128": "Week 3, Day 4",
    },
    "week2_kernels.metal": {
        "week2_rms_norm": "Week 2, Day 4",
        "week2_rope": "Week 2, Day 4",
        "week2_swiglu": "Week 2, Day 4",
        "week2_decode_attention": "Week 2, Day 5",
    },
    "paged_attention.metal": {
        "paged_cache_update_kernel": "Week 3, Day 3",
        "paged_attention_decode": "Week 3, Day 4",
        "paged_attention_scalar_f32": "Week 3, Day 4",
        "paged_attention_mma_bf16_d128": "Week 3, Day 5",
    },
}

DOC_TASK_MARKERS = {
    "book/src/week2-03-quantized-matvec.md": {
        "Task 1": {"QuantizedWeights.from_mlx_layer", "QuantizedEmbedding.__call__"},
        "Task 2": {"tiny_llm_ext::quantized_matmul", "QuantizedMatmul::eval_cpu"},
        "Task 3": {
            "QuantizedMatmul::eval_gpu",
            "quantized_matmul_vanilla_w4a16_g128",
            "quantized_matvec_x4_fast_w4a16_g128",
        },
        "Task 4": {"Qwen3ModelWeek2.__init__", "Qwen3MultiHeadAttention.__call__"},
    },
    "book/src/week2-04-fused-model-kernels.md": {
        "Task 1": {
            "tiny_llm_ext::rms_norm",
            "Week2RMSNorm::eval_gpu",
            "week2_rms_norm",
        },
        "Task 2": {"tiny_llm_ext::rope", "Week2RoPE::eval_gpu", "week2_rope"},
        "Task 3": {"tiny_llm_ext::swiglu", "Week2SwiGLU::eval_gpu", "week2_swiglu"},
        "Task 4": {"Qwen3ModelWeek2.__init__", "Qwen3MLP.__call__"},
    },
    "book/src/week2-05-decode-attention.md": {
        "Task 1": {"scaled_dot_product_attention"},
        "Task 2": {
            "tiny_llm_ext::decode_attention",
            "Week2DecodeAttention::eval_gpu",
            "week2_decode_attention",
        },
        "Task 3": {"Qwen3MultiHeadAttention.__call__", "decode_attention_custom"},
    },
    "book/src/week2-06-simd-matrix-prefill.md": {
        "Task 1": {
            "QuantizedMatmul::eval_gpu",
            "quantized_matmul_simdgroup_w4a16_g128",
        },
        "Task 2": {"quantized_matmul_simdgroup_w4a16_g128"},
        "Task 3": {"quantized_matmul_simdgroup_w4a16_g128"},
        "Task 4": {"Qwen3ModelWeek2.__call__"},
        "Task 5": {"QuantizedMatmul::eval_gpu", "Qwen3ModelWeek2.__call__"},
    },
    "book/src/week2-07-split-k-prefill.md": {
        "Task 1": {"quantized_matmul_simdgroup_w4a16_g128"},
        "Task 2": {"quantized_matmul_simdgroup_splitk_w4a16_g128"},
        "Task 3": {"QuantizedMatmul::eval_gpu"},
        "Task 4": {"quantized_matmul_splitk_reduce", "QuantizedMatmul::eval_gpu"},
    },
    "book/src/week3-03-paged-attention-part1.md": {
        "Task 1": {
            "TinyKvPagedPool.__init__",
            "tiny_llm_ext::paged_cache_update",
            "paged_cache_update_kernel",
        },
        "Task 2": {"TinyKvPagedCache.__init__", "update_and_fetch"},
        "Task 3": {"TinyKvPagedCache.gather_dense", "Qwen3ModelWeek3.create_kv_cache"},
    },
    "book/src/week3-04-paged-attention-part2.md": {
        "Task 1": {"TinyKvPagedCache.block_table", "Request.try_prefill"},
        "Task 2": {
            "tiny_llm_ext::paged_attention",
            "PagedAttention::eval_gpu",
            "paged_attention_decode",
            "paged_attention_scalar_f32",
            "tiny_llm_ext::quantized_embedding",
            "quantized_embedding_w4a16_g128",
        },
        "Task 3": {"Qwen3MultiHeadAttention.__call__", "Qwen3ModelWeek3.__call__"},
        "Task 4": {"Request.try_prefill", "batch_generate"},
    },
    "book/src/week3-05-flash-attention.md": {
        "Task 1": {"paged_attention_mma_bf16_d128"},
        "Task 2": {"paged_attention_mma_bf16_d128"},
        "Task 3": {"PagedAttention::eval_gpu", "paged_attention_mma_bf16_d128"},
        "Task 4": {"Qwen3MultiHeadAttention.__call__", "PagedAttention::eval_gpu"},
    },
    "book/src/week3-optional-moe.md": {
        "Task 1": {"tiny_llm_ext::grouped_quantized_matmul", "grouped_expert_linear"},
        "Task 2": {"route_topk"},
        "Task 3": {"Moe.__init__", "Moe.__call__"},
        "Task 4": {
            "is_qwen3_moe_sparse_layer",
            "Qwen3ModelWeek3.__init__",
            "dispatch_model",
        },
    },
}


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _header_functions(path: str) -> set[str]:
    return set(re.findall(r"mx::array\s+([a-z][a-z0-9_]*)\s*\(", _read(path)))


def _header_declaration(path: str, function: str) -> str:
    source = _read(path)
    match = re.search(rf"mx::array\s+{function}\s*\(.*?\);", source, re.DOTALL)
    assert match is not None
    declaration = re.sub(r"//.*", "", match.group(0))
    declaration = re.sub(r"\s+", " ", declaration).strip()
    return re.sub(r"\s+([);,])", r"\1", declaration)


def _binding_functions(path: str) -> set[str]:
    return set(re.findall(r'm\.def\("([a-z][a-z0-9_]*)"', _read(path)))


def _binding_arguments(path: str, function: str) -> list[str]:
    source = _read(path)
    match = re.search(rf'm\.def\("{function}"(.*?)\);', source, re.DOTALL)
    assert match is not None
    return re.findall(r'"([a-z][a-z0-9_]*)"_a', match.group(1))


def test_starter_and_reference_publish_the_same_learner_extension_functions():
    expected = set(INTERFACES)
    assert _header_functions("src/extensions_ref/src/tiny_llm_ext.h") == expected
    assert _header_functions("src/extensions/src/tiny_llm_ext.h") == expected

    reference_bindings = _binding_functions("src/extensions_ref/bindings.cpp") - {
        "load_library"
    }
    starter_bindings = _binding_functions("src/extensions/bindings.cpp") - {
        "load_library",
        "axpby",
    }
    assert reference_bindings == expected
    assert starter_bindings == expected

    for function in expected:
        assert _header_declaration(
            "src/extensions/src/tiny_llm_ext.h", function
        ) == _header_declaration("src/extensions_ref/src/tiny_llm_ext.h", function)
        assert _binding_arguments(
            "src/extensions/bindings.cpp", function
        ) == _binding_arguments("src/extensions_ref/bindings.cpp", function)


def test_starter_cpp_stubs_are_registered_and_checkpoint_labeled():
    cmake = _read("src/extensions/CMakeLists.txt")
    header = _read("src/extensions/src/tiny_llm_ext.h")
    for function, (checkpoint, filename) in INTERFACES.items():
        source = _read(f"src/extensions/src/{filename}")
        assert function in source
        assert checkpoint in source
        assert f"src/{filename}" in cmake

    for primitive, (checkpoint, filename) in PRIMITIVE_CLASSES.items():
        source = _read(f"src/extensions/src/{filename}")
        assert f"class {primitive}" in header
        assert f"{primitive}::eval_cpu" in source
        assert f"{primitive}::eval_gpu" in source
        assert checkpoint in source

    for filename in {filename for _, filename in INTERFACES.values()}:
        source = _read(f"src/extensions/src/{filename}")
        assert "starter stub" in source
        assert "get_kernel" not in source
        assert "dispatch_thread" not in source


def test_starter_metal_stubs_name_each_learner_owned_kernel_and_checkpoint():
    cmake = _read("src/extensions/CMakeLists.txt")
    for filename, kernels in METAL_CHECKPOINTS.items():
        source = _read(f"src/extensions/src/{filename}")
        assert f"src/{filename}" in cmake
        for kernel, checkpoint in kernels.items():
            assert kernel in source
            assert checkpoint in source


def test_each_extension_task_names_the_exact_starter_functions_to_modify():
    for path, tasks in DOC_TASK_MARKERS.items():
        chapter = _read(path)
        for task, markers in tasks.items():
            match = re.search(
                rf"^## {task}:.*?(?=^## Task |\Z)", chapter, re.MULTILINE | re.DOTALL
            )
            assert match is not None
            for marker in markers:
                assert marker in match.group(0)


def test_optional_grouped_moe_interface_remains_a_staged_reveal():
    starter_header = _read("src/extensions/src/tiny_llm_ext.h")
    reference_header = _read("src/extensions_ref/src/tiny_llm_ext.h")
    optional_chapter = _read("book/src/week3-optional-moe.md")
    assert "grouped_quantized_matmul" not in starter_header
    assert "grouped_quantized_matmul" not in reference_header
    assert "intentionally not predeclared" in optional_chapter
    assert "`grouped_quantized_matmul` Metal kernel" in optional_chapter


def test_setup_distinguishes_the_runnable_demo_from_future_stubs():
    setup = _read("book/src/week1-07-sampling-prepare.md")
    assert "fail-closed starter stubs" in setup
    assert "this setup check calls\nonly `axpby`" in setup
