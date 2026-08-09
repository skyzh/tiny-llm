import importlib
import re
import sys
from pathlib import Path

import mlx.core as mx
import pytest


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
    "book/src/week2-03-quantize-model.md": {
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

EXTENSION_TASK_PAIRS = {
    "book/src/week2-03-quantize-model.md": {
        "Task 2": {
            (
                "src/extensions/src/quantized_matmul.cpp",
                "tiny_llm_ext::quantized_matmul",
            ),
            ("src/extensions/src/quantized_matmul.cpp", "QuantizedMatmul::eval_cpu"),
        },
        "Task 3": {
            ("src/extensions/src/quantized_matmul.cpp", "QuantizedMatmul::eval_gpu"),
            (
                "src/extensions/src/quantized_matmul.metal",
                "quantized_matmul_vanilla_w4a16_g128",
            ),
            (
                "src/extensions/src/quantized_matmul.metal",
                "quantized_matvec_x4_fast_w4a16_g128",
            ),
        },
    },
    "book/src/week2-04-fused-model-kernels.md": {
        "Task 1": {
            ("src/extensions/src/week2_kernels.cpp", "tiny_llm_ext::rms_norm"),
            ("src/extensions/src/week2_kernels.cpp", "Week2RMSNorm::eval_cpu"),
            ("src/extensions/src/week2_kernels.cpp", "Week2RMSNorm::eval_gpu"),
            ("src/extensions/src/week2_kernels.metal", "week2_rms_norm"),
        },
        "Task 2": {
            ("src/extensions/src/week2_kernels.cpp", "tiny_llm_ext::rope"),
            ("src/extensions/src/week2_kernels.cpp", "Week2RoPE::eval_cpu"),
            ("src/extensions/src/week2_kernels.cpp", "Week2RoPE::eval_gpu"),
            ("src/extensions/src/week2_kernels.metal", "week2_rope"),
        },
        "Task 3": {
            ("src/extensions/src/week2_kernels.cpp", "tiny_llm_ext::swiglu"),
            ("src/extensions/src/week2_kernels.cpp", "Week2SwiGLU::eval_cpu"),
            ("src/extensions/src/week2_kernels.cpp", "Week2SwiGLU::eval_gpu"),
            ("src/extensions/src/week2_kernels.metal", "week2_swiglu"),
        },
    },
    "book/src/week2-05-decode-attention.md": {
        "Task 2": {
            (
                "src/extensions/src/week2_kernels.cpp",
                "tiny_llm_ext::decode_attention",
            ),
            (
                "src/extensions/src/week2_kernels.cpp",
                "Week2DecodeAttention::eval_cpu",
            ),
            (
                "src/extensions/src/week2_kernels.cpp",
                "Week2DecodeAttention::eval_gpu",
            ),
            (
                "src/extensions/src/week2_kernels.metal",
                "week2_decode_attention",
            ),
        },
    },
    "book/src/week3-03-paged-attention-part1.md": {
        "Task 1": {
            (
                "src/extensions/src/paged_attention.cpp",
                "tiny_llm_ext::paged_cache_update",
            ),
            (
                "src/extensions/src/paged_attention.cpp",
                "PagedCacheUpdate::eval_cpu",
            ),
            (
                "src/extensions/src/paged_attention.cpp",
                "PagedCacheUpdate::eval_gpu",
            ),
            (
                "src/extensions/src/paged_attention.metal",
                "paged_cache_update_kernel",
            ),
        },
    },
    "book/src/week3-04-paged-attention-part2.md": {
        "Task 2": {
            (
                "src/extensions/src/paged_attention.cpp",
                "tiny_llm_ext::paged_attention",
            ),
            (
                "src/extensions/src/paged_attention.cpp",
                "PagedAttention::eval_cpu",
            ),
            (
                "src/extensions/src/paged_attention.cpp",
                "PagedAttention::eval_gpu",
            ),
            (
                "src/extensions/src/paged_attention.metal",
                "paged_attention_decode",
            ),
            (
                "src/extensions/src/paged_attention.metal",
                "paged_attention_scalar_f32",
            ),
            (
                "src/extensions/src/quantized_matmul.cpp",
                "tiny_llm_ext::quantized_embedding",
            ),
            (
                "src/extensions/src/quantized_matmul.cpp",
                "QuantizedEmbedding::eval_cpu",
            ),
            (
                "src/extensions/src/quantized_matmul.cpp",
                "QuantizedEmbedding::eval_gpu",
            ),
            (
                "src/extensions/src/quantized_matmul.metal",
                "quantized_embedding_w4a16_g128",
            ),
        },
    },
}


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _cpp_function_body(source: str, symbol: str) -> str:
    source = _strip_comments(source)
    definitions = list(
        re.finditer(rf"\b(?:mx::array|void)\s+{re.escape(symbol)}\s*\(", source)
    )
    assert len(definitions) == 1, f"expected one definition of {symbol}"

    opening_brace = source.find("{", definitions[0].end())
    assert opening_brace != -1, f"missing body for {symbol}"
    assert ";" not in source[definitions[0].end() : opening_brace], (
        f"found a declaration instead of a definition for {symbol}"
    )

    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError(f"unterminated body for {symbol}")


def _normalized_cpp_body(body: str) -> str:
    return " ".join(body.split())


def _learner_owned_cpp_definitions(source: str) -> set[str]:
    source = _strip_comments(source)
    wrappers = set(re.findall(r"\bmx::array\s+([a-z][a-z0-9_]*)\s*\(", source))
    evaluators = set(
        re.findall(r"\bvoid\s+([A-Z][A-Za-z0-9_]*::eval_(?:cpu|gpu))\s*\(", source)
    )
    return wrappers | evaluators


def _assert_checkpoint_call(source: str, symbol: str, checkpoint: str) -> None:
    assert _normalized_cpp_body(_cpp_function_body(source, symbol)) == (
        f'checkpoint_todo("{symbol}", "{checkpoint}");'
    )


def _assert_checkpoint_helper_throws(source: str) -> None:
    assert "[[noreturn]] void checkpoint_todo" in _strip_comments(source)
    assert _normalized_cpp_body(_cpp_function_body(source, "checkpoint_todo")) == (
        'throw std::runtime_error(std::string(function) + " is a starter stub; '
        'implement it in " + checkpoint);'
    )


def _assert_metal_scaffold_only(source: str) -> None:
    executable = _strip_comments(source)
    assert "[[kernel]]" not in executable
    executable = re.sub(r"^\s*#.*$", "", executable, flags=re.MULTILINE)
    executable = re.sub(r"\busing\s+namespace\s+metal\s*;", "", executable)
    assert not executable.strip(), (
        "starter Metal file contains executable declarations or bodies"
    )


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


def _binding_contract(
    source: str, function: str
) -> tuple[str, list[tuple[str, str | None]]]:
    match = re.search(rf'm\.def\("{function}"(.*?)\);', source, re.DOTALL)
    assert match is not None, f"missing binding for {function}"
    target = re.search(r"&([A-Za-z_][A-Za-z0-9_:]*)", match.group(1))
    assert target is not None, f"missing binding target for {function}"
    arguments = re.findall(
        r'"([a-z][a-z0-9_]*)"_a(?:\s*=\s*([A-Za-z0-9_:().+\-]+))?',
        match.group(1),
    )
    return target.group(1), [(name, default or None) for name, default in arguments]


def _assert_binding_parity(starter_source: str, reference_source: str) -> None:
    for function in INTERFACES:
        starter_target, starter_arguments = _binding_contract(starter_source, function)
        reference_target, reference_arguments = _binding_contract(
            reference_source, function
        )
        assert starter_target == f"tiny_llm_ext::{function}"
        assert reference_target == f"tiny_llm_ext_ref::{function}"
        assert starter_arguments == reference_arguments


def _task_body(chapter: str, task: str) -> str:
    match = re.search(
        rf"^## {task}:.*?(?=^## Task |\Z)", chapter, re.MULTILINE | re.DOTALL
    )
    assert match is not None
    return match.group(0)


def _functions_in_code_tokens(
    tokens: list[str], expected_functions: set[str]
) -> list[tuple[int, str]]:
    functions = []
    for index, token in enumerate(tokens):
        for function in expected_functions:
            if token == function or token.startswith(f"{function}("):
                functions.append((index, function))
                continue

            if "::" not in function:
                continue
            owner, short_name = function.rsplit("::", 1)
            if token == short_name and any(
                previous.startswith(f"{owner}::") for previous in tokens[:index]
            ):
                functions.append((index, function))
    return functions


def _assert_task_pairs(chapter: str, task: str, pairs: set[tuple[str, str]]) -> None:
    body = _task_body(chapter, task)
    expected_files = {filename for filename, _ in pairs}
    expected_functions = {function for _, function in pairs}
    associations: set[tuple[str, str]] = set()

    blocks = re.split(r"\n\s*\n|(?=^\s*-\s)", body, flags=re.MULTILINE)
    for block in blocks:
        tokens = re.findall(r"`([^`\n]+)`", block)
        files = []
        for index, token in enumerate(tokens):
            for filename in expected_files:
                if token in {filename, Path(filename).name}:
                    files.append((index, filename))
        functions = _functions_in_code_tokens(tokens, expected_functions)

        cursor = 0
        for index, filename in files:
            for function_index, function in functions:
                if cursor <= function_index < index:
                    associations.add((filename, function))
            cursor = index + 1

        if files and not any(
            function_index < files[0][0] for function_index, _ in functions
        ):
            filename = files[-1][1]
            for function_index, function in functions:
                if function_index >= cursor:
                    associations.add((filename, function))

    missing = pairs - associations
    assert not missing, (
        f"{task} does not bind exact starter file/function pairs: {missing}"
    )


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
    _assert_binding_parity(
        _read("src/extensions/bindings.cpp"),
        _read("src/extensions_ref/bindings.cpp"),
    )


def test_starter_cpp_stubs_are_registered_and_checkpoint_labeled():
    cmake = _read("src/extensions/CMakeLists.txt")
    header = _read("src/extensions/src/tiny_llm_ext.h")
    starter_cpp_files = {filename for _, filename in INTERFACES.values()} | {
        filename for _, filename in PRIMITIVE_CLASSES.values()
    }
    actual_definitions = set()
    for filename in starter_cpp_files:
        actual_definitions |= _learner_owned_cpp_definitions(
            _read(f"src/extensions/src/{filename}")
        )
    expected_definitions = set(INTERFACES)
    for primitive in PRIMITIVE_CLASSES:
        expected_definitions.add(f"{primitive}::eval_cpu")
        expected_definitions.add(f"{primitive}::eval_gpu")
    assert actual_definitions == expected_definitions

    for function, (checkpoint, filename) in INTERFACES.items():
        source = _read(f"src/extensions/src/{filename}")
        _assert_checkpoint_call(source, function, checkpoint)
        assert checkpoint in source
        assert f"src/{filename}" in cmake

    for primitive, (checkpoint, filename) in PRIMITIVE_CLASSES.items():
        source = _read(f"src/extensions/src/{filename}")
        assert f"class {primitive}" in header
        _assert_checkpoint_call(source, f"{primitive}::eval_cpu", checkpoint)
        _assert_checkpoint_call(source, f"{primitive}::eval_gpu", checkpoint)
        assert checkpoint in source

    for filename in {filename for _, filename in INTERFACES.values()}:
        source = _read(f"src/extensions/src/{filename}")
        _assert_checkpoint_helper_throws(source)
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
        _assert_metal_scaffold_only(source)


def test_each_extension_task_names_the_exact_starter_functions_to_modify():
    for path, tasks in DOC_TASK_MARKERS.items():
        chapter = _read(path)
        for task, markers in tasks.items():
            body = _task_body(chapter, task)
            for marker in markers:
                assert marker in body

    for path, tasks in EXTENSION_TASK_PAIRS.items():
        chapter = _read(path)
        for task, pairs in tasks.items():
            _assert_task_pairs(chapter, task, pairs)


def test_optional_grouped_moe_interface_remains_a_staged_reveal():
    reference_header = _read("src/extensions_ref/src/tiny_llm_ext.h")
    optional_chapter = _read("book/src/week3-optional-moe.md")

    starter_surfaces = [
        "src/extensions/src/tiny_llm_ext.h",
        "src/extensions/bindings.cpp",
        "src/extensions/CMakeLists.txt",
        *(
            str(path.relative_to(ROOT))
            for path in (ROOT / "src/extensions/src").glob("*.cpp")
        ),
        *(
            str(path.relative_to(ROOT))
            for path in (ROOT / "src/extensions/src").glob("*.metal")
        ),
    ]
    withheld_symbols = {
        "grouped_quantized_matmul",
        "quantized_matvec_x2",
        "quantized_matvec_x8",
    }
    for path in starter_surfaces:
        source = _read(path)
        for symbol in withheld_symbols:
            assert symbol not in source, f"{symbol} leaked into {path}"

    assert "grouped_quantized_matmul" not in reference_header
    assert "intentionally not predeclared" in optional_chapter
    assert "`grouped_quantized_matmul` Metal kernel" in optional_chapter


def test_cpp_fail_closed_guard_rejects_a_fake_success_body():
    source = _read("src/extensions/src/week2_kernels.cpp")
    fake_success = source.replace(
        'checkpoint_todo("rms_norm", "Week 2, Day 4");',
        "return mx::zeros({1});",
        1,
    )
    assert fake_success != source
    with pytest.raises(AssertionError):
        _assert_checkpoint_call(fake_success, "rms_norm", "Week 2, Day 4")


def test_built_starter_extension_fails_closed_for_every_public_operation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src/extensions"))
    sys.modules.pop("tiny_llm_ext", None)
    extension = importlib.import_module("tiny_llm_ext")

    scalar = mx.ones((1,), dtype=mx.float32)
    calls = {
        "quantized_matmul": lambda: extension.quantized_matmul(
            scalar, scalar, 128, 4, scalar, scalar
        ),
        "rms_norm": lambda: extension.rms_norm(scalar, scalar, 1e-5),
        "rope": lambda: extension.rope(scalar, scalar, 1, 10_000.0),
        "swiglu": lambda: extension.swiglu(scalar, scalar),
        "decode_attention": lambda: extension.decode_attention(
            scalar, scalar, scalar, scalar, 1.0, False, False, 1, 1
        ),
        "paged_cache_update": lambda: extension.paged_cache_update(
            scalar, scalar, 0, 0
        ),
        "quantized_embedding": lambda: extension.quantized_embedding(
            scalar, scalar, scalar, scalar, 128, 4
        ),
        "paged_attention": lambda: extension.paged_attention(
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            num_kv_heads=1,
            num_heads=1,
        ),
    }
    for function, (checkpoint, _) in INTERFACES.items():
        with pytest.raises(
            RuntimeError,
            match=rf"^{re.escape(function)} is a starter stub; implement it in "
            rf"{re.escape(checkpoint)}$",
        ):
            calls[function]()


def test_binding_guard_rejects_a_changed_python_default():
    source = _read("src/extensions/bindings.cpp")
    changed_default = source.replace(
        '"transpose_b"_a = false', '"transpose_b"_a = true', 1
    )
    assert changed_default != source
    with pytest.raises(AssertionError):
        _assert_binding_parity(
            changed_default, _read("src/extensions_ref/bindings.cpp")
        )


def test_task_pair_guard_rejects_a_nonexistent_source_path():
    path = "book/src/week2-04-fused-model-kernels.md"
    source = _read(path)
    wrong_path = source.replace(
        "`Week2RMSNorm::eval_gpu` in `src/extensions/src/week2_kernels.cpp`",
        "`Week2RMSNorm::eval_gpu` in `src/extensions/src/wrong.cpp`",
        1,
    )
    assert wrong_path != source
    with pytest.raises(AssertionError):
        _assert_task_pairs(
            wrong_path,
            "Task 1",
            EXTENSION_TASK_PAIRS[path]["Task 1"],
        )


def test_task_pair_guard_rejects_cross_swapped_valid_week_3_day_4_pairs():
    path = "book/src/week3-04-paged-attention-part2.md"
    source = _read(path)
    paged_swapped = source.replace(
        "`tiny_llm_ext::paged_attention`, `PagedAttention::eval_cpu`, and\n"
        "  `PagedAttention::eval_gpu` in `src/extensions/src/paged_attention.cpp`;",
        "`tiny_llm_ext::paged_attention` and `PagedAttention::eval_cpu` in\n"
        "  `src/extensions/src/paged_attention.cpp`; `PagedAttention::eval_gpu` in\n"
        "  `src/extensions/src/quantized_matmul.cpp`;",
        1,
    )
    assert paged_swapped != source
    cross_swapped = paged_swapped.replace(
        "`tiny_llm_ext::quantized_embedding` plus\n"
        "`QuantizedEmbedding::eval_cpu`/`eval_gpu` in\n"
        "`src/extensions/src/quantized_matmul.cpp`,",
        "`tiny_llm_ext::quantized_embedding` in\n"
        "`src/extensions/src/quantized_matmul.cpp` plus\n"
        "`QuantizedEmbedding::eval_cpu` and `QuantizedEmbedding::eval_gpu` in\n"
        "`src/extensions/src/paged_attention.cpp`,",
        1,
    )
    assert cross_swapped != paged_swapped

    pairs = EXTENSION_TASK_PAIRS[path]["Task 2"]
    body = _task_body(cross_swapped, "Task 2")
    for filename, function in pairs:
        assert filename in body
        assert function in body

    with pytest.raises(AssertionError):
        _assert_task_pairs(cross_swapped, "Task 2", pairs)


def test_metal_guard_rejects_a_leaked_kernel_body():
    source = _read("src/extensions/src/week2_kernels.metal")
    leaked_kernel = (
        source
        + """
    [[kernel]] void week2_rms_norm(device float *output [[buffer(0)]]) {
        output[0] = 0.0f;
    }
    """
    )
    with pytest.raises(AssertionError):
        _assert_metal_scaffold_only(leaked_kernel)


def test_setup_distinguishes_the_runnable_demo_from_future_stubs():
    setup = _read("book/src/week1-07-sampling-prepare.md")
    assert "fail-closed starter stubs" in setup
    assert "this setup check calls\nonly `axpby`" in setup
