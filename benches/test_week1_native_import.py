"""Inert checks of the learner module's optional native-import boundary."""

import ast
import builtins
import types
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src/tiny_llm/week2_kernels.py"


class OpaqueArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype


def load_definitions(native=None):
    """Load actual definitions with opaque MLX inputs and an inert importer."""
    tree = ast.parse(SOURCE.read_text())
    mlx_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "mlx.core" for alias in node.names)
    ]
    assert len(mlx_imports) == 1
    tree.body.remove(mlx_imports[0])
    requests = []

    def inert_import(name, globals=None, locals=None, fromlist=(), level=0):
        requests.append((name, fromlist, level))
        if name != "extensions.tiny_llm_ext" or fromlist != ("_ext",) or level:
            raise AssertionError(f"unexpected fixture import: {name}")
        if native is None:
            raise ImportError("native extension fixture unavailable")
        return types.SimpleNamespace(_ext=native)

    fixture_builtins = vars(builtins).copy()
    fixture_builtins["__import__"] = inert_import
    mx = types.SimpleNamespace(
        array=OpaqueArray,
        zeros=lambda *a, **kw: object(),
        float32="float32",
        bfloat16="bfloat16",
    )
    namespace = {"__builtins__": fixture_builtins, "mx": mx}
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace, requests


class NativeImportBoundaryTests(unittest.TestCase):
    def test_definitions_load_without_native_extension(self):
        namespace, requests = load_definitions()
        self.assertEqual(requests, [])
        for name in (
            "FastRMSNorm",
            "FastRoPE",
            "swiglu",
            "scaled_dot_product_attention",
            "decode_attention_custom",
            "dense_prefill_attention_mma",
        ):
            self.assertIn(name, namespace)
        self.assertNotIn("tiny_llm_ext_private", namespace)

    def test_native_call_imports_and_preserves_delegation(self):
        calls = []
        reshapes = []
        result = types.SimpleNamespace(
            reshape=lambda shape: reshapes.append(shape) or "result"
        )
        native = types.SimpleNamespace(
            _dense_attention_prefill_mma=lambda *args: calls.append(args) or result
        )
        namespace, requests = load_definitions(native)
        self.assertEqual(requests, [])
        q, k, v, mask = object(), object(), object(), object()
        namespace["_prepare_dense_attention_inputs"] = lambda *args: (
            q,
            k,
            v,
            mask,
            True,
            False,
        )
        query = OpaqueArray((1, 4, 9, 128), "bfloat16")
        key = OpaqueArray((1, 2, 9, 128), "bfloat16")

        actual = namespace["dense_prefill_attention_mma"](
            query, key, key, 0.25, "causal"
        )

        self.assertEqual(actual, "result")
        self.assertEqual(requests, [("extensions.tiny_llm_ext", ("_ext",), 0)])
        self.assertEqual(calls, [(q, k, v, mask, 0.25, True, False, 4, 2)])
        self.assertEqual(reshapes, [(1, 4, 9, 128)])

    def test_missing_extension_fails_only_when_native_function_is_called(self):
        namespace, requests = load_definitions()
        with self.assertRaisesRegex(
            ImportError, "native extension fixture unavailable"
        ):
            namespace["dense_prefill_attention_mma"](object(), object(), object(), 1.0)
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
