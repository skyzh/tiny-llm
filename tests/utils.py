import numpy as np
import mlx.core as mx
import torch

AVAILABLE_STREAMS = [mx.cpu, mx.gpu]
AVAILABLE_STREAMS_IDS = ["cpu", "gpu"]
PRECISIONS = [np.float32, np.float16]
PRECISION_IDS = ["f32", "f16"]
TORCH_DEVICE = torch.device("cpu")


def assert_allclose(
    a: mx.array,
    b: torch.Tensor | mx.array,
    precision: np.dtype,
    rtol: float | None = None,
    atol: float | None = None,
):
    a = np.array(a)
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    elif isinstance(b, mx.array):
        b = np.array(b)
    else:
        raise ValueError(f"Unsupported type: {type(b)}")
    if precision == np.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-8
    elif precision == np.float16:
        rtol = rtol or 3.0e-2
        atol = atol or 1.0e-5
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        with np.printoptions(precision=3, suppress=True):
            print("a=", a)
            print("b=", b)
            diff = np.invert(np.isclose(a, b, rtol=rtol, atol=atol))
            print("diff_a=", a * diff)
            print("diff_b=", b * diff)
            assert False, f"result mismatch"


def np_type_to_mx_type(np_type: np.dtype) -> mx.Dtype:
    if np_type == np.float32:
        return mx.float32
    elif np_type == np.float16:
        return mx.float16
    else:
        raise ValueError(f"Unsupported numpy type: {np_type}")
