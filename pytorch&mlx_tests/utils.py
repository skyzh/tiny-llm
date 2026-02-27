import numpy as np
from .backend import *

def assert_allclose(
    a,
    b,
    precision,
    rtol: float | None = None,
    atol: float | None = None,
):
    # Convert backend tensors to numpy arrays
    if hasattr(a, 'numpy'):
        # Handle CUDA tensors by moving to CPU first
        if hasattr(a, 'device') and str(a.device) != 'cpu':
            a = a.cpu().numpy()
        else:
            a = a.numpy()
    elif hasattr(a, '__array__'):
        a = np.array(a)
    elif not isinstance(a, np.ndarray):
        raise ValueError(f"Unsupported type for 'a': {type(a)}")

    if hasattr(b, 'numpy'):
        # Handle CUDA tensors by moving to CPU first
        if hasattr(b, 'device') and str(b.device) != 'cpu':
            b = b.cpu().numpy()
        else:
            b = b.numpy()
    elif hasattr(b, '__array__'):
        b = np.array(b)
    elif not isinstance(b, np.ndarray):
        raise ValueError(f"Unsupported type for 'b': {type(b)}")

    # Convert precision to numpy dtype if needed
    if hasattr(precision, '__name__') and precision.__name__ in ['float32', 'float16']:
        if precision.__name__ == 'float32':
            precision = np.float32
        elif precision.__name__ == 'float16':
            precision = np.float16

    if precision == np.float32:
        rtol = rtol or 1e-5
        atol = atol or 1e-8
    elif precision == np.float16:
        rtol = rtol or 5e-2
        atol = atol or 1e-3
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    if not np.allclose(a, b, rtol=rtol, atol=atol):
        with np.printoptions(precision=3, suppress=True):
            diff = np.abs(a - b)
            tol = atol + rtol * np.abs(b)
            mask = diff > tol

            max_diff = np.max(diff[mask]) if np.any(mask) else 0.0
            print(f"Max abs diff (masked): {max_diff}")
        assert False, "result mismatch"


def softmax(x, axis: int):
    """Unified softmax function that works with both backends"""
    return be_softmax(x, axis)
