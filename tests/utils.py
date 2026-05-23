import numpy as np
import torch

AVAILABLE_STREAMS = ["cpu", "cuda"]
AVAILABLE_STREAMS_IDS = ["cpu", "cuda"]
PRECISIONS = [torch.float32, torch.float16]
PRECISION_IDS = ["f32", "f16"]


def assert_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    precision: torch.dtype,
    rtol: float | None = None,
    atol: float | None = None,
    message: str | None = None,
):
    if a.dtype == torch.bfloat16:
        a = a.float()
    if b.dtype == torch.bfloat16:
        b = b.float()
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    if precision == torch.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-6
    elif precision == torch.float16:
        rtol = rtol or 5.0e-2
        atol = atol or 1.0e-3
    elif precision == torch.bfloat16:
        rtol = rtol or 5.0e-2
        atol = atol or 1.0e-2
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    assert a_np.shape == b_np.shape, f"shape mismatch: {a_np.shape} vs {b_np.shape}"
    if not np.allclose(a_np, b_np, rtol=rtol, atol=atol):
        diff = np.invert(np.isclose(a_np, b_np, rtol=rtol, atol=atol))
        with np.printoptions(precision=3, suppress=True):
            print("a=", a_np)
            print("b=", b_np)
            print("diff_a=", a_np * diff)
            print("diff_b=", b_np * diff)
            print("diff_a_val=", a_np[diff])
            print("diff_b_val=", b_np[diff])
            assert False, f"result mismatch: {message}"


def qwen3_0_6b_model_exists() -> bool:
    return False


def qwen3_1_7b_model_exists() -> bool:
    return False


def qwen3_4b_model_exists() -> bool:
    return False
