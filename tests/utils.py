import numpy as np
import mlx.core as mx
import huggingface_hub
import mlx.nn as nn

AVAILABLE_STREAMS = [mx.cpu, mx.gpu]
AVAILABLE_STREAMS_IDS = ["cpu", "gpu"]
PRECISIONS = [mx.float32, mx.float16]
PRECISION_IDS = ["f32", "f16"]


def assert_allclose(
    a: mx.array,
    b: mx.array,
    precision: mx.Dtype,
    rtol: float | None = None,
    atol: float | None = None,
    message: str | None = None,
):
    a = np.array(a)
    b = np.array(b)
    if precision == mx.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-8
    elif precision == mx.float16:
        rtol = rtol or 3.0e-2
        atol = atol or 1.0e-5
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.invert(np.isclose(a, b, rtol=rtol, atol=atol))
        if diff.size > 10000 and np.sum(diff) <= 3:
            # if only a small number of elements are different in a large array, probably fine
            return
        with np.printoptions(precision=3, suppress=True):
            print("a=", a)
            print("b=", b)
            print("diff_a=", a * diff)
            print("diff_b=", b * diff)
            print("diff_a_val=", a[diff])
            print("diff_b_val=", b[diff])
            assert False, f"result mismatch: {message}"


def np_type_to_mx_type(np_type: np.dtype) -> mx.Dtype:
    if np_type == np.float32:
        return mx.float32
    elif np_type == np.float16:
        return mx.float16
    else:
        raise ValueError(f"Unsupported numpy type: {np_type}")


def qwen_3_06b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "mlx-community/Qwen3-0.6B-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-0.6B-4bit model: {e}")
        return False


def qwen_3_17b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "mlx-community/Qwen3-1.7B-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-1.7B-4bit model: {e}")
        return False


def qwen_3_8b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "mlx-community/Qwen3-8B-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-8B-4bit model: {e}")
        return False


def force_convert_bf16_to(model, precision: mx.Dtype):
    if isinstance(model, list):
        for idx, item in enumerate(model):
            res = force_convert_bf16_to(item, precision)
            if res is not None:
                model[idx] = res
    elif isinstance(model, dict):
        for key, value in model.items():
            res = force_convert_bf16_to(value, precision)
            if res is not None:
                model[key] = res
    elif isinstance(model, nn.Module):
        for _, param in model.parameters().items():
            force_convert_bf16_to(param, precision)
    elif isinstance(model, mx.array):
        if model.dtype == mx.bfloat16:
            return model.astype(precision)
        else:
            return model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
