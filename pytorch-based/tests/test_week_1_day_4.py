import pytest
import torch
import numpy as np
from torch.testing import assert_close
from .tiny_llm_base import *

def to_torch_dtype(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float16:
        return torch.float16
    raise ValueError("Unsupported dtype")

@pytest.mark.parametrize("precision", [np.float32, np.float16])
def test_task_1_rms_norm(precision):
    SIZE = 100
    SIZE_Y = 111
    for _ in range(100):
        data_np = np.random.rand(SIZE, SIZE_Y).astype(precision)
        weight_np = np.random.rand(SIZE_Y).astype(precision)

        eps = np.finfo(precision).eps
        data = torch.tensor(data_np)
        weight = torch.tensor(weight_np)

        model = RMSNorm(SIZE_Y, weight, eps=eps)
        out = model(data)


        mean = torch.mean(data.float() ** 2, dim=-1, keepdim=True)
        ref_out = data / torch.sqrt(mean + eps)
        ref_out = ref_out * weight

        assert out.dtype == data.dtype
        assert_close(out, ref_out.to(data.dtype), atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("precision", [np.float16])
def test_task_1_rms_norm_cast_to_float32(precision):
    SIZE, SIZE_Y = 32, 64
    data = torch.tensor(np.random.uniform(-1000, 1000, size=(SIZE, SIZE_Y)).astype(precision))
    weight = torch.tensor(np.random.uniform(-1000, 1000, size=(SIZE_Y,)).astype(precision))
    eps = np.finfo(precision).eps

    model = RMSNorm(SIZE_Y, weight, eps=eps)
    out = model(data)

    mean = torch.mean(data.float() ** 2, dim=-1, keepdim=True)
    ref_out = data / torch.sqrt(mean + eps)
    ref_out = ref_out * weight

    assert_close(out, ref_out.to(data.dtype), atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("precision", [np.float32, np.float16])
@pytest.mark.parametrize("target", ["torch", "manual"])
def test_task_2_silu(precision, target):
    B, D = 10, 10
    for _ in range(100):
        x_np = np.random.rand(B, D).astype(precision)
        x = torch.tensor(x_np)

        user_output = basics.silu(x)

        if target == "torch":
            ref_output = torch.nn.functional.silu(x)
        else:
            ref_output = x * torch.sigmoid(x)

        assert_close(user_output, ref_output, atol=1e-4, rtol=1e-4)

@pytest.mark.parametrize("params", [
    {"batch_size": 1, "seq_len": 5, "dim": 4, "hidden_dim": 8},
    {"batch_size": 2, "seq_len": 16, "dim": 32, "hidden_dim": 64},
    {"batch_size": 1, "seq_len": 1, "dim": 128, "hidden_dim": 256},
])
@pytest.mark.parametrize("precision", [np.float32, np.float16])
def test_task_2_qwen_mlp(params, precision):
    B, L, D, H = params["batch_size"], params["seq_len"], params["dim"], params["hidden_dim"]
    dtype = to_torch_dtype(precision)

    x = torch.rand(B, L, D, dtype=dtype)
    w_gate = torch.rand(H, D, dtype=dtype)
    w_up = torch.rand(H, D, dtype=dtype)
    w_down = torch.rand(D, H, dtype=dtype)

    model = qwen2_week1.Qwen2MLP(D, H, w_gate, w_up, w_down)
    out = model(x)

    
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w_gate))
    up = torch.nn.functional.linear(x, w_up)
    ref_out = torch.nn.functional.linear(gate * up, w_down)

    assert_close(out, ref_out, atol=1e-3, rtol=1e-3)
