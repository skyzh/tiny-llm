import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *


@pytest.mark.parametrize("target", ["torch", "mlx"])
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rms_norm_week_1_day_4_task_1(
    stream: mx.Stream, precision: np.dtype, target: str
):
    SIZE = 100
    SIZE_Y = 111
    with mx.stream(stream):
        for _ in range(100):
            data = np.random.rand(SIZE, SIZE_Y).astype(precision)
            weight = np.random.rand(SIZE_Y).astype(precision)
            eps = np.finfo(precision).eps
            if target == "torch":
                reference_output = torch.nn.functional.rms_norm(
                    torch.tensor(data, device=TORCH_DEVICE),
                    (SIZE_Y,),
                    torch.tensor(weight, device=TORCH_DEVICE),
                    eps=eps,
                )
            else:
                reference_output = mx.fast.rms_norm(
                    mx.array(data),
                    mx.array(weight),
                    eps=eps,
                )
            user_output = RMSNorm(SIZE_Y, mx.array(weight), eps=eps)(mx.array(data))
            assert_allclose(user_output, reference_output, precision)
