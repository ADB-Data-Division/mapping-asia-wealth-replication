import torch

from maw.preprocessing.augmentation import DequantiseAEF


def test_dequantise_aef_registers_nodata_as_buffer() -> None:
    module = DequantiseAEF()
    assert "nodata_val" in dict(module.named_buffers())

    x = torch.tensor([-128, 0, 127], dtype=torch.int8)
    out = module(x)
    assert torch.isnan(out[0])
