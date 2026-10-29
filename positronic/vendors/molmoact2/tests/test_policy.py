from unittest import mock

import pytest

pytest.importorskip('transformers')

from positronic.vendors.molmoact2 import policy
from positronic.vendors.molmoact2.policy import MolmoAct2Model


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(policy.torch.cuda, 'is_available', lambda: False)
    with (
        mock.patch.object(policy.AutoProcessor, 'from_pretrained') as processor,
        mock.patch.object(policy.AutoModelForImageTextToText, 'from_pretrained') as model,
    ):
        yield processor, model


def test_the_model_refuses_to_load_without_cuda_unless_told_to_use_the_cpu(no_cuda):
    processor, model = no_cuda

    with pytest.raises(RuntimeError, match='no CUDA GPU.*--model.device_map=cpu'):
        MolmoAct2Model('allenai/MolmoAct2-DROID')

    processor.assert_not_called()
    model.assert_not_called()


def test_the_model_loads_on_the_cpu_when_told_to(no_cuda):
    _processor, model = no_cuda

    MolmoAct2Model('allenai/MolmoAct2-DROID', device_map='cpu')

    assert model.call_args.kwargs['device_map'] == 'cpu'
