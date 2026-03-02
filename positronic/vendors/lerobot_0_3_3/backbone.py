"""Timm backbone registration for ACT.

Registers timm models into the ``torchvision.models`` namespace so that
ACT's ``getattr(torchvision.models, config.vision_backbone)(...)`` picks
them up without any changes to upstream lerobot code.

The returned module satisfies ACT's three constraints:
  1. Has a ``layer4`` direct child (for ``IntermediateLayerGetter``)
  2. Has ``fc.in_features`` (to size the Conv2d projection)
  3. Accepts and ignores torchvision-specific kwargs
"""

import math

import timm
import torchvision
from torch import nn


class _ViTToSpatial(nn.Module):
    """Runs a ViT and reshapes patch tokens into a spatial feature map."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_prefix_tokens = model.num_prefix_tokens

    def forward(self, x):
        tokens = self.model.forward_features(x)
        tokens = tokens[:, self.num_prefix_tokens :]
        b, n, c = tokens.shape
        h = w = math.isqrt(n)
        assert h * w == n, f'ViT produced {n} patch tokens (not a perfect square) — non-square inputs are not supported'
        return tokens.permute(0, 2, 1).reshape(b, c, h, w)


class _TimmViTBackbone(nn.Module):
    def __init__(self, timm_name):
        super().__init__()
        vit = timm.create_model(timm_name, pretrained=True)
        self.layer4 = _ViTToSpatial(vit)
        self.fc = type('_Fc', (), {'in_features': vit.embed_dim})()


class _FeatureExtractor(nn.Module):
    """Extracts the last feature map from a timm features_only CNN."""

    def __init__(self, feat_model):
        super().__init__()
        self.feat_model = feat_model

    def forward(self, x):
        return self.feat_model(x)[-1]


class _TimmCNNBackbone(nn.Module):
    def __init__(self, timm_name):
        super().__init__()
        feat_model = timm.create_model(timm_name, pretrained=True, features_only=True)
        self.layer4 = _FeatureExtractor(feat_model)
        self.fc = type('_Fc', (), {'in_features': feat_model.feature_info[-1]['num_chs']})()


def register_timm_backbone(name: str, timm_name: str):
    def factory(**kwargs):
        probe = timm.create_model(timm_name, pretrained=False)
        if hasattr(probe, 'layer4') and hasattr(probe, 'fc'):
            return timm.create_model(timm_name, pretrained=True)
        elif hasattr(probe, 'embed_dim'):
            return _TimmViTBackbone(timm_name)
        else:
            return _TimmCNNBackbone(timm_name)

    setattr(torchvision.models, name, factory)


BACKBONES = {
    'resnet18': ('resnet18', 'ResNet18_Weights.IMAGENET1K_V1', None),
    'resnet50': ('resnet50', 'ResNet50_Weights.IMAGENET1K_V2', None),
    'clip': ('resnet50_clip', None, 'resnet50_clip.openai'),
    'dinov2': ('resnet50_dinov2', None, 'vit_base_patch14_dinov2.lvd142m'),
    'dinov3': ('resnet50_dinov3', None, 'vit_base_patch16_dinov3.lvd1689m'),
}


_registered = False


def register_all():
    global _registered
    if _registered:
        return
    for vision_backbone, _, timm_name in BACKBONES.values():
        if timm_name:
            register_timm_backbone(vision_backbone, timm_name)
    _registered = True


register_all()
