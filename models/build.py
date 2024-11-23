# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .simmim import build_simmim
from .module import RMSNormLayer, CCLinear
import torch.nn as nn


def build_model(config, is_pretrain=False):
    model_type = config.arch
    patch_size = config.patch_size

    # accelerate layernorm
    if config.method == 'RMS-C':
        layernorm = RMSNormLayer
        centering = True
    elif config.method == 'RMS':
        layernorm = RMSNormLayer
        centering = False
    else:
        layernorm = nn.LayerNorm
        centering = False
    
    if config.linear == 'CC':
        linearlayer = CCLinear
    else:
        linearlayer = nn.Linear

    model = SwinTransformer(img_size=224,
                            patch_size=32,
                            in_chans=3,
                            num_classes=100,
                            embed_dim=96,
                            depths=[ 2, 2, 6, 2 ],
                            num_heads=[ 3, 6, 12, 24 ],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.2,
                            ape=False,
                            norm_layer=layernorm,
                            patch_norm=True,
                            use_checkpoint=False,
                            fused_window_process=False,
                            linear_layer=linearlayer,
                            centering=centering)
    return model
