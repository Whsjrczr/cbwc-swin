# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .module import RMSNorm, CCLinear, Centering, LayerNorm
import torch.nn as nn


def build_model(config, is_pretrain=False):
    model_type = config.arch
    patch_size = config.patch_size

    # accelerate layernorm
    if config.m == 'RMS-C':
        layernorm = RMSNorm
        centering = True
    elif config.m == 'RMS':
        layernorm = RMSNorm
        centering = False
    else:
        layernorm = LayerNorm
        centering = False
    
    if config.l == 'CC':
        linearlayer = CCLinear
    else:
        linearlayer = nn.Linear

    model = SwinTransformer(img_size=224, # config.DATA.IMG_SIZE
                            patch_size=config.patch_size, # config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=3, # config.MODEL.SWIN.IN_CHANS,
                            num_classes=config.num_classes, # config.MODEL.NUM_CLASSES,
                            embed_dim=96, # config.MODEL.SWIN.EMBED_DIM,
                            depths=[ 2, 2, 6, 2 ], # config.MODEL.SWIN.DEPTHS,
                            num_heads=[ 3, 6, 12, 24 ], # config.MODEL.SWIN.NUM_HEADS,
                            window_size=7, # config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=4., # config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=True, # config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=None, # config.MODEL.SWIN.QK_SCALE,
                            drop_rate=0.0, # config.MODEL.DROP_RATE,
                            drop_path_rate=0.2, # config.MODEL.DROP_PATH_RATE,
                            ape=False, # config.MODEL.SWIN.APE,
                            norm_layer=layernorm,
                            patch_norm=True, # config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=False, # config.TRAIN.USE_CHECKPOINT,
                            fused_window_process=False, # config.FUSED_WINDOW_PROCESS
                            linear_layer=linearlayer,
                            centering=centering)
    return model
