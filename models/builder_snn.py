import logging
import timm  # 用于加载预训练的 ViT 模型
import torch
import torchvision
import yaml
# from vit_pytorch import ViT

from models.darts_model import gen_darts_model
from models.mobilenet_v1 import MobileNetV1
from models.Model import ResNet18
from models.nas_model import gen_nas_model
# from models.timm_vit_re import VisionTransformer
from timm.models.vision_transformer import VisionTransformer
from timm.models import create_model
from functools import partial

import models.model_eva
import models.model_vit

# from resnet import resnet


logger = logging.getLogger()


def build_snn_model(args, pretrained=False, pretrained_ckpt=''):
    if args.model.lower() == 'resnet18':
        model = ResNet18(args.num_classes)

    elif args.model.lower() == 'vit_base':
        model = VisionTransformer(
                img_size=224,
                patch_size=16,
                num_classes=args.num_classes,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                drop_rate=0.1,
                attn_drop_rate=0.1,
            )

    elif args.model.lower() == 'vit_tiny':
        model = VisionTransformer(
                        img_size = 224,
                        patch_size = 16,
                        num_classes = args.num_classes,
                        embed_dim = 192,
                        depth = 12,
                        num_heads = 3,
                        mlp_ratio = 4,
                        drop_rate = 0.1,
                        attn_drop_rate = 0.1
        )

    elif args.model.lower() == 'vit_small':
        model = create_model('vit_small_patch16_224_cifar10',pretrained = False, img_size = 224, num_classes = args.num_classes)

    elif args.model.lower() == 'vit_huge':
        model = VisionTransformer(
                img_size=224,
                patch_size=16,
                num_classes=args.num_classes,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                mlp_ratio=4,
                drop_rate=0.1,
                attn_drop_rate=0.1,
            )

    elif args.model.lower() == 'vit_large':
        model = VisionTransformer(
                img_size=224,
                patch_size=16,
                num_classes=args.num_classes,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                drop_rate=0.1,
                attn_drop_rate=0.1,
            )

    elif args.model.lower() == 'vit_1b':
        model = VisionTransformer(
                        img_size = 224,
                        patch_size = 16,
                        num_classes = args.num_classes,
                        embed_dim = 1280,
                        depth = 64,
                        num_heads = 16,
                        mlp_ratio = 4,
                        drop_rate = 0.1,
                        attn_drop_rate = 0.1
        )

        
    else:
        raise RuntimeError(f'Model {args.model} not found.')

    if pretrained and pretrained_ckpt != '':
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = \
                model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) != 0:
            logger.info(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.info(f'Unexpected keys in source state dict: {unexpected_keys}')

    return model
