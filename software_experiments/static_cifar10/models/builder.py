import logging
import os
import timm  # 用于加载预训练的 ViT 模型
import torch
import torchvision
import yaml
# from vit_pytorch import ViT

from models.darts_model import gen_darts_model
from models.mobilenet_v1 import MobileNetV1
from models.Model import ResNet18
from models.nas_model import gen_nas_model
from models.cifar_vgg import vgg16_cifar
# from models.timm_vit_re import VisionTransformer
from timm.models.vision_transformer import VisionTransformer
from timm.models import create_model
from functools import partial

# from resnet import resnet


logger = logging.getLogger()


def _default_pretrained_checkpoint(model_name):
    checkpoints = {
        'resnet18': 'pretrained/timm/resnet18_tv_in1k_backbone.pth',
        'vit_small': 'pretrained/timm/vit_small_patch16_224_augreg_in1k.safetensors',
    }
    path = checkpoints.get(model_name.lower(), '')
    return path if path and os.path.isfile(path) else ''


def _read_checkpoint(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    return checkpoint


def _map_resnet18_checkpoint(state_dict):
    mapped = {}
    for key, value in state_dict.items():
        key = key.removeprefix('module.')
        if key == 'conv1.weight':
            key = 'conv1.0.weight'
        elif key.startswith('bn1.'):
            key = 'conv1.1.' + key[len('bn1.'):]
        key = key.replace('.downsample.', '.shortcut.')
        mapped[key] = value
    return mapped


def _load_pretrained(model, model_name, checkpoint_path):
    state_dict = _read_checkpoint(checkpoint_path)
    if model_name.lower() == 'resnet18':
        state_dict = _map_resnet18_checkpoint(state_dict)
    target = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded pretrained {model_name}: tensors={len(compatible)}, "
        f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}, "
        f"checkpoint={checkpoint_path}"
    )


def build_model(args, model_name, pretrained=None, pretrained_ckpt=''):
    if pretrained is None:
        pretrained = bool(getattr(args, 'pretrained', 0))
    pretrained_ckpt = pretrained_ckpt or getattr(args, 'pretrained_ckpt', '')
    imagenet_stem = args.dataset.lower() == 'imagenet'
    if model_name.lower() == 'resnet18':
        model = ResNet18(args.num_classes, imagenet_stem=imagenet_stem)

    elif model_name.lower() in {'vgg', 'vgg16'}:
        model = vgg16_cifar(num_classes=args.num_classes, drop_rate=args.drop)

    elif model_name.lower() == 'mobilenet_v1':
        # mobilenet v1
        model = MobileNetV1(num_classes=args.num_classes)

    elif model_name.lower() == 'vit_base':
        model = create_model('vit_base_patch16_224', pretrained = False, img_size = 224, num_classes = args.num_classes)

    elif model_name.lower() == 'vit_tiny':
        model = create_model('vit_tiny_patch16_224', pretrained = False, img_size = 224, num_classes = args.num_classes)

    elif model_name.lower() == 'vit_small':
        model = create_model('vit_small_patch16_224_cifar10',pretrained = False, img_size = 224, num_classes = args.num_classes)

    elif model_name.lower() == 'vit_huge':
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

    elif model_name.lower() == 'vit_large':
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

    elif model_name.lower() == 'vit_1b':
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
        raise RuntimeError(f'Model {model_name} not found.')

    if pretrained:
        pretrained_ckpt = pretrained_ckpt or _default_pretrained_checkpoint(model_name)
        if not pretrained_ckpt:
            raise FileNotFoundError(
                f'No local pretrained checkpoint found for {model_name}. '
                'Pass --pretrained-ckpt explicitly.'
            )
        _load_pretrained(model, model_name, pretrained_ckpt)

    return model
