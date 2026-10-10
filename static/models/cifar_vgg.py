import torch
import torch.nn as nn


class CIFARVGG(nn.Module):
    cfgs = {
        "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    }

    def __init__(self, num_classes=10, variant="vgg16", batch_norm=True, drop_rate=0.0):
        super().__init__()
        self.features = self._make_layers(self.cfgs[variant], batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate and drop_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
                if batch_norm:
                    layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv, nn.ReLU(inplace=True)])
                in_channels = v
        return nn.Sequential(*layers)

    def _forward_logits(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def forward(self, x, T=None):
        if T is None:
            return (self._forward_logits(x),)
        return [self._forward_logits(x) for _ in range(T)]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def vgg16_cifar(num_classes=10, drop_rate=0.0):
    return CIFARVGG(num_classes=num_classes, variant="vgg16", batch_norm=True, drop_rate=drop_rate)
