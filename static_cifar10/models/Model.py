import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.trans_utils import MyTestPlace


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.place1 = MyTestPlace(place='conv')
        self.conv1 = nn.Conv2d(
            inchannel,
            outchannel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu1 = nn.ReLU(inplace=True)

        self.place2 = MyTestPlace(place='conv')
        self.conv2 = nn.Conv2d(
            outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = sum(self.place1(x))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = sum(self.place2(out))
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, residual_block, num_classes=10, imagenet_stem=False):
        super(ResNet, self).__init__()
        self.Loc_reshape_list = None
        self.inchannel = 64
        if imagenet_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        self.layer1 = self.make_layer(residual_block, 64, 2, stride=1)
        self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
        self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
        self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
        self.place_fc = MyTestPlace(place='fc')
        self.fc = nn.Linear(512, num_classes)
        self.create_Loc_reshape_list()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))  
            currentIdx += len(flat)
        


    def forward_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = sum(self.place_fc(out))
        out = self.fc(out)
        return out

    def forward(self, x, T=1):
        out_list = []
        total_out = None
        for i in range(T):
            if x.dim() == 5:
                input_x = x[:, i]
            else:
                input_x = x.clone()
            out = self.forward_features(input_x)
            
            if total_out is None:
                total_out = out
            else:
                total_out = total_out + out
            out_list.append(total_out)
        return out_list


def ResNet18(num_classes, imagenet_stem=False):
    return ResNet(ResidualBlock, num_classes, imagenet_stem=imagenet_stem)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
