import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.builder import build_model
from models.dbb_block import DiverseBranchBlock
from models.dyrep import DyRep, build_optimizer
from models.recal_bn import recal_bn
from utils.misc import AverageMeter, accuracy




def train_dyfl_ANN(node, epoch, round):
    node.model.to(node.device).train()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()
    loss_m = AverageMeter()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for batch_idx, (data, target) in enumerate(epochs):
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data = data.float().to(node.device)
            if target.dim() > 1 and target.size(1) == 1:
                target = target.long().to(node.device).squeeze(dim=1)
            else:
                target = target.long().to(node.device)
            if node.args.dataset.lower() == 'chestxray':
                target = (target.sum(dim=1) > 0).long()
            for p in node.model.parameters():
                p.grad = None
            output = node.model(data)
            # 用create_model创建模型，因为有封装，所以必须取第一个元素。
            output = output[0]
            loss = loss_fn(output, target)
            loss.backward()
            node.optimizer.step()
            loss_m.update(loss.item(), n=data.size(0))
            total_loss += loss
            avg_loss = total_loss / (batch_idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100




# def train_dyfl_SNN(node, epoch, round):
#     node.model.to(node.device)
#     # node.model.eval()
#     train_loader = node.train_data
#     total_loss = 0.0
#     num_samples = 0
#     loss_fn = nn.CrossEntropyLoss()
#     avg_loss = 0.0
#     correct = 0.0
#     acc = 0.0
#     batch_size = None

#     # 训练过程
#     description = "Node{:d}: loss={:.4f} acc={:.2f}%"
#     with tqdm(train_loader) as epochs:
#         for batch_idx, (data, target) in enumerate(epochs):
#             if batch_size is None:
#                 batch_size = data.size(0)
#             if data.size(0) != batch_size:
#                 continue
#             epochs.set_description(description.format(node.num, avg_loss, acc))
#             data, target = data.to(node.device), target.to(node.device)
#             with torch.no_grad():  
#                     output = node.model(data)  # 前向传播
#                     output = output[0]
#                     loss = loss_fn(output, target)  # 计算 loss

#             total_loss += loss.item() * data.size(0)
#             num_samples += data.size(0)
#             avg_loss = total_loss / num_samples
#             pred = output.argmax(dim=1)
#             correct += pred.eq(target.view_as(pred)).sum()
#             acc = correct / num_samples  * 100

#     return avg_loss  # 返回 loss，供 zero-order update 使用


def train_dyfl_SNN(node, epoch, round):
    node.model.to(node.device)
    node.model.eval()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_samples = 0
    correct = 0
    batch_size = None

    # 训练过程
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"

    with torch.no_grad():  # 关闭梯度计算
        with tqdm(train_loader, mininterval=2) as epochs:  # 降低 tqdm 刷新频率
            for batch_idx, (data, target) in enumerate(epochs):
                if batch_size is None:
                    batch_size = data.size(0)
                if data.size(0) != batch_size:
                    continue  # 跳过不完整 batch

                data, target = data.to(node.device, non_blocking=True), target.to(node.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = node.model(data) 
                    output = output[0]  
                    loss = loss_fn(output, target) 

                # 更新 loss 统计
                total_loss += loss.detach() * data.size(0)
                num_samples += data.size(0)

                # 计算准确率
                pred = output.argmax(dim=1)
                correct += (pred == target).sum()  

                # 计算平均 loss 和 acc
                avg_loss = total_loss / num_samples
                acc = correct / num_samples * 100

                # tqdm 进度条更新
                epochs.set_description(description.format(node.num, avg_loss.item(), acc.item()))

    return avg_loss.item()  # 返回 loss




def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    batch_time_m = AverageMeter()
    start_time = time.time()

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        input = input.float().to(args.device)
        target = target.long().to(args.device)
        with torch.no_grad():
            output = model(input)
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)

        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}



class Trainer(object):

    def __init__(self, args, type = 'ANN'):
        self.args = args  # 存储 args，后续可能需要
        self.train_ANN = train_dyfl_ANN
        self.train_SNN = train_dyfl_SNN


    def __call__(self, node, epoch, round, type = 'ANN'):
        if type == "ANN":
            return self.train_ANN(node, epoch, round)
        elif type == "SNN":
            return self.train_SNN(node, epoch, round)
        else:
            raise ValueError("Invalid type: must be 'ANN' or 'SNN'")





