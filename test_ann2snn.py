"""
FedNeurX: 异构神经网络联邦学习框架
----------------------------------------------------------------------------
# 项目描述:
# 实现ANN设备与SNN设备的联合训练,通过中央服务器协调两种异构神经网络模型。
# 
# 训练流程:
# 1. 中央服务器同时维护ANN与SNN两个全局模型
# 2. 每轮训练,服务器先将ANN全局模型分发给ANN设备进行训练
# 3. 服务器收集并聚合ANN设备的平均梯度,更新全局ANN模型
# 4. 服务器将更新后的全局ANN模型转换为全局SNN模型
# 5. 计算转换前后SNN模型参数差异,生成扰动向量(zero_direct)
# 6. SNN客户端利用zero_direct执行零阶优化:
#    - 对本地SNN模型应用正负扰动(+/-zero_direct)
#    - 计算两种扰动下的损失值，估算梯度方向
#    - 基于估算的梯度更新本地模型
#
# 此方法使异构设备能够在联邦学习环境中高效协同训练，无需共享原始数据。
"""

import os
import random
import time

import torch
import ast

from Data import Data
from models.dyrep import DyRep, build_optimizer, get_params
from Node.Node import Global_Node, Node
from Trainer import Trainer
from utils.utils import (LR_scheduler, Recorder, Summary, get_log_file_name,
                         init_args, print_memory_usage, generate_node_list, initialize_device_types, ann2snn)
from timm.models import create_model
from utils_ann2snn import evaluate_snn, replace_test_by_testneuron, evaluate
import models.model_eva
import models.model_vit
import torchvision


# init 
args = init_args()
lr_initial = args.lr
args.type = 'VIT'
args.shape = 224  # For Vit
# args.capacity_values = generate_node_list(args)

if args.wandb ==1:
    import wandb
    run_name = f"{args.dataset}_num-{args.node_num}_lepoch-{args.E}_lr-{args.lr}_note-{args.notes}"
    wandb.init(project="DyFL", name = run_name, entity="paridis")
    config_dict = vars(args)
    wandb.config.update(config_dict)

Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
file_name  = get_log_file_name(args, directory = "logs/log2410")


# init nodes
snn_ratio = 0.5  # SNN 设备比例
ann_devices, snn_devices = initialize_device_types(args.node_num, snn_ratio)
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [
    Node(
        k,
        Data.train_loader[k],
        Data.test_loader,
        args,
        device_type="SNN" if k in snn_devices else "ANN"
    )
    for k in range(args.node_num)
]
device = args.device


# train
for rounds in range(args.R): 
    Summary(args)
    print('===============The {:d}-th round, device: {:s}==============='.format(rounds + 1, str(device)))
    args.lr = LR_scheduler(lr_initial, rounds, args.R)
    # 按设备类型分类
    ann_nodes = [node for node in Edge_nodes if node.device_type == "ANN"]
    snn_nodes = [node for node in Edge_nodes if node.device_type == "SNN"]
    Global_node.merge_init()
    # 训练 ANN 设备
    for index, node in enumerate(ann_nodes):
        print(f'---------- Rounds: {rounds+1}, ANN Node: {node.num}, Notes: {args.notes} ---------------')
        node.ann_fork(Global_node)  # edge_node get global model
        for epoch in range(args.E):
            Train(node, epoch, rounds, type = "ANN")
        Global_node.merge_now(node, device_type="ANN")
        node.delete_model()

    Global_node.finish_merge_ann(num_nodes=len(ann_nodes), device_type="ANN")


    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name = file_name, rounds = rounds)




snn_model = ann2snn(Global_node.model, Global_node.test_data, args = Global_node.args)
test_stats = evaluate_snn(Global_node.test_data, snn_model, device,args.test_T,args)

# ss_model = create_model('vit_small_patch16_224_cifar10',pretrained = False, img_size = 224, num_classes = 10)


# model_without_ddp = ss_model
# ss_model.load_state_dict(Global_node.model.state_dict(), strict=False)
# print("成功加载 Global_node.model 的权重到 ss_model！")


# # train_loader = Global_node.test_data
# train_loader = Edge_nodes[0].train_data


# replace_test_by_testneuron(ss_model, 0.99)
# ss_model = ss_model.float().to(Global_node.device)
# test_stats = evaluate(train_loader, ss_model, device)



# snn_model = ann2snn(ss_model, args)
# test_stats = evaluate_snn(Global_node.test_data, snn_model, device,args.test_T,args)

Summary(args)