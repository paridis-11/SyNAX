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
                         init_args, print_memory_usage, generate_node_list, initialize_device_types, set_random_seed)
from utils_ann2snn import evaluate_snn



# init 
args = init_args()
set_random_seed(args.seed)
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
    # ann_nodes = [node for node in Edge_nodes if node.device_type == "ANN"]
    # snn_nodes = [node for node in Edge_nodes if node.device_type == "SNN"]

    # 全部设备
    full_ann_nodes = [node for node in Edge_nodes if node.device_type == "ANN"]
    full_snn_nodes = [node for node in Edge_nodes if node.device_type == "SNN"]

    # 客户端采样
    sample_size_ann = max(1, int(len(full_ann_nodes) * args.client_sample_ratio))
    sample_size_snn = max(1, int(len(full_snn_nodes) * args.client_sample_ratio))

    ann_nodes = random.sample(full_ann_nodes, sample_size_ann)
    snn_nodes = random.sample(full_snn_nodes, sample_size_snn)
    Global_node.merge_init()
    # 训练 ANN 设备
    for index, node in enumerate(ann_nodes):
        print(f'---------- Rounds: {rounds+1}, ANN Node: {node.num}, Notes: {args.notes} ---------------')
        node.ann_fork(Global_node)  # edge_node get global model
        for epoch in range(args.E):
            Train(node, epoch, rounds, type = "ANN")
        Global_node.merge_now(node, device_type="ANN")
        node.delete_model()

    zero_direct, norm = Global_node.finish_merge(num_nodes=len(ann_nodes), device_type="ANN")


    if args.mode == 'ann_only':  # 只有ANN模型
        print("🚀 没有 SNN 设备，跳过 SNN 训练，直接进行 ANN 训练结果聚合。")
    elif args.mode == 'FedNeurx':
        # 训练 SNN 设备
        scaling_factors = []
        for index, node in enumerate(snn_nodes):
            print(f'---------- Rounds: {rounds+1}, SNN Node: {node.num}, Notes: {args.notes} ---------------')
            node.snn_fork(Global_node, zero_direct)  # edge_node get global model

            for epoch in range(args.E):
                # 2️⃣ 计算扰动方向的 SNN 模型参数
                node.model = node.get_perturbed_model(zero_direct, sign=1)   # 方向 "+"
                loss_plus = Train(node, epoch, rounds, type="SNN")

                node.model = node.get_perturbed_model(zero_direct, sign=-2) # 方向 "-"
                loss_minus = Train(node, epoch, rounds, type="SNN")

                node.model = node.get_perturbed_model(zero_direct, sign=1) # 复位

                # 4️⃣ 估计扰动系数
                node.zero_order_update(loss_plus, loss_minus, zero_direct)
                scaling_factors.append(node.scaling_factor)
            node.delete_model()

        avg_scaling_factor = sum(scaling_factors) / len(scaling_factors)
        avg_scaling_factor =  max(args.zero_min, min(args.zero_max, avg_scaling_factor))
        print(f"📢 平均 Scaling Factor: {avg_scaling_factor:.4f}")
        print(f"📢 阈值: {args.zero_min}, {args.zero_max}")
        Global_node.aggregate_snn(avg_scaling_factor)

    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name = file_name, rounds = rounds)
test_stats = evaluate_snn(Global_node.test_data, Global_node.snn_model, device,args.test_T,args)

Summary(args)