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
                         init_args, print_memory_usage, generate_node_list, initialize_device_types)

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
    print('===============The {:d}-th round==============='.format(rounds + 1))
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

    zero_direct = Global_node.finish_merge(num_nodes=len(ann_nodes), device_type="ANN")

    # 训练 SNN 设备
    for index, node in enumerate(snn_nodes):
        print(f'---------- Rounds: {rounds+1}, SNN Node: {node.num}, Notes: {args.notes} ---------------')
        node.snn_fork(Global_node, zero_direct)  # edge_node get global model

        if Global_node.global_grad_momentum is None:
            raise ValueError("Global gradient momentum is not initialized. Please call merge_init first.")
        for epoch in range(args.E):
            Train(node, epoch, rounds, type = "SNN")
        Global_node.merge_now(node, device_type="SNN")
        node.delete_model()

    snn_gradient = Global_node.finish_merge(num_nodes=len(snn_nodes), device_type="SNN")


    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name = file_name, rounds = rounds)




Summary(args)