import copy
# from torch.cuda import random
import random
import numpy as np

from re import S

import torch
import torch.nn as nn
from numpy import s_

import models.Model as Model
from models.builder import build_model
from models.builder_snn import build_snn_model
import utils.trans_utils
from utils.MeZO import MeZO
from utils.utils import ann2snn, get_params_need_grad, compute_snn_difference
from utils.trans_utils import SOPMonitor, reset_net, accuracy

from timm.models import create_model
import models.model_eva
import models.model_vit


def init_model(model_type):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'MLP':
        model = Model.MLP()
    elif model_type == 'ResNet18':
        model = Model.ResNet18()
    elif model_type == 'CNN':
        model = Model.CNN()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05)
    return optimizer





class Node(object):
    def __init__(self, num, train_loader, test_data, args, device_type="zero"):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_data = train_loader
        self.device_type = device_type
        self.test_data = test_data
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.zero_epsilon = args.zo_epsilon
        self.scaling_factor = None


    def ann_fork(self, global_node):
        self.model = copy.deepcopy(global_node.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)


    def snn_fork(self, global_node, zero_direct):
        self.model = copy.deepcopy(global_node.snn_model).to(self.device)


    def fork(self, global_node):
        self.model = copy.deepcopy(global_node.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
    

    def get_perturbed_model(self, zero_direct, sign):
        """ 获取扰动后的 SNN 模型 """
        perturbed_model = copy.deepcopy(self.model)  # 复制模型，避免修改原始模型
        for name, param in perturbed_model.named_parameters():
            if name in zero_direct:
                param.data += sign * self.zero_epsilon * zero_direct[name]  # 直接修改模型参数
        return perturbed_model  # 返回的是一个 PyTorch 模型


    def zero_order_update(self, loss_plus, loss_minus, zero_direct, zo_lr=0.01):
        """ 通过零阶优化更新 SNN 模型 """
        # 确保 loss 不是 None
        diff = loss_plus - loss_minus
        status = "有害的" if diff > 0 else "有益的"
        print(f"loss_plus: {loss_plus:.4f}, loss_minus: {loss_minus:.4f}, diff: {diff:.4f} -> 此次参考梯度为{status}")

        if loss_plus is None or loss_minus is None:
            print("🚨 Error: loss_plus or loss_minus is None!")
            return  # 避免错误

        self.scaling_factor = -((loss_plus - loss_minus) / (2 * self.zero_epsilon))



    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        test_loader = self.test_data  # 确保 test_loader 正确
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()  # 损失函数
        total_loss = 0.0

        with torch.no_grad():  # 禁用梯度计算
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)  # 前向传播
                loss = loss_fn(output, target)  # 计算 loss
                total_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1)  # 获取预测类别
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total  # 计算准确率
        avg_loss = total_loss / total  # 计算平均 loss

        print(f"🌟 测试结果: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
        return avg_loss, acc  # 返回 loss 和 准确率


    def delete_model(self):
        # 删除模型以释放显存
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'optimizer'):
            del self.optimizer 
        torch.cuda.empty_cache()  # 清空未使用的显存缓存

    def adjust(self):
        self.model = copy.deepcopy(self.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)




class Select_Node(object):
    def __init__(self, args):
        self.args = args
        self.s_list = []   
        self.c_list = []   
        self.node_list = list(range(args.node_num))
        self.max_lost = args.max_lost   

        for j in range(self.max_lost):
            self.s_list.extend(self.node_list)
    

    def random_select(self):
        index = random.randrange(len(self.s_list))      
        chosen_number = self.s_list.pop(index)          
        self.c_list.append(chosen_number)               
        print(self.c_list)

        if len(set(self.c_list)) == self.args.node_num :
            self.s_list.extend(self.node_list)          
            [self.c_list.remove(i) for i in range(self.args.node_num)]     
        return chosen_number


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = build_model(args, args.model).to(args.device)
        self.test_data = test_data  
        self.accumulated_state_dict = None  # 累加字典，初始为 None
        self.merged_nodes = 0  # 用于记录已参与合并的节点数
        self.ann_gradient = None  # 记录 ANN 全局梯度
        self.snn_gradient = None  # 记录 SNN 全局梯度
        # self.snn_model = build_snn_model(args).to(args.device)
        self.snn_model = ann2snn(self.model, self.test_data, args = self.args)
        self.snn_old_model = copy.deepcopy(self.snn_model)
        # 动量存储
        self.global_grad_momentum = None  # 历史全局梯度
        self.ann_grad_momentum = None  # 历史 ANN 梯度
        self.snn_grad_momentum = None  # 历史 SNN 梯度
        self.zero_direct = None
        self.zero_momentum = {}

        
        self.Dict = self.model.state_dict()

        # self.edge_node = [build_model(args, args.local_model).to(args.device) for k in range(args.node_num)]
        self.init = False
        self.save = []


    def merge_init(self):
        """
        初始化或重置模型的累加字典和计数器。
        """
        # 获取当前模型的参数字典
        state_dict = self.model.state_dict()

        # 初始化累加字典为全零
        self.accumulated_state_dict = {key: torch.zeros_like(value) for key, value in state_dict.items()}
        
        # 重置已合并节点计数器
        self.merged_nodes = 0

        # 初始化动量存储
        if self.global_grad_momentum is None:
            self.global_grad_momentum = {key: torch.zeros_like(value) for key, value in state_dict.items()}
        if self.ann_grad_momentum is None:
            self.ann_grad_momentum = {key: torch.zeros_like(value) for key, value in state_dict.items()}
        if self.snn_grad_momentum is None:
            self.snn_grad_momentum = {key: torch.zeros_like(value) for key, value in state_dict.items()}

        # 初始化 ANN 和 SNN 梯度存储字典
        self.ann_gradient = {key: torch.zeros_like(value) for key, value in state_dict.items()}
        self.snn_gradient = {key: torch.zeros_like(value) for key, value in state_dict.items()}

        # 初始化动量存储
        if self.global_grad_momentum is None:
            self.global_grad_momentum = {key: torch.zeros_like(value) for key, value in state_dict.items()}

        print("Global model merge initialized.")


    def update_momentum(self, current_gradient, momentum_dict, beta=0.9):
        """
        更新动量变量。

        Args:
            current_gradient: 当前的梯度字典。
            momentum_dict: 对应的动量字典（global_grad_momentum, ann_grad_momentum, snn_grad_momentum）。
            beta: 动量系数。
        """
        for key in current_gradient.keys():
            momentum_dict[key] = beta * momentum_dict[key] + (1 - beta) * current_gradient[key]

    def merge_now(self, Edge_node, device_type):
        """
        合并客户端模型到全局模型，并计算梯度。

        Args:
            Edge_node: 当前客户端节点，包含其模型和编号。
            device_type: 客户端设备类型（"ANN" 或 "SNN"）。
        """
        Edge_node_State_List = Edge_node.model.state_dict()
        Global_node_State_List = self.model.state_dict()

        # 计算梯度：客户端模型参数 - 全局模型参数
        gradient = {key: Edge_node_State_List[key].float() - Global_node_State_List[key].float()
                    for key in Global_node_State_List.keys()}

        # 累加当前客户端模型参数（用于全局模型更新）
        for key in self.accumulated_state_dict.keys():
            self.accumulated_state_dict[key] += Edge_node_State_List[key].float()

        # 累加梯度到对应设备类型的梯度存储字典
        if device_type == "ANN":
            for key in self.ann_gradient.keys():
                self.ann_gradient[key] += gradient[key]
        elif device_type == "SNN":
            for key in self.snn_gradient.keys():
                self.snn_gradient[key] += gradient[key]

        # 更新已合并的节点数
        self.merged_nodes += 1


    def aggregate_snn(self, avg_scaling_factor):
        global_state_dict  = self.model.state_dict()
        
        # 计算更新后的参数
        for key in global_state_dict.keys():
            if key in self.ann_gradient:  # 确保梯度字典里有这个参数
                global_state_dict[key] += avg_scaling_factor * self.ann_gradient[key]

        # 更新全局模型
        self.model.load_state_dict(global_state_dict)


    def finish_merge_momentum(self, num_nodes, beta=0.2):
        """
        完成合并, 计算参数均值和带动量的扰动(zero_direct)。

        Args:
            num_nodes: 当前参与聚合的设备数量
            beta: 动量系数(0~1),越大保留旧动量越多

        Returns:
            zero_direct: 用动量平滑后的扰动向量
            norm: 当前扰动的范数（未动量平滑前）
        """
        # 1️⃣ 聚合 ANN 参数
        for key in self.accumulated_state_dict.keys():
            self.accumulated_state_dict[key] /= num_nodes

        for key in self.ann_gradient.keys():
            self.ann_gradient[key] /= num_nodes

        # 2️⃣ 保存旧的 SNN 模型
        self.snn_old_model.load_state_dict(self.snn_model.state_dict(), strict=False)

        # 3️⃣ 更新全局 ANN 模型
        self.model.load_state_dict(self.accumulated_state_dict)

        # 4️⃣ 将 ANN 转换为新的 SNN 模型
        self.snn_model = ann2snn(self.model, self.test_data, args=self.args)

        # 5️⃣ 计算扰动向量（当前回合 SNN 参数变化）
        delta_params, norm = compute_snn_difference(self.snn_model, self.snn_old_model)

        # 6️⃣ 使用动量更新 zero_direct
        zero_direct = {}
        for name in self.snn_model.state_dict().keys():
            if name in delta_params:
                new_delta = delta_params[name]

                # 如果之前没有动量，初始化为当前值
                if name not in self.zero_momentum:
                    self.zero_momentum[name] = new_delta.clone()
                else:
                    # 应用动量公式：v_t = β * v_{t-1} + (1 - β) * g_t
                    self.zero_momentum[name] = beta * self.zero_momentum[name] +  new_delta

                zero_direct[name] = self.zero_momentum[name].clone()

        print(f"🎯 Zero-direct 动量范数 (当前扰动 norm): {norm:.4f}")
        return zero_direct, norm


    def finish_merge(self, num_nodes, device_type, beta=0.9):
        """
        完成合并，计算参数均值和梯度均值。

        Args:
            num_nodes: 设备数量（用于计算平均值）。
            device_type: 客户端设备类型（"ANN" 或 "SNN"）。

        Returns:
            平均梯度字典。
        """
        # 计算全局模型的平均参数
        for key in self.accumulated_state_dict.keys():
            self.accumulated_state_dict[key] /= num_nodes

        for key in self.ann_gradient.keys():
            self.ann_gradient[key] /= num_nodes  # 
        # 将原模型转换为SNN
        self.snn_old_model.load_state_dict(self.snn_model.state_dict(), strict=False)
        # 将平均参数更新到全局模型
        self.model.load_state_dict(self.accumulated_state_dict)

        # 将新的全局模型转换为SNN
        self.snn_model = ann2snn(self.model, self.test_data, args = self.args)

        # 计算有梯度的参数的差值并归一化
        delta_params, norm = compute_snn_difference(self.snn_model, self.snn_old_model)

        print('norm:', norm)
        #  让 zero_direct 变成字典，而不是列表
        zero_direct = {name: delta_params[name].clone() for name in self.snn_model.state_dict().keys() if name in delta_params}

        # self.zero_direct = 

        return zero_direct, norm
    
    def finish_merge_ann(self, num_nodes, device_type, beta=0.9):
        """
        完成合并，计算参数均值和梯度均值。

        Args:
            num_nodes: 设备数量（用于计算平均值）。
            device_type: 客户端设备类型（"ANN" 或 "SNN"）。

        Returns:
            平均梯度字典。
        """
        # 计算全局模型的平均参数
        for key in self.accumulated_state_dict.keys():
            self.accumulated_state_dict[key] /= num_nodes

        for key in self.ann_gradient.keys():
            self.ann_gradient[key] /= num_nodes  # 


        self.model.load_state_dict(self.accumulated_state_dict)
    
    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        test_loader = self.test_data  # 确保 test_loader 正确
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()  # 损失函数
        total_loss = 0.0

        with torch.no_grad():  # 禁用梯度计算
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)[0]  # 前向传播
                loss = loss_fn(output, target)  # 计算 loss
                total_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1)  # 获取预测类别
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total  # 计算准确率
        avg_loss = total_loss / total  # 计算平均 loss

        print(f"🌟 测试结果: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
        return avg_loss, acc  # 返回 loss 和 准确率
        
    def get_normalized_global_gradient(self, epsilon=1e-3):
        """
        获取归一化的全局梯度，并乘以扰动幅度系数 epsilon。

        Args:
            epsilon: 扰动幅度系数。

        Returns:
            归一化后的全局梯度字典。
        """
        if self.global_grad_momentum is None:
            raise ValueError("Global gradient momentum is not initialized. Please call merge_init first.")

        normalized_gradient = {}
        for key, value in self.global_grad_momentum.items():
            grad_norm = torch.norm(value)  # 计算梯度的 L2 范数
            if grad_norm > 0:  # 避免除以零
                normalized_gradient[key] = epsilon * (value / grad_norm)
            else:
                normalized_gradient[key] = torch.zeros_like(value)  # 如果梯度全为零，保持为零

        return normalized_gradient



    def evaluate_snn(self, T=8):
        self.snn_model.to(self.device)
        self.snn_model.eval()  # 设置为评估模式
        
        test_loader = self.test_data  # 确保 test_loader 正确
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()  # 损失函数
        total_loss = 0.0

        with torch.no_grad():  # 禁用梯度计算
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                # 在 T 个时间步上累积输出
                accumulated_output = torch.zeros((data.shape[0], self.snn_model.num_classes), device=self.device)
                
                for t in range(T):
                    output = self.snn_model(data)[0]  # 前向传播
                    accumulated_output += output  # 累积输出
                
                averaged_output = accumulated_output / T  # 计算平均输出

                loss = loss_fn(averaged_output, target)  # 计算 loss
                total_loss += loss.item() * data.size(0)

                pred = averaged_output.argmax(dim=1)  # 获取最终预测类别
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total  # 计算准确率
        avg_loss = total_loss / total  # 计算平均 loss

        print(f"🌟 测试结果: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
        return avg_loss, acc  # 返回 loss 和 准确率
    


    


