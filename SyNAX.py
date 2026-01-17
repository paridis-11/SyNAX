
"""
SyNAX: Brain-Inspired Symbiotic Learning Framework
----------------------------------------------------------------------------
Implementation based on FedNeurX_momentum.py and the provided pseudocode.
"""

import os
import random
import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import math

from Data import Data
from models.dyrep import DyRep, build_optimizer, get_params
from Node.Node import Global_Node, Node
from Trainer import Trainer
from utils.utils import (LR_scheduler, Recorder, Summary, get_log_file_name,
                         init_args, print_memory_usage, generate_node_list, initialize_device_types, set_random_seed, ann2snn)
from utils_ann2snn import evaluate_snn
from utils.misc import AverageMeter


# Custom ANN Training with Gradient Correction
def train_synax_ANN(node, epoch, round, g_t, beta=1):
    node.model.to(node.device).train()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()
    loss_m = AverageMeter()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    
    # Ensure g_t is on the correct device
    g_t_device = {k: v.to(node.device) for k, v in g_t.items()}
    
    # Calculate scaling factor for g_t to match gradient scale
    # We assume effective step size is roughly lr.
    # To be consistent with the formula: g_correct = beta * grad + (1-beta) * g_t
    # g_t is the global update direction (accumulated over a round).
    # Gradients are per-batch. 
    # If we strictly follow the formula, we just mix them.
    # However, g_t might be much larger than a single batch gradient.
    # Let's trust the pseudocode: g_{k,t}^{(j)} <- beta * grad + (1-beta) * g_t
    
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
                
            node.optimizer.zero_grad()
            output = node.model(data)
            output = output[0] # Handle model output tuple
            loss = loss_fn(output, target)
            loss.backward()
            
            # --- SyNAX: Gradient Correction ---
            with torch.no_grad():
                for name, param in node.model.named_parameters():
                    if param.grad is not None and name in g_t_device:
                        # g_{k,t}^{(j)} <- beta * grad + (1-beta) * g_t
                        # Note: param.grad is updated in place
                        param.grad.data = beta * param.grad.data + (1 - beta) * g_t_device[name]
            
            node.optimizer.step()
            
            loss_m.update(loss.item(), n=data.size(0))
            total_loss += loss
            avg_loss = total_loss / (batch_idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
            
    return avg_loss


class SyNAX_Node(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def snn_fork_synax(self, global_node):
        """
        Load global SNN model.
        Equivalent to w_{k,t}^{SNN} = w_{k,t-1}^{SNN} - G_t if w_{k,t-1} tracks global.
        """
        self.model = copy.deepcopy(global_node.snn_model).to(self.device)

    def compute_snn_feedback(self, G_t, epsilon=0.01):
        """
        Compute scalar feedback alpha.
        Normalize guidance: G_hat = G_t / ||G_t||
        L+ = F(w + eps * G_hat)
        L- = F(w - eps * G_hat)
        alpha = (L+ - L-) / (2 * eps)
        """
        # Normalize G_t
        norm = 0.0
        for k in G_t:
            norm += torch.sum(G_t[k] ** 2)
        norm = torch.sqrt(norm)
        
        if norm == 0:
            return 0.0
            
        G_hat = {k: v / norm for k, v in G_t.items()}
        
        # Helper to get loss on fixed data
        def get_loss(perturbed_model, data_batches):
            perturbed_model.eval()
            loss_fn = nn.CrossEntropyLoss()
            total_loss = 0.0
            num_samples = 0
            with torch.no_grad():
                for data, target in data_batches:
                    data, target = data.to(self.device), target.to(self.device)
                    output = perturbed_model(data)[0]
                    loss = loss_fn(output, target)
                    total_loss += loss.item() * data.size(0)
                    num_samples += data.size(0)
            return total_loss / (num_samples + 1e-8)

        # Pre-fetch data to ensure same batches for + and - perturbation
        data_batches = []
        limit_batches = 10
        for i, (data, target) in enumerate(self.train_data):
            if i >= limit_batches: break
            data_batches.append((data, target))

        # + Perturbation
        model_plus = copy.deepcopy(self.model)
        for name, param in model_plus.named_parameters():
            if name in G_hat:
                param.data += epsilon * G_hat[name]
        loss_plus = get_loss(model_plus, data_batches)
        del model_plus

        # - Perturbation
        model_minus = copy.deepcopy(self.model)
        for name, param in model_minus.named_parameters():
            if name in G_hat:
                param.data -= epsilon * G_hat[name]
        loss_minus = get_loss(model_minus, data_batches)
        del model_minus

        alpha = (loss_plus - loss_minus) / (2 * epsilon)
        return alpha


class SyNAX_Global_Node(Global_Node):
    def __init__(self, test_data, args):
        super().__init__(test_data, args)
        # Initialize g_0 = 0, G_0 = 0
        self.g_t = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        # SNN model is already initialized in super() via ann2snn(self.model)
        self.G_t = {k: torch.zeros_like(v) for k, v in self.snn_model.state_dict().items()}
        
        # Keep track of w_t (previous round models)
        self.w_t_ann = copy.deepcopy(self.model.state_dict())
        self.w_t_snn = copy.deepcopy(self.snn_model.state_dict())

    def update_global_models(self, avg_ann_update, avg_alpha, eta_ann=1.0, eta_snn=1.0):
        """
        w_{t+1}^{ANN} = w_t^{ANN} + eta_ann * Delta_w_A - eta_snn * alpha_bar * (g_t / ||g_t||)
        """
        new_ann_state = copy.deepcopy(self.model.state_dict())
        
        # Calculate global norm of g_t
        g_t_norm = 0.0
        for k in self.g_t:
            g_t_norm += torch.sum(self.g_t[k] ** 2)
        g_t_norm = torch.sqrt(g_t_norm)
        print(f"Global g_t norm: {g_t_norm:.4f}")
        
        # 1. Update ANN
        for k in new_ann_state:
            update_term = torch.zeros_like(new_ann_state[k])
            
            if k in avg_ann_update:
                update_term += eta_ann * avg_ann_update[k]
            
            if k in self.g_t:
                # Use unnormalized g_t to maintain scale
                # alpha indicates "how much" of g_t to keep/reverse
                update_term -= eta_snn * avg_alpha * self.g_t[k]
                
            new_ann_state[k] += update_term
            
        # Update self.model (w_{t+1}^{ANN})
        self.w_t_ann = copy.deepcopy(self.model.state_dict()) # Save old w_t before updating
        self.model.load_state_dict(new_ann_state)
        
        # 2. Convert to SNN (w_{t+1}^{SNN})
        self.w_t_snn = copy.deepcopy(self.snn_model.state_dict()) # Save old w_t SNN
        self.snn_model = ann2snn(self.model, self.test_data, args=self.args)
        
        # 3. Update Guidance for next round
        # g_{t+1} = w_t^{ANN} - w_{t+1}^{ANN}
        current_ann = self.model.state_dict()
        for k in self.g_t:
            self.g_t[k] = self.w_t_ann[k] - current_ann[k]
            
        # G_{t+1} = w_t^{SNN} - w_{t+1}^{SNN}
        current_snn = self.snn_model.state_dict()
        for k in self.G_t:
             # Ensure keys match (SNN structure might change slightly or keys match exactly)
             if k in current_snn and k in self.w_t_snn:
                 self.G_t[k] = self.w_t_snn[k] - current_snn[k]
             else:
                 self.G_t[k] = torch.zeros_like(self.G_t[k])


# init 
args = init_args()
set_random_seed(args.seed)
lr_initial = args.lr
args.type = 'VIT'
args.shape = 224  # For Vit

if args.wandb ==1:
    import wandb
    run_name = f"SyNAX_{args.dataset}_num-{args.node_num}_lepoch-{args.E}_lr-{args.lr}_note-{args.notes}"
    wandb.init(project="DyFL", name = run_name, entity="paridis")
    config_dict = vars(args)
    wandb.config.update(config_dict)

Data = Data(args)
recorder = Recorder(args)
file_name  = get_log_file_name(args, directory = "logs/SyNAX")

# init nodes
snn_ratio = 0.5  # SNN 设备比例
ann_devices, snn_devices = initialize_device_types(args.node_num, snn_ratio)

Global_node = SyNAX_Global_Node(Data.test_all, args)

Edge_nodes = [
    SyNAX_Node(
        k,
        Data.train_loader[k],
        Data.test_loader,
        args,
        device_type="SNN" if k in snn_devices else "ANN"
    )
    for k in range(args.node_num)
]
device = args.device

# SyNAX Hyperparameters
synax_beta = args.zero_momentum # Use args.zero_momentum as beta
eta_ann = 1.0
eta_snn = 1.0

# train
for rounds in range(args.R): 
    Summary(args)
    print(f'=============== SyNAX Round {rounds + 1}, device: {device} ===============')
    args.lr = LR_scheduler(lr_initial, rounds, args.R)
    
    # 1. Device Sampling
    sampled_nodes = random.sample(Edge_nodes, int(args.node_num * args.sample_ratio))
    ann_nodes = [node for node in sampled_nodes if node.device_type == "ANN"]
    snn_nodes = [node for node in sampled_nodes if node.device_type == "SNN"]
    
    ann_updates_list = []
    snn_alphas = []

    # 2. Parallel Execution
    
    # --- Orchestrators (ANN) ---
    for node in ann_nodes:
        print(f'--- Round: {rounds+1}, ANN Node: {node.num} ---')
        node.ann_fork(Global_node) # Get w_t (effectively w_{k,t}^{(0)})
        w_0 = copy.deepcopy(node.model.state_dict())
        
        for epoch in range(args.E):
            train_synax_ANN(node, epoch, rounds, Global_node.g_t, beta=synax_beta)
            
        # Compute Update: Delta w = w_K - w_0
        w_K = node.model.state_dict()
        delta_w = {}
        for k in w_K:
            delta_w[k] = w_K[k] - w_0[k]
        
        ann_updates_list.append(delta_w)
        node.delete_model()
        
    # --- Executors (SNN) ---
    for node in snn_nodes:
        print(f'--- Round: {rounds+1}, SNN Node: {node.num} ---')
        node.snn_fork_synax(Global_node) # Get w_t SNN
        
        alpha = node.compute_snn_feedback(Global_node.G_t, epsilon=args.zo_epsilon)
        snn_alphas.append(alpha)
        
        print(f"Node {node.num} alpha: {alpha}")
        node.delete_model()
        
    # 3. Aggregation & Update
    
    # Aggregate ANN updates
    avg_ann_update = {}
    if ann_updates_list:
        for k in ann_updates_list[0].keys():
            avg_ann_update[k] = sum(update[k] for update in ann_updates_list) / len(ann_updates_list)
    else:
        avg_ann_update = {k: torch.zeros_like(v) for k, v in Global_node.model.state_dict().items()}

    # Aggregate SNN feedback
    avg_alpha = 0.0
    if snn_alphas:
        avg_alpha = sum(snn_alphas) / len(snn_alphas)
        
    # Apply Thresholding / Limiting to alpha (as requested)
    # Using logic from FedNeurX_momentum.py lines 138-141
    original_alpha = avg_alpha
    avg_alpha = max(args.zero_min, min(args.zero_max, avg_alpha))
    print(f"📢 平均 Alpha: {original_alpha:.4f} -> Clamped: {avg_alpha:.4f}")
    print(f"📢 阈值: {args.zero_min}, {args.zero_max}")
    
    if args.wandb == 1:
        import wandb
        wandb.log({f"{args.dataset}_Alpha": avg_alpha}, step=rounds)

    print(f"Aggregated: ANN Norm: {sum(torch.norm(v) for v in avg_ann_update.values()):.4f}, SNN Alpha: {avg_alpha:.6f}")
    
    # Synthesize & Update Guidance
    Global_node.update_global_models(avg_ann_update, avg_alpha, eta_ann=eta_ann, eta_snn=eta_snn)
    
    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name=file_name, rounds=rounds)
    
    # Periodic SNN Evaluation (every 5 rounds or last round)
    if (rounds + 1) % 5 == 0 or (rounds + 1) == args.R:
        print(f"--- Evaluating SNN at round {rounds + 1} ---")
        test_stats = evaluate_snn(Global_node.test_data, Global_node.snn_model, device, args.test_T, args)

Summary(args)
