
"""
SyNAX: Brain-Inspired Symbiotic Learning Framework
----------------------------------------------------------------------------
Implementation based on the provided pseudocode.
"""

import os
import random
import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from Data import Data
from Node.Node import Global_Node, Node
from Trainer import Trainer
from utils.utils import (LR_scheduler, Recorder, Summary, get_log_file_name,
                         init_args, initialize_device_types, set_random_seed, compute_snn_difference, ann2snn)
from utils_ann2snn import evaluate_snn
from utils.misc import AverageMeter

# Custom training function for SyNAX ANN Client
def train_synax_ANN(node, epoch, round, g_t, beta=0.9):
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
    # g_t represents the total update of the previous round.
    # To add it to a single batch gradient, we scale it down by the number of steps and learning rate.
    # This assumes SGD-like behavior. For Adam, this is an approximation.
    total_steps = len(train_loader) * node.args.E
    # Avoid division by zero
    if total_steps == 0: total_steps = 1
    # We assume effective step size is roughly lr.
    g_scaling = 1.0 / (total_steps * node.args.lr) if node.args.lr > 0 else 1.0

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
            # g_{k,t}^{(j)} <- beta * grad + (1-beta) * g_t
            # We scale g_t to be comparable to a single gradient.
            with torch.no_grad():
                for name, param in node.model.named_parameters():
                    if param.grad is not None and name in g_t_device:
                        # Apply scaling to g_t component
                        corrected_g = (1 - beta) * g_t_device[name] * g_scaling
                        param.grad.data = beta * param.grad.data + corrected_g
            
            node.optimizer.step()
            
            loss_m.update(loss.item(), n=data.size(0))
            total_loss += loss
            avg_loss = total_loss / (batch_idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100

class SyNAX_Node(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def snn_fork_synax(self, global_node, G_t):
        """
        Update local SNN model using Global Guidance G_t.
        w_snn = w_snn_prev - G_t
        But since clients are stateless in this implementation (simulated), 
        we simulate this by taking the global SNN model which should ideally correspond 
        to w_snn_prev - G_t if w_snn_prev was w_global_prev.
        
        However, strictly following the pseudocode:
        w_{k,t}^{SNN} = w_{k,t-1}^{SNN} - G_t
        And G_t = w_{t-1}^{SNN} - w_{t}^{SNN}
        So w_{k,t}^{SNN} = w_{k,t-1}^{SNN} - (w_{t-1}^{SNN} - w_{t}^{SNN})
        If w_{k,t-1}^{SNN} == w_{t-1}^{SNN} (Client had global model), then w_{k,t}^{SNN} == w_{t}^{SNN}.
        
        So we can just load the current global SNN model.
        """
        self.model = copy.deepcopy(global_node.snn_model).to(self.device)
        self.G_t = G_t

    def compute_snn_feedback(self, epsilon=0.01):
        """
        Compute scalar feedback alpha.
        Normalize guidance: G_hat = G_t / ||G_t||
        L+ = F(w + eps * G_hat)
        L- = F(w - eps * G_hat)
        alpha = (L+ - L-) / (2 * eps)
        """
        # Normalize G_t
        G_t = self.G_t
        norm = 0.0
        for k in G_t:
            norm += torch.sum(G_t[k] ** 2)
        norm = torch.sqrt(norm)
        
        if norm == 0:
            return 0.0
            
        G_hat = {k: v / norm for k, v in G_t.items()}
        
        # We need to compute Loss for + and - perturbations
        # Helper to run validation/eval loop
        def get_loss(perturbed_model):
            perturbed_model.eval()
            loss_fn = nn.CrossEntropyLoss()
            total_loss = 0.0
            num_samples = 0
            with torch.no_grad():
                # Use a subset or full train data? Pseudocode says "Compute loss", implies on local data.
                # We use a few batches or full loader.
                # To save time, let's use the full loader but with no_grad (fast).
                for data, target in self.train_data:
                    data, target = data.to(self.device), target.to(self.device)
                    output = perturbed_model(data)[0]
                    loss = loss_fn(output, target)
                    total_loss += loss.item() * data.size(0)
                    num_samples += data.size(0)
            return total_loss / num_samples

        # + Perturbation
        model_plus = copy.deepcopy(self.model)
        for name, param in model_plus.named_parameters():
            if name in G_hat:
                param.data += epsilon * G_hat[name]
        loss_plus = get_loss(model_plus)
        del model_plus

        # - Perturbation
        model_minus = copy.deepcopy(self.model)
        for name, param in model_minus.named_parameters():
            if name in G_hat:
                param.data -= epsilon * G_hat[name]
        loss_minus = get_loss(model_minus)
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
        w_{t+1}^{ANN} = w_t^{ANN} + eta_ann * Delta_w_A - eta_snn * alpha_bar * g_t
        """
        new_ann_state = copy.deepcopy(self.model.state_dict())
        
        # 1. Update ANN
        for k in new_ann_state:
            update_term = torch.zeros_like(new_ann_state[k])
            
            if k in avg_ann_update:
                update_term += eta_ann * avg_ann_update[k]
            
            if k in self.g_t:
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


# Main Execution
if __name__ == '__main__':
    # init 
    args = init_args()
    set_random_seed(args.seed)
    lr_initial = args.lr
    args.type = 'VIT'
    args.shape = 224
    
    # SyNAX Parameters
    eta_ann = 1.0
    eta_snn = 1.0 # args.lr? Or keep 1.0 as the update is already scaled by local LR?
                  # Typically in FL aggregation, we just average. So eta_ann=1.0.
                  # For the SNN term, we need a reasonable scale.
                  # Let's start with 1.0 or args.lr.
                  # Given the formula: w = w + eta * delta ... delta is already (w_new - w_old).
    
    synax_beta = 0.9 # Momentum factor
    
    if args.wandb == 1:
        import wandb
        run_name = f"SyNAX_{args.dataset}_num-{args.node_num}_R-{args.R}_note-{args.notes}"
        wandb.init(project="DyFL", name = run_name, entity="paridis")
        config_dict = vars(args)
        wandb.config.update(config_dict)

    Data = Data(args)
    # Train = Trainer(args) # We use custom training loop
    recorder = Recorder(args)
    file_name  = get_log_file_name(args, directory = "logs/SyNAX")

    # init nodes
    snn_ratio = 0.5
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

    # train
    for rounds in range(args.R): 
        Summary(args)
        print(f'=============== SyNAX Round {rounds + 1}, device: {device} ===============')
        args.lr = LR_scheduler(lr_initial, rounds, args.R)
        
        # Sample Clients
        sampled_nodes = random.sample(Edge_nodes, int(args.node_num * args.sample_ratio))
        ann_nodes = [node for node in sampled_nodes if node.device_type == "ANN"]
        snn_nodes = [node for node in sampled_nodes if node.device_type == "SNN"]
        
        # Lists to store updates
        ann_updates_list = [] # List of Delta w_k
        snn_alphas = []       # List of alpha_k
        
        # Parallel Execution (Sequential in simulation)
        
        # 1. Orchestrators (ANN)
        for node in ann_nodes:
            print(f'--- Round: {rounds+1}, ANN Node: {node.num} ---')
            # Fork: Receive g_t (implicit in Global_node.g_t)
            # Standard fork: copy model
            node.ann_fork(Global_node)
            
            # Local Training with Gradient Correction
            w_0 = copy.deepcopy(node.model.state_dict()) # Initial local model
            
            for epoch in range(args.E):
                train_synax_ANN(node, epoch, rounds, Global_node.g_t, beta=synax_beta)
                
            # Compute Update: Delta w = w_K - w_0
            w_K = node.model.state_dict()
            delta_w = {}
            for k in w_K:
                delta_w[k] = w_K[k] - w_0[k]
            
            ann_updates_list.append(delta_w)
            node.delete_model()
            
        # 2. Executors (SNN)
        for node in snn_nodes:
            print(f'--- Round: {rounds+1}, SNN Node: {node.num} ---')
            # Fork: Receive G_t
            node.snn_fork_synax(Global_node, Global_node.G_t)
            
            # Compute Scalar Feedback
            alpha = node.compute_snn_feedback(epsilon=args.zo_epsilon)
            snn_alphas.append(alpha)
            
            print(f"Node {node.num} alpha: {alpha}")
            node.delete_model()

        # Server Aggregation & Symbiotic Update
        
        # Aggregate ANN updates
        avg_ann_update = {}
        if ann_updates_list:
            for k in ann_updates_list[0].keys():
                avg_ann_update[k] = sum(update[k] for update in ann_updates_list) / len(ann_updates_list)
        else:
            # Handle case with no ANN nodes sampled (unlikely if ratio balanced)
            avg_ann_update = {k: torch.zeros_like(v) for k, v in Global_node.model.state_dict().items()}

        # Aggregate SNN feedback
        avg_alpha = 0.0
        if snn_alphas:
            avg_alpha = sum(snn_alphas) / len(snn_alphas)
            
        print(f"Aggregated: ANN Updates (Norm: {sum(torch.norm(v) for v in avg_ann_update.values()):.4f}), SNN Alpha: {avg_alpha:.6f}")
        
        # Synthesize new global model & Update Guidance
        Global_node.update_global_models(avg_ann_update, avg_alpha, eta_ann=eta_ann, eta_snn=eta_snn)
        
        # Logging
        recorder.validate(Global_node)
        recorder.printer(Global_node, file_name = file_name, rounds = rounds)
        
        if args.wandb == 1:
            wandb.log({"SNN_Alpha": avg_alpha}, step=rounds)

        # Periodic SNN Evaluation (every 5 rounds or last round)
        if (rounds + 1) % 5 == 0 or (rounds + 1) == args.R:
            print(f"--- Evaluating SNN at round {rounds + 1} ---")
            test_stats = evaluate_snn(Global_node.test_data, Global_node.snn_model, device, args.test_T, args)
            if args.wandb == 1:
                # Assuming evaluate_snn returns something useful or logs internally, 
                # but if it returns stats, we can log them.
                # Looking at utils_ann2snn.py, it prints but returns (loss, acc) or similar?
                # The provided read of FedNeurX_momentum.py shows: test_stats = evaluate_snn(...)
                # Let's assume it returns a dict or tuple.
                pass

    Summary(args)
