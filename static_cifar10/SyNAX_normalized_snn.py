
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
import sys
import types
from pathlib import Path
try:
    import medmnist  # noqa: F401
except ModuleNotFoundError:
    medmnist_stub = types.ModuleType('medmnist')
    medmnist_stub.INFO = {}
    sys.modules['medmnist'] = medmnist_stub
from tqdm import tqdm
import math

from Data import Data
from models.dyrep import DyRep, build_optimizer, get_params
from Node.Node import Global_Node, Node
from Trainer import Trainer
from utils.utils import (LR_scheduler, Recorder, Summary, get_log_file_name,
                         init_args, print_memory_usage, generate_node_list, initialize_device_types, set_random_seed, ann2snn)
from utils.wandb_utils import configure_wandb
from utils_ann2snn import evaluate_snn
from utils.misc import AverageMeter


# Custom ANN Training with Gradient Correction
def train_synax_ANN(node, epoch, round, g_t, beta=1, gamma=1.0):
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
    # To be consistent with the formula: g_correct = beta * grad + (1-beta) * gamma * g_t
    
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
                        # g_{k,t}^{(j)} <- beta * grad + (1-beta) * gamma * g_t
                        # Note: param.grad is updated in place
                        param.grad.data = beta * param.grad.data + (1 - beta) * gamma * g_t_device[name]
            
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
        
        # Normalize once across all trainable parameters. State buffers such as
        # BatchNorm running statistics must not change the update direction.
        parameter_names = set(dict(self.model.named_parameters()))
        valid_g = {
            k: v for k, v in self.g_t.items()
            if k in parameter_names and v.is_floating_point()
        }
        if valid_g:
            g_t_norm = torch.sqrt(sum(
                torch.sum(v.float() ** 2) for v in valid_g.values()
            ))
        else:
            g_t_norm = torch.tensor(0.0, device=next(self.model.parameters()).device)
        correction_step = eta_snn * abs(float(avg_alpha)) if g_t_norm > 1e-12 else 0.0
        print(
            f"Global g_t norm: {g_t_norm.item():.6f}, "
            f"normalized correction step: {correction_step:.6f}"
        )
        
        # 1. Update ANN
        for k in new_ann_state:
            is_float = new_ann_state[k].is_floating_point()
            # Ensure update_term is float for calculation to avoid type errors with LongTensors (e.g. num_batches_tracked)
            update_term = torch.zeros_like(new_ann_state[k], dtype=torch.float)
            
            if k in avg_ann_update:
                update_term += eta_ann * avg_ann_update[k]
            
            if k in valid_g and g_t_norm > 1e-12 and avg_alpha != 0:
                update_term -= eta_snn * avg_alpha * (valid_g[k] / g_t_norm)
                
            if is_float:
                new_ann_state[k] += update_term
            else:
                # Cast to float for update then back to original type (e.g. Long)
                new_ann_state[k] = (new_ann_state[k].float() + update_term).long()
            
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
if 'vit' in args.model.lower():
    args.type = 'VIT'
    args.shape = 224
else:
    args.type = 'CNN'
    if args.dataset.lower() == 'cifar10':
        args.shape = 32
    else:
        args.shape = 224

wandb = configure_wandb(args, run_name_prefix="SyNAX")

Data = Data(args)
recorder = Recorder(args)
file_name  = get_log_file_name(args, directory = "logs/SyNAX")

# Keep this runner independent from the original SyNAX implementation and
# persist the best ANN state for a matched post-hoc SNN conversion.
run_name = os.environ.get(
    "SYNAX_RUN_NAME",
    f"synax_normalized_snn_{args.dataset}_{args.model}_n{args.node_num}_seed{args.seed}",
)
checkpoint_dir = Path("checkpoints") / run_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_acc = -1.0
best_checkpoint_path = checkpoint_dir / "best_ann.pt"
print(f"SyNAX SNN-eval checkpoint: {best_checkpoint_path}")

# init nodes
snn_ratio = args.snn_ratio  # SNN 设备比例
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
eta_snn = float(os.environ.get("SYNAX_ETA_SNN", "0.1"))
print(f"Normalized SyNAX executor eta_snn: {eta_snn}")

# train
for rounds in range(args.R): 
    Summary(args)
    print(f'=============== SyNAX Round {rounds + 1}, device: {device} ===============')
    args.lr = LR_scheduler(lr_initial, rounds, args.R)
    
    # 1. Device Sampling
    sample_size = int(args.node_num * args.sample_ratio)
    sampled_indices = random.sample(range(args.node_num), sample_size)
    if ann_devices and sample_size > 0 and not any(idx in ann_devices for idx in sampled_indices):
        sampled_indices[random.randrange(sample_size)] = random.choice(ann_devices)
    sampled_nodes = [Edge_nodes[idx] for idx in sampled_indices]
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
        
        # Calculate gamma for gradient scaling
        # gamma = 1 / (K * eta_l)
        # K = total_steps = E * batches_per_epoch
        batches_per_epoch = len(node.train_data)
        K = args.E * batches_per_epoch
        gamma = 1.0 / (K * args.lr) if (K * args.lr) != 0 else 1.0

        for epoch in range(args.E):
            train_synax_ANN(node, epoch, rounds, Global_node.g_t, beta=synax_beta, gamma=gamma)
            
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
        if getattr(args, 'disable_snn_feedback', False):
            print(f"Node {node.num} SNN feedback skipped for ablation")
            continue
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
    if getattr(args, 'disable_snn_feedback', False):
        print(f"SNN feedback disabled for ablation: alpha {avg_alpha:.4f} -> 0.0000")
        avg_alpha = 0.0
    print(f"📢 平均 Alpha: {original_alpha:.4f} -> Clamped: {avg_alpha:.4f}")
    print(f"📢 阈值: {args.zero_min}, {args.zero_max}")
    
    if wandb is not None:
        wandb.log({f"{args.dataset}_Alpha": avg_alpha}, step=rounds)

    print(f"Aggregated: ANN Norm: {sum(torch.norm(v) for v in avg_ann_update.values()):.4f}, SNN Alpha: {avg_alpha:.6f}")
    
    # Synthesize & Update Guidance
    Global_node.update_global_models(avg_ann_update, avg_alpha, eta_ann=eta_ann, eta_snn=eta_snn)
    
    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name=file_name, rounds=rounds)

    current_ann_acc = float(recorder.val_acc[str(Global_node.num)][-1])
    if current_ann_acc > best_checkpoint_acc:
        best_checkpoint_acc = current_ann_acc
        cpu_state = {
            key: value.detach().cpu().clone()
            for key, value in Global_node.model.state_dict().items()
        }
        torch.save(
            {
                "round": rounds + 1,
                "args": vars(args),
                "ann_model": cpu_state,
                "best_ann": best_checkpoint_acc,
                "eta_snn": eta_snn,
                "normalized_global_l2": True,
            },
            best_checkpoint_path,
        )
        print(
            f"Saved best SyNAX ANN checkpoint: round={rounds + 1}, "
            f"acc={best_checkpoint_acc:.2f}%, path={best_checkpoint_path}"
        )
    
    # Periodic SNN Evaluation (every 5 rounds or last round)
    if (rounds + 1) % 20 == 0 or (rounds + 1) == args.R:
        print(f"--- Evaluating SNN at round {rounds + 1} ---")
        test_stats = evaluate_snn(Global_node.test_data, Global_node.snn_model, device, args.test_T, args)

Summary(args)
print(
    f"Finished normalized SyNAX checkpoint run. best_ann={best_checkpoint_acc:.2f}% "
    f"checkpoint={best_checkpoint_path}"
)
