
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
                         init_args, initialize_device_types, set_random_seed, AverageMeter)
from utils_ann2snn import evaluate_snn
from utils.utils import ann2snn, compute_snn_difference

# =============================================================================
# SyNAX Training Functions
# =============================================================================

def train_synax_ANN(node, epoch, round, global_grad_vector, beta=0.9):
    """
    SyNAX ANN Training with Gradient Correction:
    g_{k,t}^{(j)} = beta * \nabla F_k + (1 - beta) * g_t
    """
    node.model.to(node.device).train()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()
    
    loss_m = AverageMeter()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    
    # Ensure global_grad_vector is on the correct device or move it lazily
    # To avoid moving it every iteration, we can do it once if memory allows, 
    # but since it's same size as model, it's fine.
    
    with tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) as epochs:
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
            output = output[0] # Handle model wrapper
            loss = loss_fn(output, target)
            loss.backward()

            # --- Gradient Correction ---
            if global_grad_vector is not None:
                with torch.no_grad():
                    for name, param in node.model.named_parameters():
                        if param.grad is not None and name in global_grad_vector:
                            # g_t is on global device usually, move to local if needed
                            g_t = global_grad_vector[name].to(node.device)
                            # Update gradient in place
                            param.grad.data.mul_(beta).add_(g_t, alpha=(1 - beta))
            # ---------------------------

            node.optimizer.step()
            
            loss_m.update(loss.item(), n=data.size(0))
            total_loss += loss.item() # Fix: accumulate item
            avg_loss = total_loss / (batch_idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(train_loader.dataset) * 100

    return avg_loss

def train_synax_SNN_inference(node, perturbed_model):
    """
    Executes inference (forward pass) on the perturbed SNN model to compute loss.
    """
    perturbed_model.to(node.device)
    perturbed_model.eval()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        # Reduce tqdm frequency or disable for performance
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(node.device, non_blocking=True)
            target = target.to(node.device, non_blocking=True)
            
            output = perturbed_model(data)[0]
            loss = loss_fn(output, target)
            
            total_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
            
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

# =============================================================================
# SyNAX Node Classes
# =============================================================================

class SyNAX_Node(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train_ann_synax(self, epochs, global_grad_vector, beta):
        for epoch in range(epochs):
            train_synax_ANN(self, epoch, 0, global_grad_vector, beta)
            
    def compute_snn_feedback(self, global_snn_model, guidance_vector_G, epsilon):
        """
        Computes scalar feedback alpha = (L+ - L-) / (2 * epsilon)
        """
        # Normalize guidance
        # Note: G_t is a dict of tensors
        norm_sq = sum(torch.sum(g ** 2) for g in guidance_vector_G.values())
        norm = torch.sqrt(norm_sq)
        
        hat_G = {k: v / (norm + 1e-8) for k, v in guidance_vector_G.items()}
        
        # Prepare perturbed models
        # We don't need to deepcopy the whole model structure if we can just load state dict,
        # but modifying weights is easier with a copy or loading.
        # To save memory, we can modify the model in-place and then revert.
        
        original_state_dict = {k: v.clone() for k, v in global_snn_model.state_dict().items()}
        
        # L+ : w + epsilon * hat_G
        for name, param in global_snn_model.named_parameters():
            if name in hat_G:
                param.data.add_(hat_G[name].to(param.device), alpha=epsilon)
                
        loss_plus = train_synax_SNN_inference(self, global_snn_model)
        
        # L- : w - epsilon * hat_G (so subtract 2*epsilon from current +epsilon state)
        for name, param in global_snn_model.named_parameters():
            if name in hat_G:
                param.data.sub_(hat_G[name].to(param.device), alpha=2*epsilon)
                
        loss_minus = train_synax_SNN_inference(self, global_snn_model)
        
        # Revert to original (optional if we download fresh model next round, but good practice)
        # Or just: w - epsilon * hat_G + epsilon * hat_G = w
        for name, param in global_snn_model.named_parameters():
            if name in hat_G:
                param.data.add_(hat_G[name].to(param.device), alpha=epsilon)
                
        alpha = (loss_plus - loss_minus) / (2 * epsilon)
        return alpha

class SyNAX_Global_Node(Global_Node):
    def __init__(self, test_data, args):
        super().__init__(test_data, args)
        self.g_t = None # Global ANN gradient direction
        self.G_t = None # Global SNN guidance direction
        self.w_t_ANN = None
        self.w_t_SNN = None # State dict
        
    def init_sy_nax_vars(self):
        # Initialize g_0 = 0, G_0 = 0
        self.w_t_ANN = copy.deepcopy(self.model.state_dict())
        self.w_t_SNN = copy.deepcopy(self.snn_model.state_dict())
        
        self.g_t = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        self.G_t = {k: torch.zeros_like(v) for k, v in self.snn_model.state_dict().items()}
        
    def update_global_models(self, ann_updates, avg_alpha, eta_ann, eta_snn):
        """
        w_{t+1}^{ANN} = w_t^{ANN} + eta_ann * Delta_w_A - eta_snn * alpha * g_t
        """
        w_next_ANN = {}
        
        # Synthesize new ANN model
        for key in self.w_t_ANN:
            update_term = torch.zeros_like(self.w_t_ANN[key])
            
            # Add ANN updates
            if key in ann_updates:
                update_term.add_(ann_updates[key], alpha=eta_ann)
            
            # Subtract SNN feedback (using previous g_t)
            if key in self.g_t:
                # g_t might be on different device
                g_t_val = self.g_t[key].to(update_term.device)
                update_term.sub_(g_t_val, alpha=eta_snn * avg_alpha)
                
            w_next_ANN[key] = self.w_t_ANN[key] + update_term
            
        # Load new weights to ANN model
        self.model.load_state_dict(w_next_ANN)
        
        # Update Guidance for next round
        # w_{t+1}^{SNN} = Convert(w_{t+1}^{ANN})
        # Note: ann2snn might need the model to be on a specific device
        self.snn_model = ann2snn(self.model, self.test_data, args=self.args)
        w_next_SNN = self.snn_model.state_dict()
        
        # g_{t+1} = w_t^{ANN} - w_{t+1}^{ANN}
        new_g_t = {}
        for key in self.w_t_ANN:
            new_g_t[key] = self.w_t_ANN[key].float() - w_next_ANN[key].float()
        self.g_t = new_g_t
        
        # G_{t+1} = w_t^{SNN} - w_{t+1}^{SNN}
        new_G_t = {}
        # SNN keys might differ or be subset? usually ann2snn keeps structure but converts layers
        # compute_snn_difference handles finding common keys
        for key in self.w_t_SNN:
            if key in w_next_SNN:
                new_G_t[key] = self.w_t_SNN[key].float() - w_next_SNN[key].float()
            else:
                new_G_t[key] = torch.zeros_like(self.w_t_SNN[key])
                
        self.G_t = new_G_t
        
        # Update stored state for next iteration
        self.w_t_ANN = copy.deepcopy(w_next_ANN)
        self.w_t_SNN = copy.deepcopy(w_next_SNN)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    args = init_args()
    set_random_seed(args.seed)
    
    # Force some args if needed for SyNAX
    # args.type = 'VIT'
    # args.shape = 224
    
    if args.wandb == 1:
        import wandb
        run_name = f"SyNAX_{args.dataset}_N{args.node_num}_E{args.E}_lr{args.lr}_{args.notes}"
        wandb.init(project="DyFL", name=run_name, entity="paridis")
        wandb.config.update(vars(args))

    data_loader = Data(args)
    # trainer = Trainer(args) # Not used directly, using custom train functions
    recorder = Recorder(args)
    file_name = get_log_file_name(args, directory="logs/SyNAX")
    
    # Init Nodes
    snn_ratio = 0.5
    ann_devices, snn_devices = initialize_device_types(args.node_num, snn_ratio)
    
    global_node = SyNAX_Global_Node(data_loader.test_all, args)
    global_node.init_sy_nax_vars()
    
    # Create Clients
    # Note: reusing Node class structure but we will use new methods
    clients = []
    for k in range(args.node_num):
        client = SyNAX_Node(
            k,
            data_loader.train_loader[k],
            data_loader.test_loader,
            args,
            device_type="SNN" if k in snn_devices else "ANN"
        )
        clients.append(client)

    # Hyperparameters
    # eta_ann, eta_snn from args or default
    eta_ann = 1.0 # Global learning rate for ANN updates
    eta_snn = args.lr * 1.0 # Learning rate for Symbiotic updates
    beta = 0.9 # Momentum factor
    epsilon = args.zo_epsilon # Probe radius
    
    # Ensure cifar10 if not specified (though init_args handles defaults)
    if not args.dataset:
        args.dataset = 'cifar10'
    
    print(f"Starting SyNAX Training: {args.R} rounds, {args.node_num} nodes")
    
    for round_idx in range(args.R):
        Summary(args)
        print(f'=============== Round {round_idx + 1} ===============')
        
        # 1. Select Clients
        sample_size = int(args.node_num * args.client_sample_ratio)
        if sample_size < 1: sample_size = 1
        active_clients = random.sample(clients, sample_size)
        
        ann_clients = [c for c in active_clients if c.device_type == "ANN"]
        snn_clients = [c for c in active_clients if c.device_type == "SNN"]
        
        print(f"Active: {len(ann_clients)} ANN, {len(snn_clients)} SNN")
        
        # 2. Parallel Client Execution
        
        # --- ANN Clients (Orchestrators) ---
        ann_updates_list = []
        
        # Broadcast g_t (implicit by passing it)
        # We need to simulate the delta update.
        # Delta_w = w_final - w_initial
        # But wait, w_initial at client should be w_global.
        # In the paper pseudocode: w_{k,t}^(0) = w_{k,t-1}^(0) - g_t.
        # As discussed, we approximate this by sending current global model w_t^{ANN}
        
        for client in ann_clients:
            print(f"  [ANN] Client {client.num} training...")
            # Load global model
            client.model.load_state_dict(global_node.model.state_dict())
            client.optimizer = torch.optim.SGD(client.model.parameters(), lr=args.lr, momentum=args.momentum)
            
            # Initial weights
            w_initial = {k: v.clone() for k, v in client.model.state_dict().items()}
            
            # Train with gradient correction
            client.train_ann_synax(args.E, global_node.g_t, beta)
            
            # Compute Update: Delta = w_final - w_initial
            w_final = client.model.state_dict()
            delta_w = {}
            for k in w_initial:
                delta_w[k] = w_final[k] - w_initial[k]
                
            ann_updates_list.append(delta_w)
            client.delete_model()
            
        # --- SNN Clients (Executors) ---
        snn_alphas = []
        
        for client in snn_clients:
            print(f"  [SNN] Client {client.num} probing...")
            # Load global SNN model
            client.model = copy.deepcopy(global_node.snn_model) # Use deepcopy to avoid modifying global
            
            # Compute scalar feedback
            # G_t is passed
            alpha = client.compute_snn_feedback(client.model, global_node.G_t, epsilon)
            snn_alphas.append(alpha)
            
            print(f"    -> Alpha: {alpha:.6f}")
            client.delete_model()

        # 3. Server Aggregation & Symbiotic Update
        
        # Aggregate ANN updates
        avg_ann_update = {}
        if ann_updates_list:
            for k in ann_updates_list[0].keys():
                avg_ann_update[k] = sum(update[k] for update in ann_updates_list) / len(ann_updates_list)
        
        # Aggregate SNN feedback
        avg_alpha = sum(snn_alphas) / len(snn_alphas) if snn_alphas else 0.0
        print(f"  Aggregated ANN updates from {len(ann_updates_list)} clients")
        print(f"  Aggregated SNN Alpha: {avg_alpha:.6f}")
        
        if args.wandb == 1:
            wandb.log({"avg_alpha": avg_alpha}, step=round_idx)

        # Synthesize new global model
        # If no ANN clients, we might skip or just use SNN feedback (though algorithm implies symbiotic)
        # If no SNN clients, avg_alpha is 0, so standard FL.
        
        global_node.update_global_models(avg_ann_update, avg_alpha, eta_ann, eta_snn)
        
        # Validation
        recorder.validate(global_node)
        recorder.printer(global_node, file_name=file_name, rounds=round_idx)
        
        # Optional: Test SNN
        # test_stats = evaluate_snn(global_node.test_data, global_node.snn_model, args.device, args.test_T, args)

    Summary(args)

if __name__ == "__main__":
    main()
