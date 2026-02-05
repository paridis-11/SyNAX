import torch
import torch.nn as nn
import time
import sys
import os
import gc

# Add current directory to path to handle imports
sys.path.append(os.getcwd())

from models.Model import ResNet18
from models.builder_snn import build_snn_model
from utils.utils import ann2snn
from utils.trans_utils import AtNeuron
from timm.models import create_model
import models.builder_snn 
import models.model_vit

# Patch VisionTransformer in builder_snn to use the one from models.model_vit
# This fixes vit_large/huge which use direct instantiation in builder_snn
# and ensures they use the SNN-compatible implementation (with MyTestPlace, MyAt, T-loop)
class PatchedVisionTransformer(models.model_vit.VisionTransformer):
    def __init__(self, **kwargs):
        # Map argument names from timm style (used in builder_snn) to model_vit style
        if 'drop_rate' in kwargs:
            kwargs['drop_ratio'] = kwargs.pop('drop_rate')
        if 'attn_drop_rate' in kwargs:
            kwargs['attn_drop_ratio'] = kwargs.pop('attn_drop_rate')
        super().__init__(**kwargs)

# Apply patch to models.builder_snn
models.builder_snn.VisionTransformer = PatchedVisionTransformer

# Define safe forward for AtNeuron to avoid inplace errors during BPTT
def safe_at_neuron_forward(self, x, y):
    if self.t == 0:
        self.tot_a=x
        self.tot_b=y
        self.tot_t = x@y
        self.t = 1
        return x@y
    else:
        # Avoid in-place operations for BPTT compatibility
        # Original: self.tot_t+= x@y+x@self.tot_b+self.tot_a@y
        self.tot_t = self.tot_t + (x@y + x@self.tot_b + self.tot_a@y)
        self.tot_a = self.tot_a + x
        self.tot_b = self.tot_b + y
        self.t += 1
        return (x@self.tot_b+self.tot_a@y-x@y)/(self.t-1)-self.tot_t/(self.t*(self.t-1))

# Define Surrogate Gradient components locally to avoid modifying codebase
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Surrogate gradient: rectangle function
        mask = (input.abs() < 0.5).float() 
        return grad_input * mask

class SurrogateNeuron(nn.Module):
    def __init__(self, scale_p=1., scale_n=1., place=None, times=3, gap=2):
        super(SurrogateNeuron, self).__init__()
        self.mem = 0.0
        self.v_th = 0.5
        self.decay = 0.5

    def forward(self, x):
        # Clone input to avoid any potential inplace modification issues with autograd
        x = x.clone()
        if isinstance(self.mem, float):
            self.mem = torch.zeros_like(x)
        # Avoid in-place modification for autograd compatibility
        self.mem = self.mem * self.decay + x
        spike = SurrogateHeaviside.apply(self.mem - self.v_th)
        self.mem = self.mem * (1 - spike)
        # Return list to match ResNet/SyNAX structure expectation (sum(list))
        return [spike] 
    
    def reset(self):
        self.mem = 0.0

def replace_with_surrogate(model):
    from utils.trans_utils import TwoSideNeuron
    for name, module in model.named_children():
        if isinstance(module, TwoSideNeuron):
            setattr(model, name, SurrogateNeuron())
        else:
            replace_with_surrogate(module)
    return model

def get_memory_mb(device=None):
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)

def reset_memory(device=None):
    if device:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

class MockArgs:
    def __init__(self, model_name='resnet18'):
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'cifar10'
        self.model = model_name
        self.num_classes = 10
        self.linear_num = 2
        self.qkv_num = 2
        self.softmax_num = 2
        self.softmax_p = 0.1
        self.zero_momentum = 0.9
        self.seed = 42
        # Determine shape based on model
        if 'vit' in model_name.lower():
            self.shape = 224
            self.batchsize = 4 # Reduced batchsize for ViT to avoid OOM
        else:
            self.shape = 32
            self.batchsize = 4
        self.test_T = 8
        self.device_ratio = "1:1:1"
        self.node_num = 10
        self.zo_epsilon = 0.01

def run_test_for_model(model_name):
    print(f"\n{'='*20} Testing Model: {model_name} {'='*20}")
    args = MockArgs(model_name)
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)
        
    print(f"Device: {args.device}, Input Shape: {args.shape}x{args.shape}, Batch Size: {args.batchsize}, T: {args.test_T}")

    # Prepare Data
    data = torch.randn(args.batchsize, 3, args.shape, args.shape).to(args.device)
    target = torch.randint(0, 10, (args.batchsize,)).to(args.device)
    
    class DummyLoader:
        def __init__(self, data, target):
            self.data = data
            self.target = target
        def __iter__(self):
            yield self.data, self.target
        def __len__(self):
            return 1
    test_loader = DummyLoader(data, target)

    # 1. ANN Backpropagation Training
    print("\n--- [Test 1] ANN Backpropagation Training ---")
    reset_memory(args.device)
    
    ann_model = None
    optimizer = None
    try:
        if 'resnet' in model_name:
            ann_model = ResNet18(num_classes=10).to(args.device)
        else:
            ann_model = build_snn_model(args).to(args.device)
            
        # Calculate and print model size
        param_size = sum(p.numel() for p in ann_model.parameters()) * 4 / (1024 ** 2)
        print(f"Model Size: {param_size:.2f} MB")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(ann_model.parameters(), lr=0.01)
        ann_model.train()
        
        # Reset memory stats to track PEAK usage during training step
        # We want to include model weights + gradients + activations
        # So we reset stats, but we assume model is already loaded.
        # However, torch.cuda.max_memory_allocated returns the peak since reset.
        # If we reset here, we might miss the model weight memory if it was allocated before.
        # But we want the TOTAL peak memory footprint.
        # memory_allocated() gives current usage.
        # reset_peak_memory_stats() sets peak = current.
        # So max_memory_allocated will be max(current_start, peak_during_op).
        # This is exactly what we want.
        reset_memory(args.device)
        start_mem = torch.cuda.memory_allocated(args.device) / 1024**2
        start_time = time.time()
        
        output = ann_model(data)
        if isinstance(output, list):
            output = output[0]
            
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        peak_mem = get_memory_mb(args.device)
        print(f"Time: {end_time - start_time:.4f} s")
        print(f"Peak Memory: {peak_mem:.2f} MB")
        
    except Exception as e:
        print(f"ANN Test Failed: {e}")
    finally:
        del ann_model, optimizer
        if 'loss' in locals(): del loss
        if 'output' in locals(): del output
        reset_memory(args.device)

    # 2. SNN Surrogate Gradient Training (BPTT)
    print("\n--- [Test 2] SNN Surrogate Gradient Training (BPTT) ---")
    reset_memory(args.device)
    
    # Enable anomaly detection to debug inplace errors
    torch.autograd.set_detect_anomaly(True)
    
    snn_model_sg = None
    optimizer_sg = None
    
    # Patch AtNeuron for ViT BPTT to avoid inplace errors
    original_at_forward = AtNeuron.forward
    if 'vit' in model_name.lower():
        AtNeuron.forward = safe_at_neuron_forward
        
    try:
        base_model = build_snn_model(args).to(args.device)
        snn_model_sg = ann2snn(base_model, test_loader, args)
        del base_model
        
        snn_model_sg = replace_with_surrogate(snn_model_sg)
        snn_model_sg = snn_model_sg.to(args.device)
        
        optimizer_sg = torch.optim.Adam(snn_model_sg.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        snn_model_sg.train()
        
        reset_memory(args.device)
        start_mem = torch.cuda.memory_allocated(args.device) / 1024**2
        start_time = time.time()
        
        output_sg = snn_model_sg(data, T=args.test_T)
        
        if isinstance(output_sg, list):
            final_output = output_sg[-1] / args.test_T
        else:
            final_output = output_sg
            
        loss_sg = criterion(final_output, target)
        loss_sg.backward()
        optimizer_sg.step()
        
        end_time = time.time()
        peak_mem = get_memory_mb(args.device)
        print(f"Time: {end_time - start_time:.4f} s")
        print(f"Peak Memory: {peak_mem:.2f} MB")
        
    except Exception as e:
        print(f"SNN SG Test Failed: {e}")
    finally:
        if snn_model_sg is not None: del snn_model_sg
        if optimizer_sg is not None: del optimizer_sg
        if 'loss_sg' in locals(): del loss_sg
        if 'output_sg' in locals(): del output_sg
        
        # Restore original forward
        if 'vit' in model_name.lower():
            AtNeuron.forward = original_at_forward
            
        reset_memory(args.device)

    # 3. SyNAX Client Training (Forward Only)
    print("\n--- [Test 3] SyNAX Client Training (Forward Only) ---")
    reset_memory(args.device)
    
    snn_model_synax = None
    try:
        base_model = build_snn_model(args).to(args.device)
        snn_model_synax = ann2snn(base_model, test_loader, args)
        del base_model
        
        snn_model_synax.eval() 
        
        reset_memory(args.device)
        start_time = time.time()
        
        with torch.no_grad():
            output_1 = snn_model_synax(data, T=args.test_T)
            if isinstance(output_1, list):
                out1 = output_1[-1]
            
            output_2 = snn_model_synax(data, T=args.test_T)
            if isinstance(output_2, list):
                out2 = output_2[-1]
                
            loss_1 = criterion(out1/args.test_T, target)
            loss_2 = criterion(out2/args.test_T, target)
            
            g_t_sim = {}
            for name, param in snn_model_synax.named_parameters():
                g_t_sim[name] = torch.zeros_like(param)
        
        end_time = time.time()
        peak_mem = get_memory_mb(args.device)
        print(f"Time: {end_time - start_time:.4f} s")
        print(f"Peak Memory: {peak_mem:.2f} MB")
        
    except Exception as e:
        print(f"SyNAX Test Failed: {e}")
    finally:
        if snn_model_synax is not None: del snn_model_synax
        if 'g_t_sim' in locals(): del g_t_sim
        reset_memory(args.device)

if __name__ == "__main__":
    for model_name in ['resnet18', 'vit_small', 'vit_base', 'vit_large']:
        run_test_for_model(model_name)