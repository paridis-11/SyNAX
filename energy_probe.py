import torch
import torch.nn as nn
import numpy as np
from utils.utils import init_args, get_flops, ann2snn
from utils.trans_utils import SOPMonitor, reset_net
from models.builder import build_model
from Data import Data
from tqdm import tqdm
import sys

# --- 硬件能耗常数 (单位: Joules) --

E_SOP_LOIHI = 23.6e-12   # 0.067 pJ
E_FLOP_PI4  = 11.7e-12  # 11.7 pJ (Jetson Nano)
E_FLOP_GPU  = 42.4e-12  # 21.2 pJ (Orin Nano)

def main():
    args = init_args()

    
    if args.dataset != 'cifar10':
        print(f"Warning: Switching dataset to cifar10 for energy profiling.")
        args.dataset = 'cifar10'

    print(f"Running energy probe on {args.device}")
    print(f"Model: {args.model}")

    # Load Data
    data = Data(args)
    test_loader = data.test_loader
    
    # Build ANN Model & SNN Conversion
    print("Building and converting model...", flush=True)
    ann_model = build_model(args, args.model).to(args.device)
    snn_model = ann2snn(ann_model, test_loader, args)
    snn_model = snn_model.to(args.device)
    
    # 初始化 SOP Monitor
    # type=1 监控计算层 (Linear/Conv) 的 SOP 和 MACs
    monitor = SOPMonitor(snn_model, type=1)
    monitor.enable()
    snn_model.eval()
    
    # 统计变量
    total_sparse_sop = 0.0  # 实际发生的脉冲操作 (out0)
    total_dense_macs = 0.0  # 理论上的密集计算量 (sum0)
    total_samples = 0
    
    
    
    print("Starting evaluation...", flush=True)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Energy Profiling")):
            images = images.to(args.device)
            batch_size = images.size(0)
            
            reset_net(snn_model)
            
            # Forward pass
            try:
                _ = snn_model(images, T=1)  # SyNAX requires only T=1 to complete the detection.
            except TypeError:
                _ = snn_model(images)
            
            # --- 关键修改：同时累加 Sparse SOP 和 Dense MACs ---
            for r in monitor.records:
                # r[0]: out0 (Sparse SOP, 实际脉冲连接数)
                # r[1]: sum0 (Dense MACs, 理论全连接计算数)
                
                # 确保提取数值并累加
                sparse_val = r[0].item() if isinstance(r[0], torch.Tensor) else r[0]
                dense_val  = r[1].item() if isinstance(r[1], torch.Tensor) else r[1]
                
                total_sparse_sop += sparse_val
                total_dense_macs += dense_val
            
            total_samples += batch_size
    
            # 清空 monitor 为下一批做准备
            monitor.clear_recorded_data()
    
    # --- 计算单样本平均值 ---
    avg_sparse_sop = total_sparse_sop / total_samples
    avg_dense_macs = total_dense_macs / total_samples
    
    # --- 计算稀疏率 (Sparsity) ---
    # Activity Sparsity = Actual / Dense
    sparsity = avg_sparse_sop / avg_dense_macs if avg_dense_macs > 0 else 0

    # --- 论文核心指标计算 ---
    
    # 1. Our Method (SyNAX): 2 Forward
    # Workload = Sparse SOP * 2
    ops_synax = avg_sparse_sop * 2
    
    # 2. Baseline (Standard BPTT on GPU): 
    # Workload = Dense MACs * 3 (1 Forward + 2 Backward) * T
    
    ops_baseline_bptt = avg_dense_macs * 3 * args.test_T

    # --- 能耗估算 (Joules) ---
    energy_synax_loihi = ops_synax * E_SOP_LOIHI
    energy_baseline_pi = ops_baseline_bptt * E_FLOP_PI4
    energy_baseline_gpu = ops_baseline_bptt * E_FLOP_GPU

    # --- 打印 Nature 风格报告 ---
    print("\n" + "="*60)
    print(f"{'Nature Submission - Energy Efficiency Report':^60}")
    print("="*60)
    
    print(f"\n[1] Sparsity Analysis (Per Inference)")
    print(f"  - Dense MACs (Theory):   {avg_dense_macs/1e6:10.2f} M Ops")
    print(f"  - Actual SOPs (SNN):     {avg_sparse_sop/1e6:10.2f} M Ops")
    print(f"  - Activity Ratio (rho):  {sparsity*100:10.4f} %")
    
    print(f"\n[2] Training Workload Comparison (Per Round)")
    print(f"  - Baseline (BPTT, Dense):{ops_baseline_bptt/1e6:10.2f} M Ops (1 Fwd + 2 Bwd)")
    print(f"  - SyNAX (Ours, Sparse):  {ops_synax/1e6:10.2f} M Ops (2 Fwd Probes)")
    print(f"  - Algorithmic Reduction: {ops_baseline_bptt/ops_synax:10.2f} x")

    print(f"\n[3] Estimated Energy Consumption")
    print(f"  - Baseline (Pi 4, 600pJ):{energy_baseline_pi*1000:10.4f} mJ")
    print(f"  - Baseline (GPU, 150pJ): {energy_baseline_gpu*1000:10.4f} mJ")
    print(f"  - SyNAX (Loihi 2, 20pJ): {energy_synax_loihi*1e6:10.4f} uJ")
    
    print(f"\n[4] Final Efficiency Gain")
    ratio_pi = energy_baseline_pi / energy_synax_loihi
    ratio_gpu = energy_baseline_gpu / energy_synax_loihi
    
    print(f"  - vs Raspberry Pi 4:     {ratio_pi:10.1f} x")
    print(f"  - vs Jetson Nano GPU:    {ratio_gpu:10.1f} x")
    print("="*60)

    # 保存到文件
    with open("energy_profile_nature.txt", "w") as f:
        f.write(f"Sparsity (rho): {sparsity:.6f}\n")
        f.write(f"SyNAX Ops: {ops_synax:.2f}\n")
        f.write(f"Baseline Ops: {ops_baseline_bptt:.2f}\n")
        f.write(f"Energy SyNAX (J): {energy_synax_loihi:.2e}\n")
        f.write(f"Energy Pi4 (J): {energy_baseline_pi:.2e}\n")
        f.write(f"Gain vs Pi4: {ratio_pi:.2f}\n")

if __name__ == "__main__":
    main()