# SyNAX

**Brain-inspired Symbiotic Learning Enables Adaptation on Forward-Only Neuromorphic Hardware (Under Review)**

![image](Framework.png)

## Contents

| Directory | Experiments | Entry points |
| --- | --- | --- |
| `static/` | CIFAR-10 and ImageNet-1K; ResNet-18 and ViT-Small/Base; executor GPU memory | `SyNAX_normalized_snn.py`, `eval_cifar10_snn_checkpoint.py`, `benchmark_gpu_memory.py` |
| `event/` | CIFAR10-DVS and N-Caltech101, ResNet-18 | `synax_neuro.py`, `bp_sg_burst_resnet_event.py`, `bp_sg_sew_resnet_event.py`, `resnet18_event_online_baselines.py` |
| `scripts/` | Portable launch commands and dependency setup | See commands below |

Event-stream baselines include architecture-matched Burst SG-BPTT,
SEW-ResNet SG-BPTT, and local-loss S-TLLR.

## Environment

Use Python 3.10 on Linux and a CUDA-capable GPU. Create an isolated environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/fetch_burst_dependency.py
```

Dependency versions are listed in [environment_observed.json](environment_observed.json).
See [DATA.md](DATA.md) for dataset preparation. Weights & Biases is disabled
by default.

## Static-image Training

```bash
GPU=0 bash scripts/run_static_cifar10.sh resnet18
GPU=0 bash scripts/run_static_cifar10.sh vit_small
GPU=0 bash scripts/run_static_cifar10.sh vit_base
```

The scripts train, then evaluate `best_ann.pt` using the appropriate converter.
ResNet uses Burst ANN2SNN; ViTs use the native project converter.
The default is IID with 50 clients. For a Dirichlet partition, set
`PARTITION=dir DIR_ALPHA=0.5`.

| Setting | ResNet-18 | ViT-Small / ViT-Base |
| --- | ---: | ---: |
| Communication rounds | 100 | 500 |
| Local epochs | 5 | 5 |
| Total clients | 50 | 50 |
| Client participation per round | 0.2 | 0.2 |
| SNN client fraction | 0.5 | 0.5 |
| Training batch size | 32 | 32 |
| ANN optimizer / learning rate | AdamW / 0.001 | AdamW / 0.001 |
| Executor coefficient | 0.1 | 0.1 |
| SNN timesteps | 8 | 8 |

## Event-stream Training

Prepare both frame caches and supply their parent directory through `DATA_ROOT`.

```bash
GPU=0 DATA_ROOT=/path/to/event_data bash scripts/run_event_resnet18.sh synax cifar10dvs
GPU=0 DATA_ROOT=/path/to/event_data bash scripts/run_event_resnet18.sh synax ncaltech101
GPU=0 DATA_ROOT=/path/to/event_data bash scripts/run_event_resnet18.sh burst_sgbptt cifar10dvs
GPU=0 DATA_ROOT=/path/to/event_data bash scripts/run_event_resnet18.sh sew_sgbptt ncaltech101
GPU=0 DATA_ROOT=/path/to/event_data bash scripts/run_event_resnet18.sh stllr_local cifar10dvs
```

Each method supports both datasets. The shared federation protocol is
six clients, full participation, 200 rounds, one local epoch, and SNN `T=8`.
Method-specific optimization and preprocessing settings are defined in
`scripts/run_event_resnet18.sh`.

Event SyNAX uses AdamW with learning rate 0.001 and executor coefficient 0.005.
Burst and SEW SG-BPTT use SGD at 0.05. S-TLLR-local uses Adam at 0.001 with
stage-local classifiers and persistent client optimizer states.

## GPU Memory Benchmark

```bash
GPU=0 bash scripts/run_gpu_memory.sh
```

The benchmark compares one SG-BPTT training step with SyNAX executor-side
two-point forward probing using synthetic inputs. Peak allocated CUDA tensor
memory is measured with `torch.cuda.max_memory_allocated` and reported as
`peak_allocated_gib` in GiB (bytes / 1024^3). Measurement begins after model
conversion and includes resident tensors and allocations during the measured step.

Defaults are ResNet-18 and ViT-Small/Base/Large, batch size 4, SNN `T=8`,
FP32, and three repeats. SG-BPTT uses SGD with learning rate 0.05, momentum
0.9 and weight decay 0.0005. ResNet inputs are 32x32; ViT inputs are 224x224.
An 80 GB GPU is recommended for ViT-Large SG-BPTT.

Raw and summary CSV files and metadata JSON are saved under `results/memory/`.
To select models:

```bash
GPU=0 bash scripts/run_gpu_memory.sh resnet18 vit_small
```

## Third-party Code

ANN-to-SNN conversion uses [Burst ANN2SNN](https://github.com/bic-L/burst-ann2snn).
Run `python scripts/fetch_burst_dependency.py` to install the pinned version
with the supplied compatibility patches.

The [S-TLLR](https://github.com/mapolinario94/S-TLLR) baseline is included under
its MIT license. See [THIRD_PARTY.md](THIRD_PARTY.md) for references,
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) for source hashes, and
[VALIDATION.md](VALIDATION.md) for validation details.

## Contact

For questions or issues, please open an issue on this repository.
