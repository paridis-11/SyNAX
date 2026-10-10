# SyNAX
**Brain-inspired Symbiotic Learning Unlocks Dynamic On-Chip Neuromorphic Intelligence (Under Review)**

![image](Framework.png)

## Software Experiment Release

The current CIFAR-10 ResNet/ViT and Figure 2 event-stream software experiments
are in [`software_experiments/`](software_experiments/README.md). That directory
contains their source dependencies, environment instructions, launch commands,
checkpoint evaluator, and the Figure 2j GPU memory benchmark. Follow its README
for the packaged experiments; the older instructions below describe legacy
entry points and may not apply to this release.

## **1. Data**
This project uses three publicly available datasets. Please follow the steps below to download and prepare the datasets:

1. Create a folder named `data` in the root directory of this project. 

2. Extract the downloaded dataset into the `data/covid` directory.  
   - After extraction, the folder structure should look like:  
     ```
     data/
       cifar10/
       imagenet/
       dvs-cifar-10/
       n-caltech101/
     ```

---


## 2. Install Dependencies 
To set up the environment, ensure you have Python and Anaconda installed, then install the required libraries using the `requirements.txt` file, 12.1 cuda environment is recommended:

```bash
conda create -n SyNAX python=3.9
conda activate SyNAX
pip install -r requirements.txt
pip install --upgrade wandb protobuf

```

## **3. Usage**

**3.1 Main Results**

```python

python SyNAX.py --dataset cifar10 --node_num 50 --model Vit_small --R 500 --E 5 --partition iid  --batchsize 32 --test_T 8
```

**3.2 Energy Consumption**

```python

python energy_probe.py
```

**3.3 Memory Usage**

```bash
cd software_experiments
GPU=0 bash scripts/run_gpu_memory.sh
```

This uses `static_cifar10/benchmark_gpu_memory.py` for the standardized
SG-BPTT/SyNAX executor memory comparison. The obsolete memory diagnostic has
been removed.

---

## **4. Contact**

For any questions or issues, please open an issue on the GitHub repository.
