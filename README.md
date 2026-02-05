# SyNAX
**Brain-inspired Symbiotic Learning Unlocks Dynamic On-Chip Neuromorphic Intelligence (Under Review)**

![image](Framework.png)

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

```python

python test_memory_usage.py
```

---

## **4. Contact**

For any questions or issues, please open an issue on the GitHub repository.
