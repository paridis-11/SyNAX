import os
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torchvision import transforms
from torch.utils.data import DataLoader

# 设置数据根目录
root_dir = './data/cifar10_dvs'

# 设置下载和预处理路径
download_root = os.path.join(root_dir, 'download')
extract_root = os.path.join(root_dir, 'extract')
npz_root = os.path.join(root_dir, 'npz')

# 创建目录
os.makedirs(download_root, exist_ok=True)
os.makedirs(extract_root, exist_ok=True)
os.makedirs(npz_root, exist_ok=True)

# 1️⃣ 下载每个类别的 zip 文件
for name, url, md5 in CIFAR10DVS.resource_url_md5():
    zip_path = os.path.join(download_root, name)
    if not os.path.exists(zip_path):
        print(f'Downloading {name} from {url}...')
        # 使用 requests 下载
        import requests
        r = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        print(f'Downloaded {name}.')

# 2️⃣ 解压 zip 文件
CIFAR10DVS.extract_downloaded_files(download_root, extract_root)

# 3️⃣ 转换为 `.npz` 格式
CIFAR10DVS.create_events_np_files(extract_root, npz_root)

# 4️⃣ 加载为 SpikingJelly 数据集
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CIFAR10DVS(
    root=npz_root,
    data_type='frame',
    frames_number=10,         # 每个样本的帧数，可调整
    split_by='number',        # 按样本数划分帧
    transform=transform
)

# 5️⃣ 使用 PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 6️⃣ 简单测试
for i, (frames, label) in enumerate(dataloader):
    print(f'Batch {i}: frames.shape={frames.shape}, label={label}')
    break