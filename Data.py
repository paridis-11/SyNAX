import os.path
from random import shuffle

import medmnist
import numpy as np
import wandb
import matplotlib.pyplot as plt
import torch
from medmnist import INFO
from PIL import Image
from scipy.stats import dirichlet
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, utils
from sklearn.model_selection import train_test_split
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

def Dataset(args):
    trainset, testset = None, None

    if args.dataset == 'cifar10':
        args.num_classes = 10
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.RandomCrop(args.shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=tra_trans)
        testset = CIFAR10(root="./data", train=False, download=True, transform=val_trans)
    


    elif args.dataset.lower() == 'cancerslides':
        args.num_classes = 9
        info = INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.RandomCrop(args.shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        trainset = DataClass(root="./data", split='train', transform=tra_trans, download=True, size = 224)
        testset = DataClass(root="./data", split='test', transform=val_trans, download=True, size = 224)


    elif args.dataset.lower() == 'chestxray':
        args.num_classes = 2
        info = INFO['chestmnist']
        DataClass = getattr(medmnist, info['python_class'])
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.RandomCrop(args.shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        trainset = DataClass(root="./data", split='train', transform=tra_trans, download=True, size = 224)
        testset = DataClass(root="./data", split='test', transform=val_trans, download=True, size = 224)

    elif args.dataset.lower() == 'covid19':  # 私人数据集
        args.num_classes = 4  # COVID, lung, normal, viral Pneumonia 四个类别
        
        # 数据增强和预处理
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.RandomCrop(args.shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 加载数据，只针对 `images` 文件夹
        dataset = datasets.ImageFolder(
            root="./data/COVID-19",  # 主数据目录
            transform=tra_trans  # 初始使用训练数据的变换
        )

        # 获取所有样本的索引
        indices = list(range(len(dataset)))

        # 使用 sklearn 的 train_test_split 划分训练集和测试集
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, stratify=[dataset.targets[i] for i in indices], random_state=42
        )

        # 创建 Subset 数据集
        trainset = Subset(dataset, train_indices)
        testset = Subset(dataset, test_indices)

        # 替换测试集的 transform 为验证集预处理
        testset.dataset.transform = val_trans


    elif args.dataset.lower() == 'bloodcell':
        args.num_classes = 8
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.RandomCrop(args.shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        trainset = DataClass(root="./data", split='train', transform=tra_trans, download=True, size = 224)
        testset = DataClass(root="./data", split='test', transform=val_trans, download=True, size = 224)


    elif args.dataset.lower() == 'tiny-imagenet':
        args.num_classes = 200

        data_dir = './data/tiny-imagenet-200'
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val', 'images')

        # Define transforms
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.RandomCrop(args.shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                (0.2770, 0.2691, 0.2821))
        ])
        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                (0.2770, 0.2691, 0.2821))
        ])

        # Load training data
        trainset = datasets.ImageFolder(train_dir, transform=tra_trans)

        # Load validation data
        # Tiny ImageNet validation set has a separate annotation file for labels
        from collections import defaultdict
        val_annotations_file = os.path.join(data_dir, 'val', 'val_annotations.txt')
        val_img_to_class = {}
        with open(val_annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                val_img_to_class[parts[0]] = parts[1]

        # Map class names to index
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(train_dir)))}

        # Build validation dataset manually
        val_data = []
        val_targets = []
        for img_name, cls_name in val_img_to_class.items():
            img_path = os.path.join(val_dir, img_name)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = val_trans(img)
                val_data.append(img)
                val_targets.append(class_to_idx[cls_name])

        val_data_tensor = torch.stack(val_data)
        val_targets_tensor = torch.tensor(val_targets)

        # Custom Dataset for validation
        class TinyImageNetValDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        testset = TinyImageNetValDataset(val_data_tensor, val_targets_tensor)

    elif args.dataset.lower() == 'cifar10dvs':
        from spikingjelly.datasets import cifar10_dvs
        print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
        print('resource, url, md5/n', CIFAR10DVS.resource_url_md5())
        # help(CIFAR10DVS)
        print(cifar10_dvs.__file__)
        args.num_classes = 10
        os.makedirs('./data/cifar10_dvs', exist_ok=True)
        tra_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 可选
        ])

        val_trans = transforms.Compose([
            transforms.Resize((args.shape, args.shape)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 可选
        ])

        trainset = CIFAR10DVS(
            root='./data/cifar10_dvs/train',
            data_type='frame',
            frames_number=args.frames,
            split_by='number',
            transform=tra_trans
        )

        testset = CIFAR10DVS(
            root='./data/cifar10_dvs/test',
            data_type='frame',
            frames_number=args.frames,
            split_by='number',
            transform=val_trans
        )


    else:
        raise RuntimeError(f'Dataset {args.dataset} not found.')

    return trainset, testset


def visualize_data_distribution(trainset, splited_trainset, args):
    targets = np.array(trainset.targets)
    num_classes = len(np.unique(targets))
    
    # 统计每个客户端的标签分布
    client_distribution = []
    client_data_counts = []
    for i, subset in enumerate(splited_trainset):
        subset_targets = np.array([trainset.targets[idx] for idx in subset.indices])
        client_distribution.append(np.bincount(subset_targets, minlength=num_classes))
        client_data_counts.append(len(subset))

    # 将每个客户端的数据样本数记录到 W&B
    for i, count in enumerate(client_data_counts):
        wandb.log({f'client_{i+1}_data_count': count})

    # 绘制每个客户端的标签分布
    fig, axs = plt.subplots(1, args.node_num, figsize=(15, 5), sharey=True)
    for i, dist in enumerate(client_distribution):
        axs[i].bar(range(num_classes), dist)
        axs[i].set_title(f'Client {i+1}')
        axs[i].set_xlabel('Label')
        axs[i].set_ylabel('Count')

    plt.tight_layout()

    # 将图像上传到 W&B
    wandb.log({"data_distribution": wandb.Image(fig)})

    plt.close(fig)  # 关闭图像，释放内存


class Data(object):

    def __init__(self, args):
        self.args = args
        self.trainset, self.testset, self.test_all = None, None, None
        trainset, testset = Dataset(args)

        if self.args.partition == 'iid':
            total_length = len(trainset)
            num_train = [total_length // args.node_num] * (args.node_num - 1) + [total_length - (total_length // args.node_num) * (args.node_num - 1)]
            splited_trainset = random_split(trainset, num_train, generator=torch.Generator().manual_seed(42))
            # self.test_all = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=4)



        elif args.partition == 'dir':
            if args.dataset.lower() == 'cifar10':
                targets = np.array(trainset.targets)
            else:
                targets = np.array(trainset.labels)
            num_classes = len(np.unique(targets))
            num_data_per_client = [len(trainset) // self.args.node_num] * self.args.node_num
            num_data_per_client[-1] += len(trainset) - sum(num_data_per_client)

            label_proportions = dirichlet([self.args.dir] * num_classes).rvs(size=self.args.node_num)

            client_indices = []
            for i in range(self.args.node_num):
                proportions = label_proportions[i]
                label_counts = [int(p * num_data_per_client[i]) for p in proportions]
                label_counts[-1] += num_data_per_client[i] - sum(label_counts)
                indices = []
                for label, count in enumerate(label_counts):
                    label_indices = np.where(targets == label)[0]
                    np.random.shuffle(label_indices)
                    indices.extend(label_indices[:count])
                np.random.shuffle(indices)
                client_indices.append(indices)

            splited_trainset = [Subset(trainset, client_indices[i]) for i in range(self.args.node_num)]



        # self.train_loader = [DataLoader(splited_trainset[i], batch_size=args.batchsize, shuffle=True, num_workers=10)
        #                     for i in range(args.node_num)]
        
        self.train_loader = [DataLoader(splited_trainset[i], batch_size=args.batchsize, shuffle=True, num_workers=10)
                            for i in range(args.node_num)]
        
        self.test_loader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=2)
        self.test_all = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=2)
        # visualize_data_distribution(trainset, splited_trainset, args)


# , pin_memory=True, prefetch_factor=2, persistent_workers=True