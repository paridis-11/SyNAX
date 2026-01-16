import numpy as np
import torch
from collections import defaultdict, deque
import datetime
import time
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
# from torchvision.transforms import functional as F
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder,CIFAR10,CIFAR100
from timm.data import create_transform



@torch.no_grad()
def evaluate_snn(data_loader, model, device,T,args):
    metric_logger = MetricLogger(delimiter="  ")
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test SNN :'

    # switch to evaluation mode
    model.eval()
    tot = np.array([0. for i in range(T)])
    length = 0
    nownum = 0
    all_sops = [0 for i in range(T)]
    all_nums = [0 for i in range(T)]
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        with torch.cuda.amp.autocast():
            output = model(images,T=T)
        #output is a list of results from time 1 to T
        acc1_list = []
        for i in range(T):
            acc1,acc5 = accuracy(output[i], target, topk=(1, 5))
            acc1_list.append(float(acc1))
        output=output[-1]
        output/=T
        # reset potential of neuron
        reset_net(model)
        # loss = criterion(output, target)
        length += batch_size
        nownum += 1
        # tot records the correct num
        tot += np.array(acc1_list) * batch_size
        # metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return tot/length



def reset_net(model):#initialize all neurons
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'neuron' in module.__class__.__name__.lower():
            module.reset()
    return model




class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    def add_meter(self, name, meter):
        self.meters[name] = meter
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    @property
    def global_avg(self):
        return self.total / self.count
    @property
    def max(self):
        return max(self.deque)
    @property
    def value(self):
        return self.deque[-1]
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
    


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



def replace_test_by_testneuron(model,percent=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_test_by_testneuron(module,percent)
        if module.__class__.__name__.lower() == 'mytestplace':
            model._modules[name] = TestNeuron(place=model._modules[name].place,percent=percent)
    return model



# class TestNeuron(nn.Module):
#     def __init__(self,place=None,percent=None):
#         super(TestNeuron, self).__init__()
#         self.place = place
#         self.percent = percent
#         self.num = 0
#         self.scale_p = torch.nn.Parameter(torch.FloatTensor([0.]))
#         self.scale_n = torch.nn.Parameter(torch.FloatTensor([0.]))
#     #choose threshold
#     def forward(self, x, times=2,gap=3,show=0, tmptime=0,scaletimes=1):
#         x2 = x.reshape(-1)
#         threshold = torch.kthvalue(x2, int(self.percent * x2.numel()), dim=0).values.item()
#         self.scale_p = torch.nn.Parameter((self.scale_p*self.num+threshold)/(self.num+1))
#         threshold = -torch.kthvalue(x2, int((1-self.percent) * x2.numel()), dim=0).values.item()
#         self.scale_n = torch.nn.Parameter((self.scale_n*self.num+threshold)/(self.num+1))
#         self.num+=1
#         return [x,]
#     def reset(self):
#         pass



class TestNeuron(nn.Module):
    def __init__(self,place=None,percent=None):
        super(TestNeuron, self).__init__()
        self.place = place
        self.percent = percent
        self.num = 0
        self.scale_p = torch.nn.Parameter(torch.FloatTensor([0.]))
        self.scale_n = torch.nn.Parameter(torch.FloatTensor([0.]))
    #choose threshold
    def forward(self, x, times=2,gap=3,show=0, tmptime=0,scaletimes=1):
        x2 = x.flatten().float()
        threshold = torch.quantile(x2, self.percent, dim=0).item()
        self.scale_p = torch.nn.Parameter((self.scale_p*self.num+threshold)/(self.num+1))
        threshold = -torch.quantile(x2, (1-self.percent), dim=0).item()
        self.scale_n = torch.nn.Parameter((self.scale_n*self.num+threshold)/(self.num+1))
        self.num+=1
        return [x,]
    def reset(self):
        pass

# class TestNeuron(nn.Module):
#     def __init__(self, place=None, percent=None):
#         super(TestNeuron, self).__init__()
#         self.place = place
#         self.percent = percent
#         self.num = 0
#         self.scale_p = torch.tensor(0.0)  # 直接用 tensor，避免多余的 Parameter
#         self.scale_n = torch.tensor(0.0)

#     def forward(self, x, update_interval=10):
#         x2 = x.flatten().float()   # 替换 reshape(-1) 为 flatten()，减少内存开销

#         if self.num % update_interval == 0:  # 控制更新频率
#             threshold_p = torch.quantile(x2, self.percent)  # 替代 kthvalue，更高效
#             threshold_n = -torch.quantile(x2, 1 - self.percent)
            
#             # 避免使用 nn.Parameter，改用 .data 直接更新 tensor
#             self.scale_p.data = (self.scale_p * self.num + threshold_p) / (self.num + 1)
#             self.scale_n.data = (self.scale_n * self.num + threshold_n) / (self.num + 1)

#         self.num += 1
#         return [x,]

#     def reset(self):
#         pass





@torch.no_grad()
def evaluate(data_loader, model, device):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    header = 'Test ANN :'
    model.eval()
    nownum=0

    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        nownum += 1

        with torch.cuda.amp.autocast():
            output = model(images)[0]
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        with open('log_ann.txt','a') as f:
            f.write(str(nownum)+': '+'* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss)+'\n')

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    nownum = 0

    for batch in data_loader: 
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        nownum += 1

        with torch.cuda.amp.autocast():
            output = model(images)[0]
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 更新指标
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    print(f"* Acc@1 {metric_logger.acc1.global_avg:.3f} "
          f"Acc@5 {metric_logger.acc5.global_avg:.3f} "
          f"loss {metric_logger.loss.global_avg:.3f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

def build_dataset(is_train, args,root_path='.'):
    transform = build_transform(is_train, args)
    if args.dataset == 'cifar10':
        dataset_val = CIFAR10(root="./data", train=False ,transform=transform, download=True)
        if is_train==True:
            dataset_train = CIFAR10(root="./data", train=True ,transform=transform, download=True)
    elif args.dataset == 'CIFAR100':
        dataset_val = CIFAR100(root="./data", train=False ,transform=transform, download=True)
        if is_train==True:
            dataset_train = CIFAR100(root="./data", train=True ,transform=transform, download=True)
    elif args.dataset == "image_folder":
        dataset_val = ImageFolder("./data", transform=transform)
    else:
        raise NotImplementedError()
    # print("Number of the class = %d" % args.nb_classes)
    if is_train==True:
        return dataset_train, dataset_val, 10
    else:
        return dataset_val, 10


def build_transform(is_train, args):
    # imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([transforms.Resize(224, interpolation=3),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean, std)])


def save_model(args, model, model_without_ddp):
    output_dir = Path(args.output_dir)
    checkpoint_paths = [output_dir / (args.savename+'.pth')]
    for checkpoint_path in checkpoint_paths:
        to_save = {'model': model_without_ddp.state_dict(),'args': args,}
        save_on_master(to_save, checkpoint_path)