import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='dyfl_vit',
                        help='Type of algorithms:{fed_mutual, fed_avg, fed_coteaching, normal, parallel}')
    parser.add_argument('--wandb', default=1, type=int)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--node_num', type=int, default=9,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=500,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=5,
                        help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes of Experiments')
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
    parser.add_argument('--shape', type=int, default=224,
                    help='random seed (default: 224, 32)')
    parser.add_argument('--device_ratio', type=str, default="1:1:1",
                    help='device_ratio')
    parser.add_argument('--mode', type=str, default="FedNeurx",
                    help='ann_only, FedNeurx')

    # Model
    parser.add_argument('--model', type=str, default='vit_small')
    # parser.add_argument('--local_model', type=str, default='nas_model')
    parser.add_argument('--model-config', type=str, default='vgg16_cifar10.yaml',
                    help='Path to net config.')
    parser.add_argument('--drop', type=float, default=0.0,
                    help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path-rate', type=float, default=0., 
                    help='Drop path rate, (default: 0.)')
    parser.add_argument('--zo_epsilon', type=float, default=0.01,
                    help='zero')
    parser.add_argument('--weight-decay', default=0.001, type=float)
    parser.add_argument('--monitor', default=True, type=bool)

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='datasets: {cifar10, CancerSlides, mnist}')
    parser.add_argument('--partition', type=str, default='iid',
                        help='partition : {iid, dir}')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='batchsize')
    parser.add_argument('--dir', type=float, default=0.5,
                        help='Degree of dirichlet')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val_ratio')
    parser.add_argument('--all_data', type=bool, default=True,
                        help='use all train_set')
    parser.add_argument('--classes', type=int, default=10,
                        help='classes')
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")
    # parser.add_argument('--sampler', type=str, default='iid', help="iid, non-iid")

    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimizer: {sgd, adam, adamw}')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='local ratio of data loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='global ratio of data loss')
    parser.add_argument('--opt-eps', default=1e-8, type=float,
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')

    parser.add_argument('--dyrep-recal-bn-iters', type=int, default=20,
                    help='how many iterations for recalibrating the bn states in dyrep')

    parser.add_argument('--opt_no_filter', action='store_true', default=True,
                    help='disable bias and bn filter of weight decay')

    parser.add_argument('--sample_ratio', type=float, default=0.2, help="每轮客户端采样比例 (0~1)")
    
    # Zero  parameter
    parser.add_argument('--zero_max', default=0.5,type=float)
    parser.add_argument('--zero_min', default=-0.5,type=float)
    parser.add_argument('--zero_momentum', default=0.2,type=float)
    # SNN Multi-threshold neuron parameter
    parser.add_argument('--linear_num', default=8,type=int)
    parser.add_argument('--qkv_num', default=8,type=int)
    parser.add_argument('--softmax_num', default=8,type=int)
    parser.add_argument('--softmax_p', default=0.0125,type=float)
    parser.add_argument('--test_T', default=8,type=int)
    parser.add_argument('--frames', type=int, default=8, help='Number of frames per DVS sample')

    args = parser.parse_args()
    return args
