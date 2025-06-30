import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from runner import trainer_picai
from framework.model import UNet

def get_args():
    parser = argparse.ArgumentParser()

    # Paths and Dataset Configs
    parser.add_argument('--train_path', type=str, default="dataset/train/", help='Root directory for training data')
    parser.add_argument('--test_path', type=str, default="dataset/test/", help='Path for validation volume data')
    parser.add_argument('--test_save_path', type=str, default="result_test/output/", help='Directory to save test results')

    # Experiment and Model Settings
    parser.add_argument('--dataset', type=str, default='picai', help='Dataset name')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output channels')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--resize', type=bool, default=False, help='Whether to resize input images')

    # Training Parameters
    parser.add_argument('--max_iterations', type=int, default=300000, help='Maximum number of training iterations')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=20, help='Number of epochs between each evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='Base learning rate')
    parser.add_argument('--n_gpu', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    return parser.parse_args()


def main():
    args = get_args()
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset-specific configuration
    dataset_config = {
        'picai': {
            'train_path': args.train_path,
            'test_path': args.test_path,
            'num_classes': 1,
        },
    }

    dataset_name = args.dataset
    assert dataset_name in dataset_config, f"Dataset {dataset_name} not found in config"

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.train_path = dataset_config[dataset_name]['train_path']
    args.test_path = dataset_config[dataset_name]['test_path']
    args.is_pretrain = False

    # Output snapshot path
    snapshot_path = "result/log/"
    os.makedirs(snapshot_path, exist_ok=True)

    # Initialize model
    net = UNet().cuda()

    # Optionally load pretrained weights
    # load_path = 'path/to/pretrained_model.pth'
    # net.load_state_dict(torch.load(load_path))

    # Start training
    trainer = {'picai': trainer_picai}
    trainer[dataset_name](args, net, snapshot_path)


if __name__ == "__main__":
    main()
