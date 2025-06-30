import os
import torch
import argparse
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

from framework.model import UNet
from dataset import test_dataset
from utils import test_single_volume


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='result_test/log/best_epoch_100_0.022.pth', help='model path')
    parser.add_argument('--test_path', type=str, default='dataset/test/', help='Path to test data')
    parser.add_argument('--save_path', type=str, default='result/test_eval/', help='Where to save results')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of segmentation classes')
    parser.add_argument('--img_size', type=int, default=256)
    return parser.parse_args()


def test(args):
    # Load model
    model = UNet().cuda()

    state_dict = torch.load(args.model_path, map_location='cuda')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # model.load_state_dict(torch.load(args.model_path))
    model.eval()

    os.makedirs(args.save_path, exist_ok=True)

    # Load test dataset
    test_set = test_dataset(data_dir=args.test_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Metrics accumulator
    metrics = {
        'dice': 0.0,
        'hd95': 0.0,
        'iou': 0.0,
        'asd': 0.0,
        'acc': 0.0,
        'recall': 0.0
    }

    with torch.no_grad():
        for data, label, meta in tqdm(test_loader, desc="Testing", ncols=80):
            data = Variable(data).cuda()
            label = Variable(label).cuda()

            image = data.squeeze(0).permute(3, 0, 1, 2)
            label = label.squeeze(0).permute(3, 0, 1, 2)
            case_name = meta[-1][0].strip().split('/')[-1]

            dice_score, hd95, iou, asd, acc, recall, *_ = test_single_volume(
                image, label, model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=args.save_path,
                case=case_name,
                img_information=meta,
                epoch='test'
            )

            metrics['dice'] += float(dice_score[0])
            metrics['hd95'] += float(hd95)
            metrics['iou'] += float(iou)
            metrics['asd'] += float(asd)
            metrics['acc'] += float(acc)
            metrics['recall'] += float(recall)

    # Compute average
    num_cases = len(test_loader)
    for k in metrics:
        metrics[k] /= num_cases

    # Print and save
    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(args.save_path, "summary.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


if __name__ == '__main__':
    args = get_args()
    test(args)
