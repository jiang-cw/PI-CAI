import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim

from dataset import *
from utils import *


def trainer_picai(args, model, snapshot_path):
    # Logging setup
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Configurations
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations

    # Dataset setup
    db_train = train_dataset(data_dir=args.train_path)
    db_test = test_dataset(data_dir=args.test_path)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, pin_memory=True)

    # Model setup
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.eval()

    # Loss and optimizer
    loss_func = BCEDiceLoss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=base_lr, weight_decay=0.001)

    # TensorBoard logging
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    # Metrics
    iter_num = 0
    best_dice = 0.0
    loss_list, dice_score_lst = [], []
    mean_loss_list, mean_dice_list = [], []
    performance_list = []

    for epoch_num in tqdm(range(args.max_epochs), ncols=70):
        model.train()
        epoch_num += 1

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].reshape(-1, 3, 128, 128).cuda()
            label_batch = sampled_batch['label'].reshape(-1, 128, 128).unsqueeze(1).cuda()
            # print(image_batch.shape)
            # print(image_batch.shape)

            outputs = model(image_batch, train=True)[0]
            bceloss, d_loss = loss_func(outputs, label_batch)
            loss = bceloss + d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Learning rate schedule
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            loss_list.append(loss.item())
            dice_score_lst.append(dice_coeff(outputs, label_batch).item())

        mean_loss = np.mean(loss_list)
        mean_dice = np.mean(dice_score_lst)
        mean_loss_list.append(mean_loss)
        mean_dice_list.append(mean_dice)

        # Validation
        if (epoch_num + 1) % args.eval_interval == 0 or (epoch_num + 1) == args.max_epochs:
            model.eval()
            metrics = {
                'dice': 0.0,
                'hd95': 0.0,
                'iou': 0.0,
                'asd': 0.0,
                'acc': 0.0,
                'recall': 0.0
            }

            for data, label, meta in tqdm(testloader):
                data = Variable(data).cuda()
                label = Variable(label).cuda()
         
                image = data.squeeze(0).permute(3, 0, 1, 2)
                label = label.squeeze(0).permute(3, 0, 1, 2)
                case_name = meta[-1][0].strip().split('/')[-1]

                dice_score, hd95, iou, asd, acc, recall, *_ = test_single_volume(
                    image, label, model,
                    classes=args.num_classes,
                    patch_size=[args.img_size, args.img_size],
                    test_save_path=args.test_save_path,
                    case=case_name,
                    img_information=meta,
                    epoch=epoch_num
                )

                metrics['dice'] += float(dice_score[0])  
                metrics['hd95'] += float(hd95)
                metrics['iou'] += float(iou)
                metrics['asd'] += float(asd)
                metrics['acc'] += float(acc)
                metrics['recall'] += float(recall)

            # Average metrics
            num_cases = len(testloader)
            for k in metrics:
                metrics[k] /= num_cases

            performance = metrics['dice']
            performance_list.append(performance)

            logging.info(
                f"epoch: {epoch_num} || best_dice: {best_dice:.4f} || Testing performance: "
                f"dice: {metrics['dice']:.4f}, hd95: {metrics['hd95']:.4f}, iou: {metrics['iou']:.4f}, "
                f"asd: {metrics['asd']:.4f}, acc: {metrics['acc']:.4f}, recall: {metrics['recall']:.4f}"
            )

            with open(snapshot_path + 'loss_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"=== epoch_{epoch_num} ===\n")
                f.write(f"best_dice: {best_dice:.3f} || performance: {performance:.3f}\n")

            if performance >= best_dice:
                best_dice = performance
                model_save_path = os.path.join(snapshot_path, f"best_epoch_{epoch_num}_{best_dice:.3f}.pth")
                torch.save(model.state_dict(), model_save_path)
                with open(os.path.join(snapshot_path, 'save_log.txt'), 'a') as f:
                    f.write(model_save_path + '\n')

    writer.close()
    return mean_loss_list, performance_list

