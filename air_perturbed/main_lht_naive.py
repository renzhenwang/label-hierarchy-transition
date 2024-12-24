# -*- coding: utf-8 -*-
import argparse
import os
import sys
import random
import shutil
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.air_dataset import AirDataset
from models.resnet_transition_naive import ResNetTransitionNaive
from losses.hierarchy_loss import Hierarchy_Loss, Hierarchy_KL_Loss
from utils import get_confusion_matrix, Logger

sys.path.insert(0, '.')


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch LHT Deployment')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--ratio', type=float, help='Proportion of species label'),
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--loss_type', type=str, default='hierarchy',
                        choices=['hierarchy', 'hierarchy_kl', 'hierarchy_smooth'],
                        help='loss func type (default: hierarchy)')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='confusion loss coefficient')

    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=10, help='manual seed')

    parser.add_argument('--snapshot', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')

    parser.add_argument('--crop_size', dest='crop_size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest='scale_size', default=232, type=int,
                        help='the size of the rescale image')
    parser.add_argument('--relabel', dest='relabel', type=str, default='family', choices=['family', 'order'],
                        metavar='RELABEL', help='relabeled hierarchy under semi-supervised setting')
    parser.add_argument('--title', type=str, default='transition model for aircraft recognition!')

    args = parser.parse_args()
    return args

args = arg_parse()
if args.seed is None:
    args.seed = random.randint(1, 10000)
else:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def main():

    print_args(args)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    # Recoder the running processing
    sys.stdout = Logger(
        os.path.join(args.out, 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))

    # Create dataloader
    print("==> Creating dataloader...")
    data_dir = args.data
    test_loader = get_test_set(os.path.join(data_dir, 'test/'), args)
    train_loader = get_train_set(os.path.join(data_dir, 'trainval/'), args)

    # load the network
    num_classes = 100
    num_families = 70
    num_orders = 30

    label_hierarchy = [num_orders, num_families, num_classes]

    print("==> Loading the network ...")
    model = ResNetTransitionNaive(label_hierarchy, pretrained=True, fine_to_coarse=True).cuda()

    backbone_params = list(map(id, model.extractor.parameters()))
    classifier_params = filter(lambda p: id(p) not in backbone_params,
                               model.parameters())

    optimizer = torch.optim.SGD([
        {'params': classifier_params},
        {'params': model.extractor.parameters(), 'lr': args.lr * 0.1}], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    def cosine_anneal_schedule(t):
        cos_inner = np.pi * (t % (args.epochs))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= (args.epochs)
        cos_out = np.cos(cos_inner) + 1
        return float(1.0 / 2 * cos_out)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_anneal_schedule)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    if args.loss_type == 'hierarchy_kl':
        criterion = Hierarchy_KL_Loss(beta=args.beta)
    else:
        criterion = Hierarchy_Loss(beta=-1*args.beta)

    if args.snapshot:
        if os.path.isfile(args.snapshot):
            print("=> loading checkpoint '{}'".format(args.snapshot))
            checkpoint = torch.load(args.snapshot)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.snapshot))
            print("Testing...")
            with torch.no_grad():
                validate(test_loader, model, args)
            return
        else:
            print("=> no checkpoint found at '{}'".format(args.snapshot))
            exit()

    start_epoch = 0
    max_val_acc = 0
    max_val_epoch = 0
    max_result = {}
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        # Training part
        train_loss, train_acc = train(train_loader, model, optimizer, criterion, epoch, args)
        scheduler.step()
        print('epoch: ', epoch, 'lr: ', scheduler.get_last_lr())
        print("Testing...")
        with torch.no_grad():
            top1 = validate(test_loader, model, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mcr': max_val_acc,
            'optimizer': optimizer.state_dict(),
        }, top1[-1] > max_val_acc, args.out, epoch=epoch)

        if top1[-1] > max_val_acc:
            max_val_acc = top1[-1]
            max_val_epoch = epoch
            max_result = top1
        print("max_val_acc == {} @ Epoch == {}".format(max_val_acc, max_val_epoch))
    print_args(args)
    print("Best result: ", max_result)


def train(train_loader, model, optimizer, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        order_targets, family_targets, species_targets = targets[:, 0], targets[:, 1], targets[:, 2]
        target_list = [order_targets.long(), family_targets.long(), species_targets.long()]

        # compute output
        outputs, t_mats = model(inputs, target_list)
        loss = criterion(outputs, target_list, t_mats)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        losses.update(loss.item(), inputs.size(0))
        select_mask = species_targets > -1
        select_species_targets = species_targets[select_mask]
        select_species_outputs = outputs[-1][select_mask]
        # print(len(select_species_outputs), len(select_species_targets))
        if select_mask.sum() > 0:
            prec1, prec5 = accuracy(select_species_outputs, select_species_targets, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * loss {losses.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(losses=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    top1_L1 = AverageMeter()
    top5_L1 = AverageMeter()
    top1_L2 = AverageMeter()
    top5_L2 = AverageMeter()
    top1_L3 = AverageMeter()
    top5_L3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        order_targets, family_targets, species_targets = targets[:, 0], targets[:, 1], targets[:, 2]
        target_list = [order_targets.long(), family_targets.long(), species_targets.long()]

        # compute output
        (pred1_L1, pred1_L2, pred1_L3), _ = model(inputs)

        # measure accuracy
        prec1_L1, prec5_L1 = accuracy(pred1_L1.data, order_targets, topk=(1, 5))
        prec1_L2, prec5_L2 = accuracy(pred1_L2.data, family_targets, topk=(1, 5))
        prec1_L3, prec5_L3 = accuracy(pred1_L3.data, species_targets, topk=(1, 5))

        top1_L1.update(prec1_L1.item(), inputs.size(0))
        top5_L1.update(prec5_L1.item(), inputs.size(0))
        top1_L2.update(prec1_L2.item(), inputs.size(0))
        top5_L2.update(prec5_L2.item(), inputs.size(0))
        top1_L3.update(prec1_L3.item(), inputs.size(0))
        top5_L3.update(prec5_L3.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print(' * L1(order): \tPrec@1 {top1_L1.avg:.3f} Prec@5 {top5_L1.avg:.3f}'
          .format(top1_L1=top1_L1, top5_L1=top5_L1))
    print(' * L2(family): \tPrec@1 {top1_L2.avg:.3f} Prec@5 {top5_L2.avg:.3f}'
          .format(top1_L2=top1_L2, top5_L2=top5_L2))
    print(' * L3(species): \tPrec@1 {top1_L3.avg:.3f} Prec@5 {top5_L3.avg:.3f}'
          .format(top1_L3=top1_L3, top5_L3=top5_L3))

    return top1_L1.avg, top1_L2.avg, top1_L3.avg


def linear_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def get_train_set(data_dir, args):
    # Data loading code
    print('Training data loading!')
    # normalize for different pretrain model:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size

    # center crop
    train_data_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomCrop(crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.crop_size == 448:
        train_data_transform = transforms.Compose([
            transforms.Resize((scale_size, scale_size)),
            transforms.RandomCrop(crop_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_data = AirDataset(data_dir, input_transform=train_data_transform, semi_level=True, ratio=args.ratio,
                            relabel=args.relabel)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                             drop_last=True, pin_memory=True)

    return train_loader


def get_test_set(data_dir, args):
    # Data loading code
    print('Testing data loading!')
    # normalize for different pretrain model:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    # center crop
    test_data_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize,
    ])
    if args.crop_size == 448:
        test_data_transform = transforms.Compose([
            transforms.Resize((scale_size, scale_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    test_data = AirDataset(data_dir, input_transform=test_data_transform, semi_level=False, ratio=1.0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False, pin_memory=True)

    return test_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch=None):
    filename = os.path.join(checkpoint, filename)
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == "__main__":
    main()

