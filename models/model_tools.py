import os
import gc
import time
import torch
import torchvision
import numpy as np
from typing import Any, Callable, Optional, Tuple


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        return '\t'.join(entries)

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion, args, verbose=True, end_batch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if verbose and i % args.print_freq == 0:
                print(progress.display(i))

            if end_batch is not None and i >= end_batch:
                # logger.info(progress.display(i))
                break

            # # ----------------------
            # # this is used to extract loss value, Todo delete this after extract loss
            # print(' * Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'
            #       .format(top1=top1, top5=top5))
            # return top1.avg, top5.avg, losses.avg
            # # ----------------------

        # TODO: this should also be done with the ProgressMeter
        # logger.info(progress.display(i))
        print(f' * Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%')
    # model.module.show_params()

    return top1.avg, top5.avg, losses.avg


def train(train_loader, model, criterion, optimizer, epoch, args, writer=None, end_batch=None, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % args.print_freq == 0:
            print(progress.display(i))
        gc.collect()

        if end_batch is not None and i >= end_batch:
            # logger.info(progress.display(i))
            break
    # writer.add_scalar('train_acc', top1.avg, epoch)
    # logger.info(progress.display(i))


def load_dataset(root: str,
                 transform: Optional[Callable] = None,
                 batch_size: int = 32,
                 workers: int = 4):
    dataset = os.path.basename(root)
    assert dataset in ["cifar10", "cifar100", "imagenet2012"]

    train_dataset, test_dataset = None, None

    if dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    if dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    if dataset == "imagenet2012":
        train_dataset = torchvision.datasets.ImageNet(root=root, split='train', transform=transform)
        test_dataset = torchvision.datasets.ImageNet(root=root, split='val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, pin_memory=True,
                                               num_workers=workers, batch_size=batch_size, persistent_workers=True, )
    val_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, pin_memory=True,
                                             num_workers=workers, batch_size=batch_size, persistent_workers=True, )
    return train_loader, val_loader


def sub_dataset(dataset: [torchvision.datasets.ImageFolder, None] = None, ratio: float = 0.1):
    assert dataset is not None, 'invalid dataset'
    dataset_new = dataset
    sub_dataset_id = []
    sub_class_id = []
    class_id = 0
    for i in range(len(dataset.targets) - 1):
        sub_class_id.append(i)
        if (dataset.targets[i] != dataset.targets[i + 1]) or i + 2 == len(dataset.targets):
            if i + 2 == len(dataset.targets):
                sub_class_id.append(i + 1)
            sub_class_id = np.array(sub_class_id)
            total_samples = len(sub_class_id)
            sub_samples = int(total_samples * ratio) + 1
            sub_samples = sub_samples if sub_samples <= total_samples else total_samples
            idx = np.random.choice(total_samples, sub_samples, replace=False)
            sub_dataset_id = sub_dataset_id + sub_class_id[idx].tolist()
            class_id = class_id + 1
            sub_class_id = []
        print(f'\r[{i}/{len(dataset.targets)}]\t{i / len(dataset.targets):.2%}', end='', flush=True)
    print("sub_dataset, done!")
    dataset_new.imgs = [dataset.imgs[i] for i in sub_dataset_id]
    dataset_new.samples = [dataset.samples[i] for i in sub_dataset_id]
    dataset_new.targets = [dataset.targets[i] for i in sub_dataset_id]
    return dataset_new

