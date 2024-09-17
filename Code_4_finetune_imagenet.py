import gc
import time
import torch.nn as nn
import os
import cgp
from cgp import ConstantTrue, ConstantFalse, Identity, NOT, NOR2, NAND2, XNOR2, OR2, AND2, XOR2
import torch
import pickle
import logging
import numpy as np
from kuai_log import get_logger
import encode_tools as tools
# from encode_tools import gen_uniform_levels_input_encode, gen_input_matrix
from models import NN_LUT
# from encode_tools import convert_searched_results
# import torch.nn.functional as F
# import torchvision

# available models please use torchvision.models.list_models()
# or find on https://pytorch.org/vision/stable/models.html#classification

# from torchvision.models import ResNet50_Weights
# from torchvision.models.quantization import ResNet50_QuantizedWeights
# from torchvision.models.quantization import resnet50 as resnet50_Quantized
# import torch
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
# from pprint import pprint
# pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
import argparse

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

# arch choices for cifar10: resnet18
# arch choices for cifar100: resnet20
# arch choices for imagenet: alexnet, resnet50
parser.add_argument('--bit', type=int, default=8)

parser.add_argument('--baseline', action='store_true')

parser.add_argument('--NUQ-bit', type=int, default=4)
parser.add_argument('--uniform', type=bool, default=True)
parser.add_argument('--target', type=str, default='general', choices=['general', 'specific'])

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument('--product-bit', type=int, default=0)
# parser.add_argument('--target', type=int, default=48)
parser.add_argument('--search', type=int, default=64)
parser.add_argument('--cols', type=int, default=1)
parser.add_argument('--rows', type=int, default=256)
parser.add_argument('--th', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--fixed-type', type=str, default='big', choices=['big', 'small'])
parser.add_argument('--fixed-num', type=int, default=1)
parser.add_argument('--delta', type=int, default=0)

parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--devices', type=str, default='0,1,2,3')
# config

# default
# python3 Code_2_finetune_cifar10.py --retrain --gpu 0 --product-bit 42 --search 64 --cols 1 --rows 256 --th 1.0 --epochs 23 --fixed-type big --fixed-num 0
# python3 Code_2_finetune_cifar10.py --retrain --bit 8 --gpu 0 --product-bit 42 --search 64 --cols 1 --rows 256 --th 1.5 --epochs 23 --fixed-type big --fixed-num 42 --delta 10
args = parser.parse_args()
args.dataset = 'imagenet'
args.arch = 'resnet50'
args.data = '/nas/ei/share/TUEIEDAprojects/NNDatasets/imagenet2012'
# args.running_cache = '/home/ge26rem/lrz-nashome/LRZ/SourceCode/CGP_search/running_cache/'
args.running_cache = './running_cache/'

if not args.uniform:
    assert 2 <= args.NUQ_bit <= 4, 'unsupported bit-width'

log_path = f"retrain-{args.dataset}-{args.arch}-{args.rows}row-{args.cols}col-{args.product_bit}bit-" \
           f"{args.search}b-{args.th}th-{args.fixed_type}fixtype-{args.fixed_num}fixnum-{args.delta}delta.log"
logger = get_logger(name='retrain', level=logging.INFO, log_filename=log_path,
                    log_path='./retrain_logs/', is_add_file_handler=True,
                    formatter_template='{host}-cuda:' + str(args.gpu) + '-{levelname}-{message}'
                    )


def main():
    # global args

    args.workers = 4

    # args.gpu = 1
    # args.bit = 8
    # args.arch = 'alexnet'  # ['resnet50', 'alexnet']
    # args.product_bit = 7
    # [7: 1e-1,  27: 1e-2,  48: 1e-3,  59: 1e-4,  64: 1e-5,  65: 7.6e-7,  66: 6.3e-7,  89: 4e-7]
    args.f_info = './search_results/searched_info-Uniform-{}bit.pickle'.format(args.product_bit)
    args.product_bit = 0 if args.bit == 32 else args.product_bit
    args.print_freq = 1

    if args.product_bit == 0:
        searched_info, args.OP, OP_info, args.model = None, None, None, None
    else:
        searched_info = tools.convert_searched_results(path=args.running_cache, th=args.th,
                                                       target=args.product_bit, search=args.search,
                                                       cols=args.cols, rows=args.rows)
        if args.delta > 0:
            searched_info = tools.apply_delta_for_finetune(searched_info, delta=args.delta)
    args.OP = None
    args.model = None
    args.by_code = True
    if searched_info[0]['bit-width of product'] < args.product_bit:
        if args.fixed_num == args.product_bit:
            args.product_bit = searched_info[0]['bit-width of product']
            args.fixed_num = searched_info[0]['bit-width of product']
    args.approx_product_value, args.approx_product_code, args.digit_weight, args.rmse = get_approx_product(searched_info,
                                                                                                           bit=args.bit,
                                                                                                           product_bit=args.product_bit,
                                                                                                           f_info=args.f_info)

    if args.product_bit == 0:
        idx = None
    else:
        if args.fixed_type == 'big':
            idx = args.digit_weight.abs().argsort(descending=True).view(-1)[0:args.fixed_num].cpu()
        elif args.fixed_type == 'small':
            idx = args.digit_weight.abs().argsort(descending=False).view(-1)[0:args.fixed_num].cpu()
        else:
            raise ValueError('invalid fixed_type')
        if args.fixed_num == args.product_bit:
            args.OP = None
            args.model = None
            args.by_code = False

    if args.product_bit == 0:
        args.digit_weight_mask = None
    else:
        args.digit_weight_mask = torch.ones(args.digit_weight.shape).view(-1).cpu()
        args.digit_weight_mask[idx] = 0.
        args.digit_weight_mask = args.digit_weight_mask.view(args.digit_weight.shape)

    if args.gpu is not None and args.approx_product_value is not None:
        args.approx_product_value = args.approx_product_value.float().cuda(args.gpu)
    if args.gpu is not None and args.approx_product_code is not None:
        args.approx_product_code = args.approx_product_code.to(torch.float16).cuda(args.gpu)
    if args.gpu is not None and args.digit_weight is not None:
        args.digit_weight = args.digit_weight.cuda(args.gpu)
    if args.gpu is not None and args.digit_weight_mask is not None:
        args.digit_weight_mask = args.digit_weight_mask.cuda(args.gpu)
    if args.gpu is not None and args.model is not None:
        args.model = args.model.cuda(args.gpu)

    # if report 'out of memory', please reduce mini_batch_size and mini_channels
    # 0 means full channels and full batch_size
    if args.by_code:
        args.mini_batch_size = 1
        args.mini_channels = 0
    else:
        args.mini_batch_size = 1
        args.mini_channels = 0
    args.batch_size = 32

    if args.mini_batch_size == 0 or args.mini_batch_size > args.batch_size:
        args.mini_batch_size = args.batch_size

    print('-----> build model and load pretrained weights')
    from torchvision.models import resnet50
    if args.arch == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V1')
        resize_size = 256
    else:
        raise ValueError('Todo for other models')

    print('-----> Done! model is built and pretrained weights are loaded!')

    from models.my_quant_layer import QuantFC, QuantConv2d, replace_conv_fc
    print('-------> starting replace original Conv2d layer and Linear layer with proposed QuantConv2d and QuantFC')
    model = replace_conv_fc(model, args=args)
    print('-------> all Conv2d and Linear layer are replaced with proposed QuantConv2d and QuantFC layer!')
    # print(list(model.modules())[0])
    if args.gpu is not None:
        model = model.cuda(args.gpu)

    # data loader by official torchversion:
    # --------------------------------------------------------------------------
    print('==> Using Pytorch Dataset')
    # import torch
    import torchvision
    import torchvision.transforms as transforms
    print('------> loading dataset of ImageNet')
    input_size = 224  # image resolution for resnets
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # torchvision.set_image_backend('accimage')

    def sub_dataset(dataset: [torchvision.datasets.ImageFolder, None] = None, ratio: float = 0.1):
        assert dataset is not None, 'invalid dataset'
        dataset_new = dataset
        sub_dataset_id = []
        sub_class_id = []
        class_id = 0
        for i in range(len(dataset.targets) - 1):
            sub_class_id.append(i)
            if (dataset.targets[i] != dataset.targets[i+1]) or i+2 == len(dataset.targets):
                if i+2 == len(dataset.targets):
                    sub_class_id.append(i+1)
                sub_class_id = np.array(sub_class_id)
                total_samples = len(sub_class_id)
                sub_samples = int(total_samples * ratio) + 1
                sub_samples = sub_samples if sub_samples <= total_samples else total_samples
                idx = np.random.choice(total_samples, sub_samples, replace=False)
                sub_dataset_id = sub_dataset_id + sub_class_id[idx].tolist()
                class_id = class_id + 1
                sub_class_id = []
            print(f'\r[{i}/{len(dataset.targets)}]\t{i/len(dataset.targets):.2%}', end='', flush=True)
        dataset_new.imgs = [dataset.imgs[i] for i in sub_dataset_id]
        dataset_new.samples = [dataset.samples[i] for i in sub_dataset_id]
        dataset_new.targets = [dataset.targets[i] for i in sub_dataset_id]
        return dataset_new

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomCrop(input_size),
            # transforms.RandomHorizontalFlip(),  # Todo
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        persistent_workers=True,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        persistent_workers=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    if args.gpu is not None:
        criterion = criterion.cuda(args.gpu)

    t_conv, t_fc = 0, 0
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            t_conv = t_conv + 1
            # if t_conv == 1 or t_conv == 5:
            #     m.product_bit = 0
        if isinstance(m, QuantFC):
            t_fc = t_fc + 1
            # m.product_bit = 0
    print('total # conv layers:', t_conv, '\ttotal # fc layers:', t_fc)

    if args.test:
        print('-------> testing begins! Good luck to you!\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\tbit: {args.bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        if args.bit == 32:
            # close find scale of x mode
            for m in model.modules():
                if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
                    print('-----> find scale of x: mode closed! for', m.layer_type)
                    m.x_max_init_mode = False
            print('-----------> starting 32bit float inference......')
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            acc1, acc5, loss_avg = validate(train_loader, model, criterion, args)
            print('-----------> 32bit float inference, done!')
            exit()
        else:
            # this validate is used for find scale of x in each layer
            print('-----------> starting find scale of x in each layer, 32bit float inference......')
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            print('-----------> find scale of x in each layer, 32bit float inference, done!')

            # close find scale of x mode
            for m in model.modules():
                if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
                    print('-----> find scale of x: mode closed! for', m.layer_type)
                    m.x_max_init_mode = False

            # this validate is used for real inference
            print('-----------> starting real inference......')
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            acc1, acc5, loss_avg = validate(train_loader, model, criterion, args)
            print('-----------> real inference, done!')
            exit()
        print('-------> testing done! info of this test:\n'
              'arc: {args.arch}\tdataset: {args.dataset}\tbit: {args.bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        contents = 'testing result: \n' \
                   'neural network architecture: {args.arch}\n' \
                   'dataset: {args.dataset}\n' \
                   'target: {args.target}\n' \
                   'uniform: {args.uniform}\n' \
                   'quantization bit: {args.bit}\n' \
                   'product bit: {args.product_bit}\n' \
                   'root mean squre error: {args.rmse:.4e}\n' \
                   'Acc-top1: {top1:.4f}%\n' \
                   'Acc-top5: {top5:.4f}%\n' \
                   'Loss: {loss:.4f}'.format(args=args, top1=acc1, top5=acc5, loss=loss_avg)
        subject = '{args.arch}({args.dataset}) / {args.bit}-bit / {args.product_bit}-bit of product'.format(args=args)
        print(subject)
        print(contents)
    elif args.retrain:
        # this validate is used for find scale of x in each layer
        print('-----------> starting find scale of x in each layer, 32bit float inference......')
        acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        print('-----------> find scale of x in each layer, 32bit float inference, done!')

        # here this code is used to config first_last in specific case
        # t_conv, t_fc
        # t = 0
        # for m in model.modules():
        #     if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
        #         t = t + 1
        #         if t == 1 or t == (t_conv + t_fc):
        #             m.act_quant = quantization(bit=m.bit, signed=True)
        #             m.wgt_quant = quantization(bit=m.bit, signed=True)
        #             m.alpha_act = Parameter(torch.tensor(m.x_max.avg))
        #             m.alpha_wgt = Parameter(m.weight.data.abs().max())

        # close find scale of x mode
        for m in model.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
                print('-----> find scale of x: mode closed! for', m.layer_type)
                m.x_max_init_mode = False

        # which parameters can be learned?
        logger.info('-----> enable or disable learnable parameters')
        if args.uniform:
            model_params = []
            for name, param in model.named_parameters():
                if 'alpha_act' in name:
                    # param.requires_grad = True
                    # model_params += [{'params': [param], 'lr': 1e-5, 'weight_decay': 1e-4}]
                    param.requires_grad = False
                    model_params += [{'params': [param]}]
                elif 'alpha_wgt' in name:
                    # param.requires_grad = True
                    # model_params += [{'params': [param], 'lr': 1e-5, 'weight_decay': 1e-4}]
                    param.requires_grad = False
                    model_params += [{'params': [param]}]
                elif 'digit_weight' in name:
                    # param.requires_grad = True
                    # model_params += [{'params': [param], 'lr': 1e-12, 'weight_decay': 1e-7}]
                    param.requires_grad = False
                    model_params += [{'params': [param]}]
                # elif 'fc' in name and ('weight' in name or 'bias' in name):
                #     param.requires_grad = True
                #     model_params += [{'params': [param], 'lr': 1e-4, 'weight_decay': 1e-4}]
                else:
                    param.requires_grad = True
                    model_params += [{'params': [param]}]
                if param.requires_grad:
                    print('Yes, enable to learn:', name)
                else:
                    print('No, disable to learn:', name)
            optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
            logger.info(f'learning rate = {args.lr}')
        # logger.info(f'the following result of this test is used for testing before retrain')
        # acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        # logger.info(f'the above result of this test is used for testing before retrain')
        # exit()
        logger.info(f'-------> retraining begins! Good luck to you!\n'
                    f'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\tbit: {args.bit}\t'
                    f'product bit: {args.product_bit}\trmse: {args.rmse:.4e}')
        sub_dataset_ratio = 0.01
        logger.info(f'sub_dataset_ratio = {sub_dataset_ratio:.2%}')
        for epoch in range(args.epochs):
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    # transforms.RandomCrop(input_size),
                    # transforms.RandomHorizontalFlip(),  # Todo
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]))

            train_dataset = sub_dataset(train_dataset, ratio=sub_dataset_ratio)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                persistent_workers=True,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            # val_dataset = torchvision.datasets.ImageFolder(
            #     valdir,
            #     transforms.Compose([
            #         transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            #         transforms.CenterCrop(input_size),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     ]))

            # val_dataset = sub_dataset(val_dataset, ratio=sub_dataset_ratio)
            #
            # val_loader = torch.utils.data.DataLoader(
            #     val_dataset,
            #     persistent_workers=True,
            #     batch_size=args.batch_size, shuffle=False,
            #     num_workers=args.workers, pin_memory=True)
            train(train_loader, model, criterion, optimizer, epoch, args, writer=None, end_batch=None)
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)

        # this validate is used for real inference
        print('-----------> starting testing after retraining......\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\tbit: {args.bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        # acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        print('-----------> testing after retraining, done!\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\tbit: {args.bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        contents = 'testing result: \n' \
                   'neural network architecture: {args.arch}\n' \
                   'dataset: {args.dataset}\n' \
                   'target: {args.target}\n' \
                   'uniform: {args.uniform}\n' \
                   'epochs: {args.epochs}\n' \
                   'quantization bit: {args.bit}\n' \
                   'product bit: {args.product_bit}\n' \
                   'root mean squre error: {args.rmse:.4e}\n' \
                   'Acc-top1: {top1:.4f}%\n' \
                   'Acc-top5: {top5:.4f}%\n' \
                   'Loss: {loss:.4f}'.format(args=args, top1=acc1, top5=acc5, loss=loss_avg)
        subject = '{args.arch}({args.dataset}) / {args.bit}-bit / {args.product_bit}-bit of product'.format(args=args)
        print(subject)
        print(contents)
    else:
        raise ValueError('test or retrain? please specify it!')


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
            logger.info(progress.display(i))
            break
    # writer.add_scalar('train_acc', top1.avg, epoch)
    logger.info(progress.display(i))


def validate(val_loader, model, criterion, args, verbose=True):
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

            # # ----------------------
            # # this is used to extract loss value, Todo delete this after extract loss
            # print(' * Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'
            #       .format(top1=top1, top5=top5))
            # return top1.avg, top5.avg, losses.avg
            # # ----------------------

        # TODO: this should also be done with the ProgressMeter
        logger.info(progress.display(i))
        print(f' * Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%')
    # model.module.show_params()

    return top1.avg, top5.avg, losses.avg


def get_approx_product(searched_info, bit=8, product_bit=20, f_info='../backup/searched_info.pickle'):
    def test_fc(approx_product):
        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() & 255 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() & 255 if w_sign else (w_l * 255).int()
            idx = x_c * 256 + w_c
            return approx_product[idx].sum(2)

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        x, w = torch.rand(2, 4) * 2 - 1, torch.rand(3, 4) * 2 - 1
        x, w = x.unsqueeze(1), w.unsqueeze(0)
        fc_real = (x * w).sum(2)
        m_x, m_w = x.abs().max(), w.abs().max()
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        fc_quant = (x_q * w_q).sum(2)
        x_l, w_l = x_q / m_x, w_q / m_w
        fc_approx = p(x_l, w_l, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(fc_real, fc_quant, fc_approx))

    def test_fc_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight):
        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() + 128 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() + 128 if w_sign else (w_l * 255).int()
            # return approx_product_value_2d[x_c, w_c].sum(2)
            return (approx_product_code_2d[x_c, w_c].sum(2) * digit_weight).sum(2)

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        x, w = torch.rand(2, 4) * 2 - 1, torch.rand(3, 4) * 2 - 1
        x, w = x.unsqueeze(1), w.unsqueeze(0)
        fc_real = (x * w).sum(2)
        m_x, m_w = x.abs().max(), w.abs().max()
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        fc_quant = (x_q * w_q).sum(2)
        x_l, w_l = x_q / m_x, w_q / m_w
        fc_approx = p(x_l, w_l, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(fc_real, fc_quant, fc_approx))

    def test_conv(approx_product, bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        import torch.nn.functional as F

        def conv2d(x, w, stride, padding, approx=False, x_sign=True, w_sign=True):
            d, c, k, j = w.shape
            x_pad = F.pad(x, padding * 2, value=0.)
            x_pad = x_pad.unfold(2, k, stride[0])
            x_pad = x_pad.unfold(3, j, stride[1])
            x_pad = x_pad.unsqueeze(1)
            w = w.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            if approx:
                return p(x_pad, w, x_sign=x_sign, w_sign=w_sign)  # .sum(dim=(5,6,2))
            else:
                return x_pad.mul(w).sum(dim=(5, 6, 2))

        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() & 255 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() & 255 if w_sign else (w_l * 255).int()
            idx = x_c * 256 + w_c
            return approx_product[idx].sum(dim=(5, 6, 2))
            # x_c = x_c * 256
            # xw_c = []
            # for di in range(w_c.shape[1]):
            #     xw_c.append(approx_product[w_c[:, di:di + 1, :, :, :, :, :] + x_c].sum(dim=(5, 6, 2)))
            # xw_c = torch.concat(xw_c, dim=1)
            # return xw_c
            # idx = x_c + w_c
            # return approx_product[idx]

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        # bias, stride, padding, dilation, groups = None, (1, 1), (1, 1), (1, 1), 1
        x, w = torch.rand(2, 3, 4, 4), torch.rand(3, 3, 2, 2) * 2 - 1
        m_x, m_w = x.abs().max(), w.abs().max()
        # real
        # F.conv2d(x,w,bias,stride,padding,dilation,groups) == x_pad.mul(w).sum(dim=(5,6,2))
        conv_real = conv2d(x, w, stride, padding)  # .sum(dim=(5,6,2))
        # quant
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        conv_quant = conv2d(x_q, w_q, stride, padding)
        # approx
        x_l, w_l = x_q / m_x, w_q / m_w
        conv_approx = conv2d(x_l, w_l, stride, padding, approx=True, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(conv_real, conv_quant, conv_approx))

    def test_conv_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight,
                          bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        import torch.nn.functional as F

        def conv2d(x, w, stride, padding, approx=False, x_sign=True, w_sign=True):
            d, c, k, j = w.shape
            x_pad = F.pad(x, padding * 2, value=0.)
            x_pad = x_pad.unfold(2, k, stride[0])
            x_pad = x_pad.unfold(3, j, stride[1])
            x_pad = x_pad.unsqueeze(1)
            w = w.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            if approx:
                return p(x_pad, w, x_sign=x_sign, w_sign=w_sign)  # .sum(dim=(5,6,2))
            else:
                return x_pad.mul(w).sum(dim=(5, 6, 2))

        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() + 128 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() + 128 if w_sign else (w_l * 255).int()
            # return approx_product_value_2d[x_c, w_c].sum(dim=(5, 6, 2))
            return (approx_product_code_2d[x_c, w_c, :].sum(dim=(5, 6, 2)) * digit_weight).sum(4)
            # xw_c = []
            # for di in range(w_c.shape[1]):
            #     xw_c.append(approx_product[w_c[:, di:di + 1, :, :, :, :, :] + x_c].sum(dim=(5, 6, 2)))
            # xw_c = torch.concat(xw_c, dim=1)
            # return xw_c
            # idx = x_c + w_c
            # return approx_product[idx]

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        # bias, stride, padding, dilation, groups = None, (1, 1), (1, 1), (1, 1), 1
        x, w = torch.rand(2, 3, 4, 4), torch.rand(3, 3, 2, 2) * 2 - 1
        m_x, m_w = x.abs().max(), w.abs().max()
        # real
        # F.conv2d(x,w,bias,stride,padding,dilation,groups) == x_pad.mul(w).sum(dim=(5,6,2))
        conv_real = conv2d(x, w, stride, padding)  # .sum(dim=(5,6,2))
        # quant
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        conv_quant = conv2d(x_q, w_q, stride, padding)
        # approx
        x_l, w_l = x_q / m_x, w_q / m_w
        conv_approx = conv2d(x_l, w_l, stride, padding, approx=True, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(conv_real, conv_quant, conv_approx))

    approx_product_value_2d, approx_product_code_2d, digit_weight = None, None, None
    if product_bit == 0:
        # return None, 0.0
        return None, None, None, 0.0
    else:
        # import pickle
        # # f_info = '../backup/searched_info.pickle'
        # with open(f_info, 'rb') as f:
        #     searched_info = pickle.load(f)
        # # print(product_bit, type(product_bit))
        for info in searched_info:
            if info['bit-width of product'] == product_bit:
                print('bit-width of product: {}\t'
                      'root mean square error:{:.2e}'.format(info['bit-width of product'],
                                                             info['root mean square error']))
                # (info['code of activation'] * 2 ** torch.arange(0, 8).flip(0).view(1, -1)).sum(1)
                # (info['code of weight'] * 2 ** torch.arange(0, 8).flip(0).view(1, -1)).sum(1)
                assert info['bit-width of weight'] == info['bit-width of activation'] == bit
                # idx = (info['input code of product'] * 2 ** torch.arange(0, 2 * bit).flip(0).view(-1, 1)).sum(0)
                # approx_product = torch.zeros_like(idx).float()
                # approx_product[idx] = info['approximate value of product']
                approx_product_code_2d = info['output code of product'].view(product_bit, 256, 256).permute(1, 2, 0)
                approx_product_value_2d = info['approximate value of product'].view(256, 256)
                digit_weight = info['digit-weight of product code']
                # approx_product[idx] = info['real value of product']
                # test_fc(approx_product)
                # test_conv(approx_product)
                # test_fc_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight)
                # test_conv_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight)
                # return approx_product, info['root mean square error']
                return approx_product_value_2d, approx_product_code_2d, digit_weight, info['root mean square error']
        if None in (approx_product_value_2d, approx_product_code_2d, digit_weight):
            raise ValueError('did not find product_bit={} in {}'.format(product_bit, f_info))


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


if __name__ == '__main__':
    if args.baseline and (args.bit == 32 or args.bit == 8):
        args.product_bit = 0
        main()
    elif args.test or args.retrain:
        if args.product_bit != 0:
            main()
        else:
            raise ValueError('please specify product bit')
        # product_bit_list = [42]  # [65,60,55,48]
        # for args.product_bit in product_bit_list:
        #     main(args)
    else:
        raise ValueError('please specify running type')

    # main(args)
