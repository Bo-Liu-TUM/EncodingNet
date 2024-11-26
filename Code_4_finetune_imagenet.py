import gc
import time
import torch.nn as nn
import os
import torch
import logging
import numpy as np
from kuai_log import get_logger
import encode_tools as tools
from models.model_tools import AverageMeter, ProgressMeter, accuracy, validate, train

# available models please use torchvision.models.list_models()
# or find on https://pytorch.org/vision/stable/models.html#classification

import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'])

parser.add_argument('--a-bit', type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8, 32])
parser.add_argument('--w-bit', type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8, 32])

parser.add_argument('--baseline', action='store_true')

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--gpu', default='None', choices=['None', '0', '1', '2', '3']) 
parser.add_argument('--product-bit', type=int, default=0)
parser.add_argument('--search', type=int, default=128)
parser.add_argument('--cols', type=int, default=2)
parser.add_argument('--rows', type=int, default=256)
parser.add_argument('--th', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--fixed-type', type=str, default='big', choices=['big', 'small'])
parser.add_argument('--fixed-num', type=int, default=64)
parser.add_argument('--delta', type=int, default=0)

parser.add_argument('--lr', type=float, default=0.00001)

# default
# --arch efficientnet_b0 --a-bit 8 --w-bit 8 --baseline --gpu 0 --test
# python3 Code_2_finetune_cifar10.py --retrain --gpu 0 --product-bit 42 --search 64 --cols 1 --rows 256 --th 1.0 --epochs 23 --fixed-type big --fixed-num 0
# python3 Code_2_finetune_cifar10.py --retrain --bit 8 --gpu 0 --product-bit 42 --search 64 --cols 1 --rows 256 --th 1.5 --epochs 23 --fixed-type big --fixed-num 42 --delta 10
args = parser.parse_args()
args.gpu = None if args.gpu == 'None' else int(args.gpu)
args.dataset = 'imagenet'
args.data = '/nas/ei/share/TUEIEDAprojects/NNDatasets/imagenet2012'
# args.running_cache = '/home/ge26rem/lrz-nashome/LRZ/SourceCode/CGP_search/running_cache/'
args.running_cache = './running_cache/'

log_path = f"retrain-{args.dataset}-{args.arch}-{args.rows}row-{args.cols}col-" \
           f"{args.product_bit}bit-a-{args.a_bit}bit-w-{args.w_bit}bit" \
           f"{args.search}b-{args.th}th-{args.fixed_type}fixtype-{args.fixed_num}fixnum-{args.delta}delta.log"
logger = get_logger(name='retrain', level=logging.INFO, log_filename=log_path,
                    log_path='./retrain_logs/', is_add_file_handler=True,
                    formatter_template='{host}-cuda:' + str(args.gpu) + '-{levelname}-{message}'
                    )


def main():
    # global args

    args.workers = 4
    args.f_info = './search_results/searched_info-Uniform-{}bit.pickle'.format(args.product_bit)
    args.product_bit = 0 if (args.a_bit == 32 or args.w_bit == 32) else args.product_bit
    args.print_freq = 1

    if args.product_bit == 0:
        searched_info, args.OP, OP_info, args.model = None, None, None, None
    else:
        searched_info = tools.convert_searched_results(path=args.running_cache, th=args.th,
                                                       target=args.product_bit, search=args.search,
                                                       cols=args.cols, rows=args.rows)
        if args.delta > 0:
            searched_info = tools.apply_delta_for_finetune(searched_info, delta=args.delta)

        if searched_info[0]['bit-width of product'] < args.product_bit:
            if args.fixed_num == args.product_bit:
                args.product_bit = searched_info[0]['bit-width of product']
                args.fixed_num = searched_info[0]['bit-width of product']

    args.OP = None
    args.model = None
    args.by_code = True
    args.approx_product_value, \
        args.approx_product_code, \
        args.digit_weight, args.rmse = tools.get_approx_product(searched_info,
                                                                a_bit=args.a_bit, w_bit=args.w_bit,
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
    if args.arch == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        model = resnet50(weights=weights)
    elif args.arch == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        model = efficientnet_b0(weights=weights)
    else:
        raise ValueError('Todo for other models')

    print('-----> Done! model is built and pretrained weights are loaded!')

    from models.my_quant_layer import QuantFC, QuantConv2d, replace_conv_fc
    print('-------> starting replace original Conv2d layer and Linear layer with proposed QuantConv2d and QuantFC')
    model = replace_conv_fc(model, args=args)
    print('-------> all Conv2d and Linear layer are replaced with proposed QuantConv2d and QuantFC layer!')
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
    # print(list(model.modules())[0])
    if args.gpu is not None:
        model = model.cuda(args.gpu)

    # data loader by official torchversion:
    # --------------------------------------------------------------------------
    print('==> Using Pytorch Dataset')
    import torchvision
    print('------> loading dataset of ImageNet')
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

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
        print("sub_dataset, done!")
        dataset_new.imgs = [dataset.imgs[i] for i in sub_dataset_id]
        dataset_new.samples = [dataset.samples[i] for i in sub_dataset_id]
        dataset_new.targets = [dataset.targets[i] for i in sub_dataset_id]
        return dataset_new

    train_dataset = torchvision.datasets.ImageFolder(traindir, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, persistent_workers=True,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_dataset = torchvision.datasets.ImageFolder(valdir, transform=preprocess)

    val_loader = torch.utils.data.DataLoader(val_dataset, persistent_workers=True,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    if args.gpu is not None:
        criterion = criterion.cuda(args.gpu)



    if args.test:
        print('-------> testing begins! Good luck to you!\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\t'
              'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        if args.a_bit == 32 and args.w_bit == 32:
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
        elif args.a_bit == 32 or args.w_bit == 32:
            raise ValueError("not supported")
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
            # acc1, acc5, loss_avg = validate(train_loader, model, criterion, args)
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            print('-----------> real inference, done!')
            # exit()
        print('-------> testing done! info of this test:\n'
              'arc: {args.arch}\tdataset: {args.dataset}\ta_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        contents = 'testing result: \n' \
                   'neural network architecture: {args.arch}\n' \
                   'dataset: {args.dataset}\n' \
                   'quantization bit: a{args.a_bit}-bit / w{args.w_bit}-bit\n\n' \
                   'product bit: {args.product_bit}\n' \
                   'root mean squre error: {args.rmse:.4e}\n' \
                   'Acc-top1: {top1:.4f}%\n' \
                   'Acc-top5: {top5:.4f}%\n' \
                   'Loss: {loss:.4f}'.format(args=args, top1=acc1, top5=acc5, loss=loss_avg)
        subject = '{args.arch}({args.dataset}) / ' \
                  'a{args.a_bit}-bit / w{args.w_bit}-bit / {args.product_bit}-bit of product'.format(args=args)
        logger.info(subject)
        logger.info(contents)
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
        model_params = []
        if args.arch == 'resnet50':
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
            sub_dataset_ratio = 0.01
        elif args.arch == 'efficientnet_b0':
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
            optimizer = torch.optim.RMSprop(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-5, alpha=0.9)
            sub_dataset_ratio = 0.01
        else:
            raise ValueError('Todo for other models')
        logger.info(f'learning rate = {args.lr}')
        # logger.info(f'the following result of this test is used for testing before retrain')
        # acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        # logger.info(f'the above result of this test is used for testing before retrain')
        # exit()
        logger.info(f'-------> retraining begins! Good luck to you!\n'
                    f'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\t'
                    f'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
                    f'product bit: {args.product_bit}\trmse: {args.rmse:.4e}')
        logger.info(f'sub_dataset_ratio = {sub_dataset_ratio:.2%}')
        best_acc = None
        for epoch in range(args.epochs):
            train_dataset = torchvision.datasets.ImageFolder(traindir, transform=preprocess)
            train_dataset = sub_dataset(train_dataset, ratio=sub_dataset_ratio)
            train_loader = torch.utils.data.DataLoader(train_dataset, persistent_workers=True,
                                                       batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
            train(train_loader, model, criterion, optimizer, epoch, args, writer=None, end_batch=None)
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            if best_acc is None or acc1 > best_acc:
                best_acc = acc1
            logger.info(f'Test * Acc@1 {acc1:.3f}% (Best Acc@1 {best_acc:.3f}%)Acc@5 {acc5:.3f}%')

        # this validate is used for real inference
        print('-----------> starting testing after retraining......\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\t'
              'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        # acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        print('-----------> testing after retraining, done!\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\t'
              'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        contents = 'testing result: \n' \
                   'neural network architecture: {args.arch}\n' \
                   'dataset: {args.dataset}\n' \
                   'epochs: {args.epochs}\n' \
                   'quantization bit: a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t\n' \
                   'product bit: {args.product_bit}\n' \
                   'root mean squre error: {args.rmse:.4e}\n' \
                   'Acc-top1: {top1:.4f}%\n' \
                   'Acc-top5: {top5:.4f}%\n' \
                   'Loss: {loss:.4f}'.format(args=args, top1=acc1, top5=acc5, loss=loss_avg)
        subject = '{args.arch}({args.dataset}) / ' \
                  'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t / {args.product_bit}-bit of product'.format(args=args)
        logger.info(subject)
        logger.info(contents)
    else:
        raise ValueError('test or retrain? please specify it!')


if __name__ == '__main__':
    if args.baseline:  # and (args.bit == 32 or 0 < args.bit <= 8):
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
