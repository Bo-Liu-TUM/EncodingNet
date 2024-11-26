import gc
import time
import torch.nn as nn
import torch
import logging
from kuai_log import get_logger
import encode_tools as tools
import argparse
from models.model_tools import AverageMeter, ProgressMeter, accuracy, validate, train

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

# https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2'])

parser.add_argument('--a-bit', type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8, 32])
parser.add_argument('--w-bit', type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8, 32])

parser.add_argument('--baseline', action='store_true')

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--gpu', default='None', choices=['None', '0', '1', '2', '3'])

parser.add_argument('--product-bit', type=int, default=0)
parser.add_argument('--epochs', type=int, default=25)

parser.add_argument('--search', type=int, default=128)
parser.add_argument('--cols', type=int, default=2)
parser.add_argument('--rows', type=int, default=256)
parser.add_argument('--th', type=float, default=0.1)

parser.add_argument('--fixed-type', type=str, default='big', choices=['big', 'small'])
parser.add_argument('--fixed-num', type=int, default=64)
parser.add_argument('--delta', type=int, default=0)


args = parser.parse_args()
args.gpu = None if args.gpu == 'None' else int(args.gpu)
args.dataset = 'cifar10'
args.data = '/nas/ei/share/TUEIEDAprojects/NNDatasets/cifar10'
args.running_cache = './running_cache/'
args.workers = 4
args.f_info = './search_results/searched_info-Uniform-{}bit.pickle'.format(args.product_bit)
args.product_bit = 0 if (args.a_bit == 32 or args.w_bit == 32) else args.product_bit
args.print_freq = 1

log_path = f"retrain-{args.dataset}-{args.arch}-{args.rows}row-{args.cols}col-" \
           f"{args.product_bit}bit-a-{args.a_bit}bit-w-{args.w_bit}bit" \
           f"{args.search}b-{args.th}th-{args.fixed_type}fixtype-{args.fixed_num}fixnum-{args.delta}delta.log"
logger = get_logger(name='retrain', level=logging.INFO, log_filename=log_path,
                    log_path='./retrain_logs/', is_add_file_handler=True,
                    formatter_template='{host}-cuda:' + str(args.gpu) + '-{levelname}-{message}')


def main():
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
        args.mini_batch_size = 5
        args.mini_channels = 0
    args.batch_size = 256

    if args.mini_batch_size == 0 or args.mini_batch_size > args.batch_size:
        args.mini_batch_size = args.batch_size

    print('-----> build model and load pretrained weights')
    # https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master
    if args.arch == 'resnet18':
        from cifar10_models.resnet import resnet18
        model = resnet18(pretrained=True)
    elif args.arch == 'mobilenet_v2':
        from cifar10_models.mobilenetv2 import mobilenet_v2
        model = mobilenet_v2(pretrained=True)
    else:
        raise ValueError('Todo for other models')

    # for m in model.modules():
    #     if isinstance(m, torch.nn.Conv2d):
    #         print(m.groups, m.dilation)

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
    print('------> loading dataset of Cifar10')
    train_dataset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                                 transform=transforms.Compose([
                                                     # transforms.RandomCrop(32, padding=4),
                                                     # transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                          std=[0.2471, 0.2435, 0.2616]),
                                                 ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, persistent_workers=True, pin_memory=True,
                                               batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    test_dataset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                         std=[0.2471, 0.2435, 0.2616]),
                                                ]))
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=args.workers)

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
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\t'
              'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        if args.a_bit == 32 and args.w_bit == 32:
            print('-----------> starting 32bit float inference......')
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            print('-----------> 32bit float inference, done!')
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
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            print('-----------> real inference, done!')
        print('-------> testing done! info of this test:\n'
              'arc: {args.arch}\tdataset: {args.dataset}\ta_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        contents = 'testing result: \n' \
                   'neural network architecture: {args.arch}\n' \
                   'dataset: {args.dataset}\n' \
                   'quantization bit: a{args.a_bit}-bit / w{args.w_bit}-bit\n' \
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

        # close find scale of x mode
        for m in model.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
                print('-----> find scale of x: mode closed! for', m.layer_type)
                m.x_max_init_mode = False

        # which parameters can be learned?
        print('-----> enable or disable learnable parameters')
        model_params = []
        for name, param in model.named_parameters():
            if 'alpha_act' in name:
                param.requires_grad = True
                model_params += [{'params': [param], 'lr': 1e-5, 'weight_decay': 1e-4}]
            elif 'alpha_wgt' in name:
                param.requires_grad = True
                model_params += [{'params': [param], 'lr': 1e-5, 'weight_decay': 1e-4}]
            elif 'digit_weight' in name:
                param.requires_grad = True
                model_params += [{'params': [param], 'lr': 1e-7, 'weight_decay': 1e-7}]
            elif 'fc' in name and ('weight' in name or 'bias' in name):
                param.requires_grad = True
                model_params += [{'params': [param], 'lr': 1e-4, 'weight_decay': 1e-4}]
            else:
                param.requires_grad = True
                model_params += [{'params': [param]}]
            if param.requires_grad:
                print('Yes, enable to learn:', name)
            else:
                print('No, disable to learn:', name)
        optimizer = torch.optim.SGD(model_params, lr=1e-4, momentum=0.9, weight_decay=1e-4)


        logger.info('-------> retraining begins! Good luck to you!\n'
              'info of this test:\narc: {args.arch}\tdataset: {args.dataset}\t'
              'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
              'product bit: {args.product_bit}\trmse: {args.rmse:.4e}'.format(args=args))
        acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        best_acc = None
        logger.info(f'before finetune * Acc@1 {acc1:.3f}% Acc@5 {acc5:.3f}%')
        for epoch in range(args.epochs):
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
              'a_bit: {args.a_bit}\tw_bit: {args.w_bit}\t'
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
        subject = '{args.arch}({args.dataset}) / a_bit: {args.a_bit} / w_bit: {args.w_bit} / ' \
                  '{args.product_bit}-bit of product'.format(args=args)
        logger.info(subject)
        logger.info(contents)
    else:
        raise ValueError('test or retrain? please specify it!')


if __name__ == '__main__':
    if args.baseline:  # and (args.bit == 32 or (0 < args.bit <= 8)):
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
