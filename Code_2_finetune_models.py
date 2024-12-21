import os
import torch.nn as nn
import torch
import logging
import torchvision
from kuai_log import get_logger
import encode_tools as tools
import argparse
from models.model_tools import validate, train, load_dataset, sub_dataset

arch_cifar10 = ['resnet18', 'mobilenet_v2']
arch_cifar100 = ['resnet20', 'mobilenetv2_x0_5']
arch_imagenet = ['resnet50', 'efficientnet_b0']

parser = argparse.ArgumentParser(description='PyTorch cifar10 or cifar100 Training')
parser.add_argument('--arch', type=str, default='resnet18', choices=arch_cifar10 + arch_cifar100 + arch_imagenet)
parser.add_argument('--data', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet2012'])
parser.add_argument('--run', type=str, default='retrain', choices=['retrain', 'test'])
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--gpu', default='None', choices=['None', '0', '1', '2', '3'])
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--running-cache', type=str, default='./running_cache/')
parser.add_argument('--mode', type=str, default='FP32', choices=['FP32', 'Exact-INT', 'Approx-INT'])
parser.add_argument('--a-bit', type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8, 32])
parser.add_argument('--w-bit', type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8, 32])
parser.add_argument('--product-bit', type=int, default=0, choices=[36, 40, 42, 44, 46, 48, 52, 56, 60, 64])
parser.add_argument('--search', type=int, default=128, choices=[64, 128, 256])
parser.add_argument('--cols', type=int, default=2, choices=[1, 2, 3])
parser.add_argument('--rows', type=int, default=256, choices=[64, 128, 256])
parser.add_argument('--th', type=float, default=0.1, choices=[0.1, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20])
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n-parents', type=int, default=10)
parser.add_argument('--n-offsprings', type=int, default=50)
parser.add_argument('--n-champions', type=int, default=2)
parser.add_argument('--mutate-strategy', type=str, default='dynamic', choices=['dynamic', 'fixed'])
parser.add_argument('--mutate-rate', type=float, default=0.1)
parser.add_argument('--delta', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])

args = parser.parse_args()

args.gpu = None if args.gpu == 'None' else int(args.gpu)

if args.mode == 'FP32':
    args.a_bit = 32
    args.w_bit = 32
elif args.mode == 'Exact-INT':
    assert args.a_bit in [1, 2, 3, 4, 5, 6, 7, 8]
    assert args.w_bit in [1, 2, 3, 4, 5, 6, 7, 8]
    args.product_bit = args.a_bit + args.w_bit
elif args.mode == 'Approx-INT':
    args.a_bit = 8
    args.w_bit = 8
    assert args.product_bit in [36, 40, 42, 44, 46, 48, 52, 56, 60, 64]
    assert args.cols in [1, 2, 3]
    assert args.rows in [64, 128, 256]
    assert args.search in [64, 128, 256]
    assert args.th in [0.1, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
    assert args.delta in [0, 1, 2, 3, 4, 5]
else:
    ValueError("valid value of --mode is in ['FP32', 'Exact-INT', 'Approx-INT']")

if args.run == 'retrain':
    args.retrain = True
    args.test = False
elif args.run == 'test':
    args.retrain = False
    args.test = True
else:
    ValueError("--run should be in ['retrain', 'test']")

dataset_root = '/nas/ei/share/TUEIEDAprojects/NNDatasets'
args.data_path = os.path.join(dataset_root, args.data)

filename = tools.get_file_name(col=args.cols, row=args.rows, target=args.product_bit, search=args.search, idx=args.idx,
                               n_parents=args.n_parents, n_offsprings=args.n_offsprings, n_champions=args.n_champions,
                               mutate_strategy=args.mutate_strategy, mutate_rate=args.mutate_rate)

log_path = filename + f"{args.th}th-{args.delta}delta.log"
logger = get_logger(name='retrain', level=logging.INFO, log_filename=log_path,
                    log_path='./retrain_logs/', is_add_file_handler=True,
                    formatter_template='{host}-cuda:' + str(args.gpu) + '-{levelname}-{message}')


def main():
    # save running info to the log file
    logger.info(f'dataset: {args.data}, model: {args.arch}')
    logger.info(f'run: {args.run}, test: {args.test}, retrain: {args.retrain}')
    if args.run == 'retrain':
        logger.info(f'epochs: {args.epochs}, '
                    f'batch_size: {args.batch_size}, gpu: {args.gpu}, '
                    f'workers: {args.workers}, print_freq: {args.print_freq}')
    elif args.run == 'test':
        logger.info(f'batch_size: {args.batch_size}, gpu: {args.gpu}, '
                    f'workers: {args.workers}, print_freq: {args.print_freq}')
    else:
        ValueError("--run should be in ['retrain', 'test']")
    logger.info(f'mode: {args.mode}, a_bit: {args.a_bit}, w_bit: {args.w_bit}')
    if args.mode == 'FP32':
        pass
    elif args.mode == 'Exact-INT':
        logger.info(f'product_bit: {args.product_bit}')
    elif args.mode == 'Approx-INT':
        logger.info(f'product_bit: {args.product_bit}')
        logger.info(f'th: {args.th}%, cols: {args.cols}, rows: {args.rows}, search: {args.product_bit}/{args.search}')
        logger.info(f'n_parents: {args.n_parents}, n_offsprings: {args.n_offsprings}, n_champions: {args.n_champions}')
        logger.info(f'mutate_strategy: {args.mutate_strategy}, mutate_rate: {args.mutate_rate}')
        logger.info(f'idx: {args.idx}, delta: {args.delta}, running_cache: {args.running_cache}')
    else:
        ValueError("valid value of --mode is in ['FP32', 'Exact-INT', 'Approx-INT']")

    # load searched results
    logger.info(f'--> according to the CGP search configuration, load searched results')
    searched_info = tools.convert_searched_results(path=args.running_cache, th=args.th,
                                                   product_bit=args.product_bit, filename=filename + '.pickle')
    searched_info = tools.apply_delta_for_finetune(searched_info, delta=args.delta)

    if searched_info[0]['bit-width of product'] < args.product_bit:
        args.product_bit = searched_info[0]['bit-width of product']

    args.approx_product_value, args.digit_weight, args.rmse = tools.get_approx_product(searched_info,
                                                                                       a_bit=args.a_bit,
                                                                                       w_bit=args.w_bit,
                                                                                       product_bit=args.product_bit)

    logger.info(f'rmse: {args.rmse:.4e}')

    # move to GPU
    if args.gpu is not None and args.approx_product_value is not None:
        args.approx_product_value = args.approx_product_value.float().cuda(args.gpu)
    if args.gpu is not None and args.digit_weight is not None:
        args.digit_weight = args.digit_weight.cuda(args.gpu)

    # if 'out of memory', reduce mini_batch_size and mini_channels. 0 means full channels and full batch_size
    args.mini_batch_size = 5
    args.mini_channels = 0
    args.batch_size = 256

    if args.mini_batch_size == 0 or args.mini_batch_size > args.batch_size:
        args.mini_batch_size = args.batch_size

    logger.info(f'mini_batch_size: {args.mini_batch_size}, mini_channels: {args.mini_channels}')

    logger.info(f'--> build model, load pretrained weights, load dataset {args.data}')
    import torchvision.transforms as transforms
    if args.data == 'cifar10':  # https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master
        assert args.arch in arch_cifar10, f'arch for cifar10 should be in {arch_cifar10}'
        from cifar10_models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, resnet18, resnet34, resnet50
        from cifar10_models import densenet121, densenet161, densenet169, mobilenet_v2, googlenet, inception_v3
        model = eval(args.arch + '(pretrained=True)')
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2471, 0.2435, 0.2616]),
        ])
    elif args.data == 'cifar100':
        assert args.arch in arch_cifar100, f'arch for cifar100 should be in {arch_cifar100}'
        model = torch.hub.load(repo_or_dir='./cifar-models',
                               model=f'{args.data}_{args.arch}',
                               trust_repo=True,
                               source='local',
                               pretrained=True)
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                                 std=[0.2673, 0.2564, 0.2761]),
        ])
    elif args.data == 'imagenet2012':
        assert args.arch in arch_imagenet, f'arch for imagenet2012 should be in {arch_imagenet}'
        if args.arch == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1
            model = resnet50(weights=weights)
            dataset_transform = weights.transforms()
        elif args.arch == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            dataset_transform = weights.transforms()
        else:
            raise ValueError(f'--arch should be in {arch_imagenet}')
    else:
        raise ValueError("--data should be in ['cifar10', 'cifar100', 'imagenet2012']")

    logger.info(f'--> replace the Linear and Conv2d layers in the model')
    from models.my_quant_layer import QuantFC, QuantConv2d, replace_conv_fc
    model = replace_conv_fc(model, args=args)

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    train_loader, val_loader = load_dataset(root=args.data_path,
                                            transform=dataset_transform,
                                            batch_size=args.batch_size,
                                            workers=args.workers)

    criterion = nn.CrossEntropyLoss()

    if args.gpu is not None:
        criterion = criterion.cuda(args.gpu)

    t_conv, t_fc = 0, 0
    for m in model.modules():
        t_conv = t_conv + 1 if isinstance(m, QuantConv2d) else t_conv
        t_fc = t_fc + 1 if isinstance(m, QuantFC) else t_fc

    logger.info(f'total # conv layers:{t_conv}, total # fc layers: {t_fc}')
    logger.info(f'--> start {args.run} in {args.mode} mode')
    if args.test:
        if args.mode == 'FP32':
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
        else:  # args.mode == 'Exact-INT' or args.mode == 'Approx-INT'
            logger.info('--> find scale of x in each layer')
            acc1, acc5, loss_avg = validate(train_loader, model, criterion, args, end_batch=10)

            # close find scale of x mode
            for m in model.modules():
                if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
                    m.x_max_init_mode = False

            logger.info('--> start real inference')
            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            logger.info(f'Acc-top1: {acc1:.4f}%\tAcc-top5: {acc5:.4f}%\tLoss: {loss_avg:.4f}\t')

    if args.retrain:
        if args.mode == 'FP32':
            pass
        else:  # args.mode == 'Exact-INT' or args.mode == 'Approx-INT'
            if args.mode == 'Approx-INT':
                model_name = f'{args.run}-{args.mode}-{filename}.pth'
                if os.path.exists(model_name):
                    model.load_state_dict(torch.load(model_name))
                else:
                    logger.info('--> find scale of x in each layer')
                    acc1, acc5, loss_avg = validate(train_loader, model, criterion, args, end_batch=10)

                    # close find scale of x mode
                    for m in model.modules():
                        if isinstance(m, QuantConv2d) or isinstance(m, QuantFC):
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
                    param.requires_grad = False
                    model_params += [{'params': [param]}]
                else:
                    param.requires_grad = True
                    model_params += [{'params': [param]}]

            if args.data == 'cifar10':
                optimizer = torch.optim.SGD(model_params, lr=1e-4, momentum=0.9, weight_decay=1e-4)
            elif args.data == 'cifar100':
                optimizer = torch.optim.SGD(model_params, lr=1e-3, momentum=0.9, weight_decay=5e-4)  # 2.5e-3
            elif args.data == 'imagenet2012':
                if args.arch == 'resnet50':
                    optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
                elif args.arch == 'efficientnet_b0':
                    optimizer = torch.optim.RMSprop(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-5, alpha=0.9)
                else:
                    raise ValueError(f'--arch should be in {arch_imagenet}')
            else:
                raise ValueError("--data should be in ['cifar10', 'cifar100', 'imagenet2012']")

            acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
            logger.info(f'before finetune * Acc@1 {acc1:.3f}% Acc@5 {acc5:.3f}%')

            best_acc = None
            for epoch in range(args.epochs):
                if args.data == 'imagenet2012' and args.mode == 'Approx-INT':
                    sub_dataset_ratio = 0.1
                    train_dataset = torchvision.datasets.ImageNet(root=args.data_path, split='train',
                                                                  transform=dataset_transform)
                    train_dataset = sub_dataset(train_dataset, ratio=sub_dataset_ratio)
                    train_loader = torch.utils.data.DataLoader(train_dataset, persistent_workers=True,
                                                               batch_size=args.batch_size, shuffle=True,
                                                               num_workers=args.workers, pin_memory=True)
                train(train_loader, model, criterion, optimizer, epoch, args, writer=None, end_batch=None)
                acc1, acc5, loss_avg = validate(val_loader, model, criterion, args)
                if best_acc is None or acc1 > best_acc:
                    best_acc = acc1
                    if args.mode == 'Approx-INT':
                        model_name = f'{args.run}-{args.mode}-{filename}.pth'
                        torch.save(model.state_dict(), model_name)
                logger.info(f'Test * Acc@1 {acc1:.3f}% (Best Acc@1 {best_acc:.3f}%)Acc@5 {acc5:.3f}%')
            logger.info(f'Acc-top1: {acc1:.4f}% (Best: {best_acc:.4f}%)\tAcc-top5: {acc5:.4f}%\tLoss: {loss_avg:.4f}\t')


if __name__ == '__main__':
    main()
