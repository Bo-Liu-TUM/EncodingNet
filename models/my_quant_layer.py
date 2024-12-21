import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
import time
from .model_tools import AverageMeter


class QuantFC(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 weights=None, biases=None,
                 a_bit=8, w_bit=8, product_bit=19,
                 mini_batch_size=0,
                 mini_channels=0,
                 approx_product_value=None,
                 digit_weight=None,
                 mode='FP32'
                 ):
        super(QuantFC, self).__init__(in_features, out_features, bias)

        self.weight = weights
        self.bias = biases

        self.x_max_init_mode = True
        self.x_for_levels = None
        self.x_max = AverageMeter('x_max', ':6.2f')  # this is used to scale x during inference

        self.layer_type = 'LFC'
        self.mode = mode
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.product_bit = product_bit
        self.product_approx = product_approximation(a_bit=self.a_bit, w_bit=self.w_bit,
                                                    mini_batch_size=mini_batch_size,
                                                    mini_channels=mini_channels,
                                                    approx_product_value=approx_product_value
                                                    )
        self.act_quant = quantization(bit=self.a_bit, signed=True)
        self.wgt_quant = quantization(bit=self.w_bit, signed=True)
        self.digit_weight = Parameter(digit_weight)
        self.alpha_act = Parameter(torch.tensor(0.))
        self.alpha_wgt = Parameter(torch.tensor(0.))

        # print notice
        print('-----> find scale of x: mode opened! for', self.layer_type)

    def forward(self, x):
        if self.mode == 'FP32' or self.x_max_init_mode:  # float, without quantization
            if self.x_max_init_mode:
                self.x_max.update(x.data.abs().max().detach().item())
                if self.x_for_levels is None:
                    self.x_for_levels = x.clone()
                self.alpha_act = Parameter(torch.tensor(self.x_max.avg))
                self.alpha_wgt = Parameter(self.weight.data.abs().max())
            return F.linear(x, self.weight, self.bias)
        else:
            weight_q = self.wgt_quant(self.weight, self.alpha_wgt)
            x_q = self.act_quant(x, self.alpha_act)
            if self.mode == 'Exact-INT':  # without our idea
                return F.linear(x_q, weight_q, self.bias)
            else:  # with our idea
                x_q = x_q.div(self.alpha_act.detach())
                weight_q = weight_q.div(self.alpha_wgt.detach())
                x_q = x_q.unsqueeze(1)
                weight_q = weight_q.unsqueeze(0)
                out = self.product_approx(x_q, weight_q, self.digit_weight)
                out = out * self.alpha_act.detach() * self.alpha_wgt.detach()
                if self.bias is not None:
                    out = out + self.bias
                return out

    def show_params(self):
        pass
        # wgt_alpha = round(self.weight_quant.alpha.data.item(), 3)
        # act_alpha = round(self.activa_quant.alpha.data.item(), 3)
        # print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 weights=None, biases=None,
                 a_bit=8, w_bit=8, product_bit=19,
                 mini_batch_size=0,
                 mini_channels=0,
                 approx_product_value=None,
                 digit_weight=None,
                 mode='FP32'
                 ):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)

        self.weight = weights
        self.bias = biases

        self.x_max_init_mode = True
        self.x_for_levels = None
        self.x_max = AverageMeter('x_max', ':6.2f')  # this is used to scale x during inference

        self.layer_type = 'QuantConv2d'
        self.mode = mode
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.product_bit = product_bit
        self.product_approx = product_approximation(a_bit=self.a_bit, w_bit=self.w_bit,
                                                    mini_batch_size=mini_batch_size,
                                                    mini_channels=mini_channels,
                                                    approx_product_value=approx_product_value
                                                    )
        self.act_quant = quantization(bit=self.a_bit, signed=True)
        self.wgt_quant = quantization(bit=self.w_bit, signed=True)
        self.digit_weight = Parameter(digit_weight)
        self.alpha_act = Parameter(torch.tensor(0.))
        self.alpha_wgt = Parameter(torch.tensor(0.))

        # print notice
        print('-----> find scale of x: mode opened! for', self.layer_type)

    def forward(self, x):
        if self.mode == 'FP32' or self.x_max_init_mode:  # float, without quantization
            if self.x_max_init_mode:
                self.x_max.update(x.data.abs().max().detach().item())
                if self.x_for_levels is None:
                    self.x_for_levels = x.clone()
                self.alpha_act = Parameter(torch.tensor(self.x_max.avg))
                self.alpha_wgt = Parameter(self.weight.data.abs().max())
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:  # with uniform quantization
            weight_q = self.wgt_quant(self.weight, self.alpha_wgt)
            x_q = self.act_quant(x, self.alpha_act)
            if self.mode == 'Exact-INT':  # without our idea
                return F.conv2d(x_q, weight_q, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            else:  # with our idea
                x_q = x_q.div(self.alpha_act.detach())
                weight_q = weight_q.div(self.alpha_wgt.detach())
                if self.groups > 1:
                    x_split = torch.split(x_q, self.in_channels // self.groups, dim=1)
                    weight_split = torch.split(weight_q, self.out_channels // self.groups, dim=0)
                    out = [self.sub_quant_conv2d(x_g, w_g) for x_g, w_g in zip(x_split, weight_split)]
                    # Concatenate the results along the channel dimension
                    return torch.cat(out, dim=1)
                else:
                    return self.sub_quant_conv2d(x_q, weight_q)

    def sub_quant_conv2d(self, x_q, weight_q):
        d, c, k, j = weight_q.shape
        x_q = F.pad(x_q, self.padding * 2, value=0.)
        x_q = x_q.unfold(2, k, self.stride[0])
        x_q = x_q.unfold(3, j, self.stride[1])
        x_q = x_q.unsqueeze(1)
        weight_q = weight_q.unsqueeze(2).unsqueeze(2).unsqueeze(0)
        out = self.product_approx(x_q, weight_q, self.digit_weight)
        out = out * self.alpha_act.detach() * self.alpha_wgt.detach()
        if self.bias is not None:
            assert self.bias.numel() == d, 'num of bias must be equal to out channels'
            out = out + self.bias.view(1, -1, 1, 1)  # 添加偏置值
        return out

    def show_params(self):
        pass
        # wgt_alpha = round(self.weight_quant.alpha.data.item(), 3)
        # act_alpha = round(self.activa_quant.alpha.data.item(), 3)
        # print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


def product_approximation(a_bit, w_bit, approx_product_value,
                          mini_batch_size=5, mini_channels=1):
    # my idea require very large memory during training and inference,
    # so try to solve a mini batch each time to save memory but require more time
    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_l, w_l, digit_weight):
            scale_w = 2 ** (w_bit - 1) - 1
            scale_x = 2 ** (a_bit - 1) - 1
            x_c = (x_l * scale_x).to(torch.int32) + (2 ** (a_bit - 1))
            w_c = (w_l * scale_w).to(torch.int32) + (2 ** (w_bit - 1))
            batch_size = x_c.size(0)
            channel_size = w_c.size(1)
            if x_c.ndim == w_c.ndim == 3:
                xw_c = []
                end_batch = 0
                for i in range(0, batch_size, mini_batch_size):
                    end_batch = i + mini_batch_size
                    xw_c.append(approx_product_value[x_c[i:end_batch], w_c].sum(2))
                if end_batch < batch_size:
                    xw_c.append(approx_product_value[x_c[end_batch:], w_c].sum(2))
                xw_c = torch.cat(xw_c, dim=0)
                # xw_c = approx_product_value[x_c, w_c].sum(2)
                ctx.save_for_backward(x_l, w_l, xw_c)
                return xw_c
            elif x_c.ndim == w_c.ndim == 7:
                xw_c = []
                end_batch = 0
                for i in range(0, batch_size, mini_batch_size):
                    end_batch = i + mini_batch_size
                    xw_sub_c = []
                    end_channel = 0
                    for j in range(0, channel_size, mini_channels):
                        end_channel = j + mini_channels
                        xw_sub_c.append(
                            approx_product_value[x_c[i:end_batch], w_c[:, j:end_channel]].sum(dim=(5, 6, 2)))
                    if end_channel < channel_size:
                        xw_sub_c.append(
                            approx_product_value[x_c[i:end_batch], w_c[:, end_channel:]].sum(dim=(5, 6, 2)))
                    xw_c.append(torch.cat(xw_sub_c, dim=1))
                if end_batch < batch_size:
                    xw_sub_c = []
                    end_channel = 0
                    for j in range(0, channel_size, mini_channels):
                        end_channel = j + mini_channels
                        xw_sub_c.append(
                            approx_product_value[x_c[end_batch:], w_c[:, j:end_channel]].sum(dim=(5, 6, 2)))
                    if end_channel < channel_size:
                        xw_sub_c.append(
                            approx_product_value[x_c[end_batch:], w_c[:, end_channel:]].sum(dim=(5, 6, 2)))
                    xw_c.append(torch.cat(xw_sub_c, dim=1))
                xw_c = torch.cat(xw_c, dim=0)
                # xw_c = approx_product_value[x_c, w_c].sum(dim=(5, 6, 2))
                ctx.save_for_backward(x_l, w_l, xw_c)
                return xw_c

        @staticmethod
        def backward(ctx, grad_output):
            x_l, w_l, xw_c = ctx.saved_tensors
            batch_size = x_l.size(0)
            if x_l.ndim == w_l.ndim == 3:
                grad_output = grad_output.unsqueeze(2)
                grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                grad_w_l = (grad_output * x_l).sum(0, keepdim=True)
            elif x_l.ndim == w_l.ndim == 7:
                grad_output = grad_output.unsqueeze(2).unsqueeze(5).unsqueeze(6)
                grad_x_l = []
                grad_w_l = []
                end_batch = 0
                for i in range(0, batch_size, mini_batch_size):
                    end_batch = i + mini_batch_size
                    grad_x_l.append((grad_output[i:end_batch] * w_l).sum(1, keepdim=True))
                    grad_w_l.append((grad_output[i:end_batch] * x_l[i:end_batch]).sum((0, 3, 4), keepdim=True))
                if end_batch < batch_size:
                    grad_x_l.append((grad_output[end_batch:] * w_l).sum(1, keepdim=True))
                    grad_w_l.append((grad_output[end_batch:] * x_l[i:end_batch]).sum((0, 3, 4), keepdim=True))
                grad_x_l = torch.cat(grad_x_l, dim=0)
                grad_w_l = torch.cat(grad_w_l, dim=0)
            else:
                raise ValueError('todo')
            return grad_x_l, grad_w_l, None

    return _pq().apply


def quantization(bit, signed=True, uniform=True, grids=None):

    def uniform_quant(x, b):
        scale = (2 ** (b - 1) - 1) if signed else (2 ** b - 1)
        return x.mul(scale).round().div(scale)

    def non_uniform_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input = input.div(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1) if signed else input.clamp(min=0, max=1) # then clipped
            if uniform:
                input_q = uniform_quant(input_c, bit)
            else:
                input_q = non_uniform_quant(input_c, grids)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            if signed:
                i = (input.abs() > 1.).float()
                sign = input.sign()
                grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            else:
                i = (input > 1.).float()
                grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
                grad_input = grad_input * (1 - i)       # grad for activation will be clipped
            return grad_input, grad_alpha

    return _pq().apply


def get_conv_params(m):
    (in_channels, out_channels, kernel_size,
     stride, padding, dilation, groups, weights, biases) = (m.in_channels, m.out_channels, m.kernel_size,
                                                            m.stride, m.padding, m.dilation, m.groups, m.weight, m.bias)
    bias = False if biases is None else True
    if m.dilation != (1, 1):
        raise ValueError('dilation > 1')
    return (in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, weights, biases, bias)


def get_fc_params(m):
    (in_features, out_features, weights, biases) = (m.in_features, m.out_features, m.weight, m.bias)
    bias = False if biases is None else True
    return (in_features, out_features, weights, biases, bias)


def replace_conv_fc(m, args):
    for k in m._modules.keys():
        if m._modules[k]._modules.keys().__len__() == 0:
            if isinstance(m._modules[k], torch.nn.Conv2d):
                (in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, weights, biases, bias) = get_conv_params(m._modules[k])
                if args.mini_channels == 0 or args.mini_channels > out_channels:
                    mini_channels = out_channels
                else:
                    mini_channels = args.mini_channels
                m._modules[k] = QuantConv2d(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=groups, bias=bias,
                                            weights=weights, biases=biases,
                                            a_bit=args.a_bit, w_bit=args.w_bit,
                                            product_bit=args.product_bit,
                                            mini_batch_size=args.mini_batch_size,
                                            mini_channels=mini_channels,
                                            approx_product_value=args.approx_product_value,
                                            digit_weight=args.digit_weight,
                                            mode=args.mode
                                            )

            elif isinstance(m._modules[k], torch.nn.Linear):
                (in_features, out_features, weights, biases, bias) = get_fc_params(m._modules[k])
                m._modules[k] = QuantFC(in_features, out_features,
                                        bias=bias, weights=weights, biases=biases,
                                        a_bit=args.a_bit, w_bit=args.w_bit, product_bit=args.product_bit,
                                        mini_batch_size=args.mini_batch_size,
                                        mini_channels=0,
                                        approx_product_value=args.approx_product_value,
                                        digit_weight=args.digit_weight,
                                        mode=args.mode
                                        )
        else:
            m._modules[k] = replace_conv_fc(m=m._modules[k], args=args)
    return m

