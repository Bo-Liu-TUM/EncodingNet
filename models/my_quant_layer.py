import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
import time

class QuantFC(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 weights=None, biases=None, bit=8, product_bit=19,
                 mini_batch_size=0,
                 mini_channels=0,
                 # approx_product=None,
                 approx_product_code=None,
                 approx_product_value=None,
                 digit_weight=None,
                 digit_weight_mask=None,
                 by_code=False,
                 model=None,
                 OP=None
                 ):
        super(QuantFC, self).__init__(in_features, out_features, bias)

        self.weight = weights
        self.bias = biases

        self.x_max_init_mode = True
        self.x_for_levels = None
        self.x_max = AverageMeter('x_max', ':6.2f')  # this is used to scale x during inference

        self.layer_type = 'LFC'
        self.bit = bit
        self.product_bit = product_bit
        self.product_approx = product_approximation(bit=self.bit,
                                                    digit_weight_mask=digit_weight_mask,
                                                    # approx_product=approx_product,
                                                    mini_batch_size=mini_batch_size,
                                                    mini_channels=mini_channels,
                                                    approx_product_code=approx_product_code,
                                                    approx_product_value=approx_product_value,
                                                    by_code=by_code,
                                                    model=None,
                                                    OP=OP
                                                    )
        self.act_quant = quantization(bit=self.bit, signed=True)
        self.wgt_quant = quantization(bit=self.bit, signed=True)
        self.digit_weight = Parameter(digit_weight)
        self.alpha_act = Parameter(torch.tensor(0.))
        self.alpha_wgt = Parameter(torch.tensor(0.))

        # print notice
        print('-----> find scale of x: mode opened! for', self.layer_type)

    def forward(self, x):
        # fc_real = F.linear(x, self.weight, self.bias)
        if self.bit == 32 or self.x_max_init_mode:  # float, without quantization
            if self.x_max_init_mode:
                self.x_max.update(x.data.abs().max().detach().item())
                if self.x_for_levels is None:
                    self.x_for_levels = x.clone()
                self.alpha_act = Parameter(torch.tensor(self.x_max.avg))
                self.alpha_wgt = Parameter(self.weight.data.abs().max())
                # print(f'x_r: min:{x.min():.4f}\tmax: {x.max():.4f}\tavg: {x.mean():.4f}\t'
                #       f'->updated fixed scale:{self.x_max.avg:.4f}\tlayer: {self.layer_type}')
            # print(x.shape, self.weight.shape)
            return F.linear(x, self.weight, self.bias)
        else:  # with uniform quantization
            # max_w = self.weight.data.abs().max()
            # scale_w = 2 ** (self.bit - 1) - 1
            # weight_q = self.weight.div(max_w).mul(scale_w).round().div(scale_w).mul(max_w)
            # weight_q = (weight_q - self.weight).detach() + self.weight
            weight_q = self.wgt_quant(self.weight, self.alpha_wgt)

            # max_x = x.data.abs().max()
            # scale_x = 2 ** self.bit - 1

            # max_x = x.data.abs().max()
            # scale_x = 2 ** (self.bit - 1) - 1
            # x_q = x.div(max_x).mul(scale_x).round().div(scale_x).mul(max_x)

            # max_x = self.x_max.avg
            # scale_x = 2 ** (self.bit - 1) - 1
            # x_q = x.div(max_x).clamp(min=-1, max=1).mul(scale_x).round().div(scale_x).mul(max_x)
            # x_q = (x_q - x).detach() + x

            x_q = self.act_quant(x, self.alpha_act)

            # print('x_q: min:{:.4f}\tmax: {:.4f}\tavg: {:.4f}\t'
            #       '->fixed scale x: {:.4f}\t'
            #       '->fixed scale w: {:.4f}\t'
            #       'layer: {}'.format(x_q.min(), x_q.max(), x_q.mean(),
            #                          self.alpha_act.detach(), self.alpha_wgt.detach(),
            #                          self.layer_type))

            # fc_quant = F.linear(x_q, weight_q, self.bias)
            if self.product_bit <= 0:  # without our idea
                # print('(real - quant) rmse: {:.4e}\t'.format(
                #     (fc_real - fc_quant).norm() / torch.sqrt(torch.tensor(fc_real.numel()))
                # ))
                return F.linear(x_q, weight_q, self.bias)
            else:  # with our idea
                # print(self.layer_type, '--> digit-weight: ', self.digit_weight.cpu().detach().numpy().round(3).reshape(-1))
                # x_q = x_q.div(max_x)
                x_q = x_q.div(self.alpha_act.detach())
                # weight_q = weight_q.div(max_w)
                weight_q = weight_q.div(self.alpha_wgt.detach())
                x_q = x_q.unsqueeze(1)
                weight_q = weight_q.unsqueeze(0)
                # xw = self.product_approx(x_q, weight_q)
                out = self.product_approx(x_q, weight_q, self.digit_weight)
                # print((x_q * weight_q - xw).norm() / torch.tensor(xw.numel()).float().sqrt())
                # out = out * max_x * max_w
                out = out * self.alpha_act.detach() * self.alpha_wgt.detach()
                if self.bias is not None:
                    out = out + self.bias

                # print('(real - quant) rmse: {:.4e}\t'
                #       '(quant - approx) rmse: {:.4e}\t'
                #       '(real - approx) rmse: {:.4e}\t'.format(
                #     (fc_real - fc_quant).norm() / torch.sqrt(torch.tensor(fc_real.numel())),
                #     (out - fc_quant).norm() / torch.sqrt(torch.tensor(fc_real.numel())),
                #     (fc_real - out).norm() / torch.sqrt(torch.tensor(fc_real.numel())),
                # ))

                return out

    def show_params(self):
        pass
        # wgt_alpha = round(self.weight_quant.alpha.data.item(), 3)
        # act_alpha = round(self.activa_quant.alpha.data.item(), 3)
        # print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 weights=None, biases=None, bit=8, product_bit=19,
                 mini_batch_size=0,
                 mini_channels=0,
                 # approx_product=None,
                 approx_product_code=None,
                 approx_product_value=None,
                 digit_weight=None,
                 digit_weight_mask=None,
                 by_code=False,
                 model=None,
                 OP=None
                 ):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)

        self.weight = weights
        self.bias = biases

        self.x_max_init_mode = True
        self.x_for_levels = None
        self.x_max = AverageMeter('x_max', ':6.2f')  # this is used to scale x during inference

        self.layer_type = 'QuantConv2d'
        self.bit = bit
        self.product_bit = product_bit
        self.product_approx = product_approximation(bit=self.bit,
                                                    digit_weight_mask=digit_weight_mask,
                                                    # approx_product=approx_product,
                                                    mini_batch_size=mini_batch_size,
                                                    mini_channels=mini_channels,
                                                    approx_product_code=approx_product_code,
                                                    approx_product_value=approx_product_value,
                                                    by_code=by_code,
                                                    model=None,
                                                    OP=OP
                                                    )
        self.act_quant = quantization(bit=self.bit, signed=True)
        self.wgt_quant = quantization(bit=self.bit, signed=True)
        self.digit_weight = Parameter(digit_weight)
        self.alpha_act = Parameter(torch.tensor(0.))
        self.alpha_wgt = Parameter(torch.tensor(0.))

        # print notice
        print('-----> find scale of x: mode opened! for', self.layer_type)

    def forward(self, x):
        # conv_real = F.conv2d(x, self.weight, self.bias, self.stride,
        #                      self.padding, self.dilation, self.groups)
        if self.bit == 32 or self.x_max_init_mode:  # float, without quantization
            if self.x_max_init_mode:
                self.x_max.update(x.data.abs().max().detach().item())
                if self.x_for_levels is None:
                    self.x_for_levels = x.clone()
                self.alpha_act = Parameter(torch.tensor(self.x_max.avg))
                self.alpha_wgt = Parameter(self.weight.data.abs().max())
                # print(f'x_r: min:{x.min():.4f}\tmax: {x.max():.4f}\tavg: {x.mean():.4f}\t'
                #       f'->updated fixed scale:{self.x_max.avg:.4f}\tlayer: {self.layer_type}')
            # print(x.shape, self.weight.shape)
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:  # with uniform quantization
            # max_w = self.weight.data.abs().max()
            # scale_w = 2 ** (self.bit - 1) - 1
            # weight_q = self.weight.div(max_w).mul(scale_w).round().div(scale_w).mul(max_w)
            # weight_q = (weight_q - self.weight).detach() + self.weight

            weight_q = self.wgt_quant(self.weight, self.alpha_wgt)

            # max_x = x.data.abs().max()
            # scale_x = 2 ** (self.bit - 1) - 1  # 2 ** self.bit - 1
            # x_q = x.div(max_x).mul(scale_x).round().div(scale_x).mul(max_x)

            # max_x = self.x_max.avg
            # scale_x = 2 ** (self.bit - 1) - 1
            # x_q = x.div(max_x).clamp(min=-1, max=1).mul(scale_x).round().div(scale_x).mul(max_x)
            # x_q = (x_q - x).detach() + x

            x_q = self.act_quant(x, self.alpha_act)

            # print('x_q: min:{:.4f}\tmax: {:.4f}\tavg: {:.4f}\t'
            #       '->fixed scale:{:.4f}\t'
            #       '->fixed scale w: {:.4f}\t'
            #       'layer: {}'.format(x_q.min(), x_q.max(), x_q.mean(),
            #                          self.alpha_act.detach(), self.alpha_wgt.detach(),
            #                          self.layer_type))
            # conv_quant = F.conv2d(x_q, weight_q, self.bias, self.stride,
            #                       self.padding, self.dilation, self.groups)

            if self.product_bit <= 0:  # without our idea
                # print('(real - quant) rmse: {:.4e}\t'.format(
                #     (conv_real - conv_quant).norm() / torch.sqrt(torch.tensor(conv_real.numel()))
                # ))
                return F.conv2d(x_q, weight_q, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            else:  # with our idea
                # print(self.layer_type, '--> digit-weight: ', self.digit_weight.cpu().detach().numpy().round(3).reshape(-1))
                # assert (x_q.dim() == 4 and weight_q.dim() == 4), 'x and weight should be 4-dim'
                # n, c, h_in, w_in = x_q.shape
                d, c, k, j = weight_q.shape
                # x_q = x_q.div(max_x)
                x_q = x_q.div(self.alpha_act.detach())
                # weight_q = weight_q.div(max_w)
                weight_q = weight_q.div(self.alpha_wgt.detach())
                x_q = F.pad(x_q, self.padding * 2, value=0.)
                x_q = x_q.unfold(2, k, self.stride[
                    0])  # Todo: try to use F.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
                x_q = x_q.unfold(3, j, self.stride[1])
                x_q = x_q.unsqueeze(1)
                weight_q = weight_q.unsqueeze(2).unsqueeze(2).unsqueeze(0)
                # xw = self.product_approx(x_q, weight_q)
                out = self.product_approx(x_q, weight_q, self.digit_weight)
                # print((x_q * weight_q - xw).norm() / torch.tensor(xw.numel()).float().sqrt())
                # out = out * max_x * max_w
                out = out * self.alpha_act.detach() * self.alpha_wgt.detach()
                # out = xw.sum(dim=(5, 6, 2))
                if self.bias is not None:
                    assert self.bias.numel() == d, 'num of bias must be equal to out channels'
                    out = out + self.bias.view(1, -1, 1, 1)  # 添加偏置值

                # print('(real - quant) rmse: {:.4e}\t'
                #       '(quant - approx) rmse: {:.4e}\t'
                #       '(real - approx) rmse: {:.4e}\t'.format(
                #     (conv_real - conv_quant).norm() / torch.sqrt(torch.tensor(conv_real.numel())),
                #     (out - conv_quant).norm() / torch.sqrt(torch.tensor(conv_real.numel())),
                #     (conv_real - out).norm() / torch.sqrt(torch.tensor(conv_real.numel())),
                # ))
                return out

    def show_params(self):
        pass
        # wgt_alpha = round(self.weight_quant.alpha.data.item(), 3)
        # act_alpha = round(self.activa_quant.alpha.data.item(), 3)
        # print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# def product_approximation(bit, approx_product):
#     class _pq(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, x_l, w_l):
#             ctx.save_for_backward(x_l, w_l)
#             scale_w = 2 ** (bit - 1) - 1
#             # scale_x = 2 ** bit - 1
#             scale_x = 2 ** (bit - 1) - 1
#             # x_c = (x_l * scale_x).int() * (2 ** bit)
#             x_c = ((x_l * scale_x).int() & (2 ** bit - 1)) * (2 ** bit)
#             w_c = (w_l * scale_w).int() & (2 ** bit - 1)
#             # print(x_c.shape, w_c.shape)
#             if x_c.ndim == w_c.ndim == 3:
#                 N = 16
#                 d = int(x_c.shape[2] / N)
#                 xw_c = []
#                 for n in range(N):
#                     dd = min(x_c.shape[2], (n + 1) * d)  # .to(torch.uint8)
#                     xw_c.append(approx_product[x_c[:, :, n * d:dd] + w_c[:, :, n * d:dd]].sum(dim=2, keepdim=True))
#                 xw_c = torch.concat(xw_c, dim=2).sum(dim=2)
#                 # xw_c = x_c + w_c.to(torch.uint8)
#                 # xw_c = approx_product[xw_c]
#             elif x_c.ndim == w_c.ndim == 7:
#                 # w_c = w_c.to(torch.uint8)
#                 N = int(w_c.shape[1] / 1)
#                 d = int(w_c.shape[1] / N)
#                 xw_c = []
#                 for n in range(N):
#                     dd = min(w_c.shape[1], (n + 1) * d)
#                     xw_c.append(approx_product[w_c[:, n * d:dd, :, :, :, :, :] + x_c].sum(dim=(5, 6, 2)))
#                 xw_c = torch.concat(xw_c, dim=1)
#                 # for di in range(w_c.shape[1]):
#                 #     xw_c.append(approx_product[w_c[:, di:di + 1, :, :, :, :, :] + x_c].sum(dim=(5, 6, 2)))
#                 # xw_c = torch.concat(xw_c, dim=1)
#             # print(xw_c.numel(), approx_product.numel())
#             # if approx_product.device != xw_c.device:
#             #     xw_c = xw_c.to(approx_product.device)
#             # if x_l.ndim == w_l.ndim == 3:
#             #     print(x_l.shape, w_l.shape, x_l.ndim, w_l.ndim)
#             #     pass
#             return xw_c
#
#         @staticmethod
#         def backward(ctx, grad_output):
#             # o = grad_output.clone()
#             x_l, w_l = ctx.saved_tensors
#             # print(x_l.shape, w_l.shape, x_l.ndim, w_l.ndim)
#             # grad_x_l, grad_w_l = torch.ones_like(x_l), torch.ones_like(w_l)
#             if x_l.ndim == w_l.ndim == 3:
#                 grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
#                 grad_w_l = (grad_output * x_l).sum(0, keepdim=True)
#             elif x_l.ndim == w_l.ndim == 7:
#                 grad_output = grad_output.unsqueeze(2).unsqueeze(5).unsqueeze(6)
#                 grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
#                 grad_w_l = (grad_output * x_l).sum((0, 3, 4), keepdim=True)
#             else:
#                 raise ValueError('todo')
#             return grad_x_l, grad_w_l
#
#     return _pq().apply


def product_approximation(bit, approx_product_code, approx_product_value, digit_weight_mask,
                          mini_batch_size=5, mini_channels=1, by_code=False, model=None, OP=None):
    if OP is not None:
        class _pq(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_l, w_l, digit_weight):
                scale_w = 2 ** (bit - 1) - 1  # scale_x = 2 ** bit - 1
                scale_x = 2 ** (bit - 1) - 1  # x_c = (x_l * scale_x).int() * (2 ** bit)
                x_c = (x_l * scale_x).to(torch.int8)
                w_c = (w_l * scale_w).to(torch.int8)
                # print(OP[0](x_c, w_c).dtype)
                # print(x_c.shape, w_c.shape)
                if x_c.ndim == w_c.ndim == 3:
                    t1 = time.perf_counter()
                    xw_c = []
                    for op in OP:
                        xw_c.append(op(x_c, w_c).sum(dim=(2,)))
                    t2 = time.perf_counter()
                    print(f'fc op forward: {t2 - t1: 3f}秒')
                    xw_c = torch.stack(xw_c, dim=2).to(torch.int16)
                    t3 = time.perf_counter()
                    print(f'fc stack forward: {t3 - t2: 3f}秒')
                    # xw_c = torch.stack([op(x_c, w_c).sum(dim=(2,)) for op in OP], dim=2).to(torch.int16)  # torch.Size([256, 10, 42])
                    # print(xw_c.shape)
                    ctx.save_for_backward(x_l, w_l, xw_c)
                    return (xw_c * digit_weight).sum(2)
                elif x_c.ndim == w_c.ndim == 7:
                    t1 = time.perf_counter()
                    xw_c = []
                    for op in OP:
                        xw_c.append(op(x_c, w_c).sum(dim=(5, 6, 2)))
                    t2 = time.perf_counter()
                    print(f'conv op forward: {t2 - t1: 3f}秒')
                    xw_c = torch.stack(xw_c, dim=4).to(torch.int16)
                    t3 = time.perf_counter()
                    print(f'conv stack forward: {t3 - t2: 3f}秒')
                    # xw_c = torch.stack([op(x_c, w_c).sum(dim=(5, 6, 2)) for op in OP], dim=4).to(torch.int16)  # torch.Size([256, 512, 2, 2, 42])
                    # print(xw_c.shape)
                    ctx.save_for_backward(x_l, w_l, xw_c)
                    return (xw_c * digit_weight).sum(4)

            @staticmethod
            def backward(ctx, grad_output):
                # pdb.set_trace()
                x_l, w_l, xw_c = ctx.saved_tensors
                # print(grad_output.shape, x_l.shape, w_l.shape, xw_c.shape)
                # print(grad_output.dtype, x_l.dtype, w_l.dtype, xw_c.dtype)
                if x_l.ndim == w_l.ndim == 3:
                    grad_output = grad_output.unsqueeze(2)
                    grad_digit_weight = (xw_c * grad_output).sum(dim=(0, 1))
                    grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                    grad_w_l = (grad_output * x_l).sum(0, keepdim=True)
                elif x_l.ndim == w_l.ndim == 7:
                    grad_digit_weight = (xw_c * grad_output.unsqueeze(4)).sum(dim=(0, 1, 2, 3))
                    grad_output = grad_output.unsqueeze(2).unsqueeze(5).unsqueeze(6)
                    grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                    grad_w_l = (grad_output * x_l).sum((0, 3, 4), keepdim=True)
                else:
                    raise ValueError('todo')
                return grad_x_l, grad_w_l, grad_digit_weight.view(1, -1) * digit_weight_mask

        return _pq().apply
    elif model is not None:
        pass
    else:
        if by_code:
            # my idea require very large memory during training and inference,
            # so try to solve a mini batch each time to save memory but require more time
            class _pq(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x_l, w_l, digit_weight):
                    scale_w = 2 ** (bit - 1) - 1
                    # scale_x = 2 ** bit - 1
                    scale_x = 2 ** (bit - 1) - 1
                    # x_c = (x_l * scale_x).int() * (2 ** bit)
                    x_c = (x_l * scale_x).to(torch.int32) + (2 ** (bit - 1))
                    w_c = (w_l * scale_w).to(torch.int32) + (2 ** (bit - 1))
                    # print((x_l * scale_x).to(torch.int32))
                    # print(x_c.shape, w_c.shape)
                    batch_size = x_c.size(0)
                    channel_size = w_c.size(1)
                    if x_c.ndim == w_c.ndim == 3:
                        xw_c = []
                        end_batch = 0
                        for i in range(0, batch_size, mini_batch_size):
                            end_batch = i + mini_batch_size
                            xw_c.append(approx_product_code[x_c[i:end_batch], w_c].sum(2))
                        if end_batch < batch_size:
                            xw_c.append(approx_product_code[x_c[end_batch:], w_c].sum(2))
                        xw_c = torch.cat(xw_c, dim=0)
                        # xw_c = approx_product_code[x_c, w_c].sum(2)
                        # print(xw_c.shape)
                        ctx.save_for_backward(x_l, w_l, xw_c)
                        return (xw_c * digit_weight).sum(2)
                    elif x_c.ndim == w_c.ndim == 7:
                        xw_c = []
                        end_batch = 0
                        for i in range(0, batch_size, mini_batch_size):
                            end_batch = i + mini_batch_size
                            xw_sub_c = []
                            end_channel = 0
                            for j in range(0, channel_size, mini_channels):
                                end_channel = j + mini_channels
                                xw_sub_c.append(approx_product_code[x_c[i:end_batch], w_c[:, j:end_channel]].sum(dim=(5, 6, 2)))
                            if end_channel < channel_size:
                                xw_sub_c.append(approx_product_code[x_c[i:end_batch], w_c[:, end_channel:]].sum(dim=(5, 6, 2)))
                            xw_c.append(torch.cat(xw_sub_c, dim=1))
                            # xw_c.append(approx_product_code[x_c[i:end_point], w_c[i:end_point], :].sum(dim=(5, 6, 2)))
                        # print(xw_sub_c[0].shape, xw_c[0].shape)
                        if end_batch < batch_size:
                            xw_sub_c = []
                            end_channel = 0
                            for j in range(0, channel_size, mini_channels):
                                end_channel = j + mini_channels
                                xw_sub_c.append(approx_product_code[x_c[end_batch:], w_c[:, j:end_channel]].sum(dim=(5, 6, 2)))
                            if end_channel < channel_size:
                                xw_sub_c.append(approx_product_code[x_c[end_batch:], w_c[:, end_channel:]].sum(dim=(5, 6, 2)))
                            xw_c.append(torch.cat(xw_sub_c, dim=1))
                            # xw_c.append(approx_product_code[x_c[end_batch:], w_c[end_batch:], :].sum(dim=(5, 6, 2)))
                        xw_c = torch.cat(xw_c, dim=0)
                        # print(xw_sub_c[0].shape, xw_c.shape)
                        # xw_c = approx_product_code[x_c, w_c, :].sum(dim=(5, 6, 2))
                        ctx.save_for_backward(x_l, w_l, xw_c)
                        return (xw_c * digit_weight).sum(4)

                @staticmethod
                def backward(ctx, grad_output):
                    # pdb.set_trace()
                    x_l, w_l, xw_c = ctx.saved_tensors
                    batch_size = x_l.size(0)
                    # channel_size = w_l.size(1)
                    # print(grad_output.shape, x_l.shape, w_l.shape, xw_c.shape)
                    if x_l.ndim == w_l.ndim == 3:
                        grad_output = grad_output.unsqueeze(2)
                        grad_digit_weight = (xw_c * grad_output).sum(dim=(0, 1))
                        grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                        grad_w_l = (grad_output * x_l).sum(0, keepdim=True)
                    elif x_l.ndim == w_l.ndim == 7:
                        grad_digit_weight = (xw_c * grad_output.unsqueeze(4)).sum(dim=(0, 1, 2, 3))
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
                        # grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                        # grad_w_l = (grad_output * x_l).sum((0, 3, 4), keepdim=True)
                    else:
                        raise ValueError('todo')
                    return grad_x_l, grad_w_l, grad_digit_weight.view(1, -1) * digit_weight_mask
            return _pq().apply
        else:
            # my idea require very large memory during training and inference,
            # so try to solve a mini batch each time to save memory but require more time
            class _pq(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x_l, w_l, digit_weight):
                    scale_w = 2 ** (bit - 1) - 1
                    scale_x = 2 ** (bit - 1) - 1
                    x_c = (x_l * scale_x).to(torch.int32) + (2 ** (bit - 1))
                    w_c = (w_l * scale_w).to(torch.int32) + (2 ** (bit - 1))
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
                        # print(xw_c.shape)
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
                            # xw_c.append(approx_product_code[x_c[i:end_point], w_c[i:end_point], :].sum(dim=(5, 6, 2)))
                        # print(xw_sub_c[0].shape, xw_c[0].shape)
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
                            # xw_c.append(approx_product_code[x_c[end_batch:], w_c[end_batch:], :].sum(dim=(5, 6, 2)))
                        xw_c = torch.cat(xw_c, dim=0)
                        # print(xw_sub_c[0].shape, xw_c.shape)
                        # xw_c = approx_product_value[x_c, w_c].sum(dim=(5, 6, 2))
                        ctx.save_for_backward(x_l, w_l, xw_c)
                        return xw_c

                @staticmethod
                def backward(ctx, grad_output):
                    # pdb.set_trace()
                    x_l, w_l, xw_c = ctx.saved_tensors
                    batch_size = x_l.size(0)
                    # channel_size = w_l.size(1)
                    # print(grad_output.shape, x_l.shape, w_l.shape, xw_c.shape)
                    if x_l.ndim == w_l.ndim == 3:
                        grad_output = grad_output.unsqueeze(2)
                        # grad_digit_weight = (xw_c * grad_output).sum(dim=(0, 1))
                        grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                        grad_w_l = (grad_output * x_l).sum(0, keepdim=True)
                    elif x_l.ndim == w_l.ndim == 7:
                        # grad_digit_weight = (xw_c * grad_output.unsqueeze(4)).sum(dim=(0, 1, 2, 3))
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
                        # grad_x_l = (grad_output * w_l).sum(1, keepdim=True)
                        # grad_w_l = (grad_output * x_l).sum((0, 3, 4), keepdim=True)
                    else:
                        raise ValueError('todo')
                    return grad_x_l, grad_w_l, None

            return _pq().apply



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


class Encoder(nn.Module):
    def __init__(self, product_bit, bit=8, ):
        super(Encoder, self).__init__()
        self.fc1 = torch.nn.Linear(2, 100)
        self.fc2 = torch.nn.Linear(100, 1000)
        self.fc3 = torch.nn.Linear(1000, product_bit)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, w):
        out = torch.concatenate((x, w), dim=1)
        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


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



# class quantization_fn(nn.Module):
#     def __init__(self, bit, signed=True, norm=False, alpha_init = torch.tensor(3.0)):
#         super(quantization_fn, self).__init__()
#         assert (bit <= 8 and bit > 0) or bit == 32
#         self.bit = bit
#         self.norm = norm
#         self.grids = build_uniform_value(bit=self.bit, signed=signed)
#         self.x_q = quantization(grids=self.grids, signed=signed)
#         self.register_parameter('alpha', Parameter(alpha_init))
#
#     def forward(self, x):
#         if self.bit == 32:
#             x_q = x
#         else:
#             if self.norm:
#                 mean = x.data.mean()
#                 std = x.data.std()
#                 x = x.add(-mean).div(std)      # weights normalization
#             x_q = self.x_q(x, self.alpha)
#         return x_q
#
#
# def quantization(grids, signed=True):
#
#     def uniform_quant(x, value_s):
#         shape = x.shape
#         xhard = x.view(-1)
#         value_s = value_s.type_as(x)
#         idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
#         xhard = value_s[idxs].view(shape)
#         return xhard
#
#     class _pq(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, input, alpha):
#             input = input.div(alpha)                          # weights are first divided by alpha
#             if signed:
#                 input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
#             else:
#                 input_c = input.clamp(min= 0, max=1)       # then clipped to [ 0,1]
#             input_q = uniform_quant(input_c, grids)
#             ctx.save_for_backward(input, input_q)
#             input_q = input_q.mul(alpha)               # rescale to the original range
#             return input_q
#
#         @staticmethod
#         def backward(ctx, grad_output):
#             grad_input = grad_output.clone()             # grad for weights will not be clipped
#             input, input_q = ctx.saved_tensors
#             if signed:
#                 i = (input.abs() > 1.).float()
#                 sign = input.sign()
#                 grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
#             else:
#                 i = (input > 1.).float()
#                 grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
#                 grad_input = grad_input * (1 - i)       # grad for activation will be clipped
#             return grad_input, grad_alpha
#
#     return _pq().apply
#
#
# def build_uniform_value(bit=4, signed=True):
#     if signed:  # bit=3, [-8 ~ 7]/7
#         values = torch.arange(2 ** bit).sub(2 ** (bit - 1)).div(2 ** (bit - 1) - 1)
#     else:    # bit=3, [0 ~ 15]/15
#         values = torch.arange(2 ** bit).div(2 ** bit - 1)
#     return values


def get_conv_params(m):
    (in_channels, out_channels, kernel_size,
     stride, padding, weights, biases) = (m.in_channels, m.out_channels, m.kernel_size,
                                          m.stride, m.padding, m.weight, m.bias)
    bias = False if biases is None else True
    if m.dilation != (1, 1) or m.groups != 1:
        raise ValueError('dilation > 1 or groups > 1 nor supported')
    return (in_channels, out_channels, kernel_size,
            stride, padding, weights, biases, bias)

def get_fc_params(m):
    (in_features, out_features, weights, biases) = (m.in_features, m.out_features, m.weight, m.bias)
    bias = False if biases is None else True
    return (in_features, out_features, weights, biases, bias)

def replace_conv_fc(m, args):
    for k in m._modules.keys():
        if m._modules[k]._modules.keys().__len__() == 0:
            if isinstance(m._modules[k], torch.nn.Conv2d):
                (in_channels, out_channels, kernel_size,
                 stride, padding, weights, biases, bias) = get_conv_params(m._modules[k])
                if args.mini_channels == 0 or args.mini_channels > out_channels:
                    mini_channels = out_channels
                else:
                    mini_channels = args.mini_channels
                m._modules[k] = QuantConv2d(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=1, groups=1, bias=bias,
                                            weights=weights, biases=biases,
                                            bit=args.bit, product_bit=args.product_bit,
                                            mini_batch_size=args.mini_batch_size,
                                            mini_channels=mini_channels,
                                            # approx_product=args.approx_product,
                                            approx_product_code=args.approx_product_code,
                                            approx_product_value=args.approx_product_value,
                                            digit_weight=args.digit_weight,
                                            digit_weight_mask=args.digit_weight_mask,
                                            by_code=args.by_code,
                                            model=args.model,
                                            OP=args.OP
                                            )

            elif isinstance(m._modules[k], torch.nn.Linear):
                (in_features, out_features, weights, biases, bias) = get_fc_params(m._modules[k])
                m._modules[k] = QuantFC(in_features, out_features,
                                        bias=bias, weights=weights, biases=biases,
                                        bit=args.bit, product_bit=args.product_bit,
                                        mini_batch_size=args.mini_batch_size,
                                        mini_channels=0,
                                        # approx_product=args.approx_product,
                                        approx_product_code=args.approx_product_code,
                                        approx_product_value=args.approx_product_value,
                                        digit_weight=args.digit_weight,
                                        digit_weight_mask=args.digit_weight_mask,
                                        by_code=args.by_code,
                                        model=args.model,
                                        OP=args.OP
                                        )
        else:
            m._modules[k] = replace_conv_fc(m=m._modules[k], args=args)
    return m

