# -*- coding:utf-8 -*-
# @Author  :   Multimodal Perception and Intelligent Analysis
# @Contact :   rzw@njust.edu.cn
import torch
import torch.nn as nn
import numpy as np
from network.moudle_transformer import cross_encoder, PatchEmbed_tensor, Recons_tensor
from network.vgg import VGG
from tools import utils

EPSILON = 1e-6

# dense conv
class Dense_ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.convd = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.convd(out)
        out = torch.cat((x, out), 1)
        return out

# dense block
class Dense_Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size, stride, dense_out):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.add_module('dense_conv' + str(i),
                            Dense_ConvLayer(in_channels + i * out_channels, out_channels, kernel_size, stride))
        self.adjust_conv = ConvLayer(in_channels + num_layers * out_channels, dense_out, kernel_size, stride)

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            dense_conv = getattr(self, 'dense_conv' + str(i))
            out = dense_conv(out)
        out = self.adjust_conv(out)
        return out

# encoder network, extract features
class Encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_layers, dense_out):
        super().__init__()
        self.kernel_size = 3
        self.stride = 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = ConvLayer(in_channels, mid_channels, self.kernel_size, self.stride)

        self.dense_blocks = nn.Sequential(
            Dense_Block(num_layers, mid_channels, out_channels, self.kernel_size, self.stride, dense_out),
            nn.MaxPool2d(2, 2),
            Dense_Block(num_layers, dense_out, out_channels, self.kernel_size, self.stride, dense_out),
            nn.MaxPool2d(2, 2),
            Dense_Block(num_layers, dense_out, out_channels, self.kernel_size, self.stride, dense_out),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.pool(out1)
        out = self.dense_blocks(out)
        return out1,out

class TwoLayerConvNet(nn.Module):
    def __init__(self):
        super(TwoLayerConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x):
        x = x/255
        x1 = self.conv1(x)
        x = self.pool(torch.relu(x1))  # Output: 4 x 16 x 128 x 128
        x = self.pool(torch.relu(self.conv2(x)))  # Output: 4 x 32 x 64 x 64
        x = self.pool(torch.relu(self.conv3(x)))  # Output: 4 x 64 x 32 x 32
        x = self.pool(x)
        return x1,x

def remove_mean(x):
    (b, ch, h, w) = x.size()

    tensor = x.reshape(b, ch, h * w)
    t_mean = torch.mean(tensor, 2)

    t_mean = t_mean.view(b, ch, 1, 1)
    t_mean = t_mean.repeat(1, 1, h, w)

    out = x - t_mean
    return out, t_mean

class Weight(nn.Module):
    def __init__(self, ch_s, ks_s, ch_d, ks_d):
        super(Weight, self).__init__()
        # weight for features
        weight_sh = torch.ones([1, 1, ks_s, ks_s])
        reflection_padding_sh = int(np.floor(ks_s / 2))
        self.conv_sh = torch.nn.Conv2d(ch_s, ch_s, (ks_s, ks_s), stride=1, padding=reflection_padding_sh, bias=False)
        self.conv_sh.weight.data = (1 / (ks_s * ks_s)) * weight_sh.repeat(ch_s, ch_s, 1, 1).float()
        self.conv_sh.requires_grad_(False)

        weight_de = torch.ones([1, 1, ks_d, ks_d])
        reflection_padding_de = int(np.floor(ks_d / 2))
        self.conv_de = torch.nn.Conv2d(ch_d, ch_d, (ks_d, ks_d), stride=1, padding=reflection_padding_de, bias=False)
        self.conv_de.weight.data = (1 / (ks_d * ks_d)) * weight_de.repeat(ch_d, ch_d, 1, 1).float()
        self.conv_de.requires_grad_(False)

    def for_sh(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        g_x = torch.sqrt((x - self.conv_sh(x)) ** 2)
        g_y = torch.sqrt((y - self.conv_sh(y)) ** 2)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)
        w_x = w_x.detach()
        w_y = w_y.detach()
        return w_x, w_y

# Convolution operation
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# decoder network for final fusion
class Decoder_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, train_flag=False):
        super().__init__()
        self.kernel_size = 3
        self.stride = 1
        self.train_flag = train_flag

        self.up = nn.Upsample(scale_factor=2)
        self.shape_adjust = utils.UpsampleReshape_eval()
        self.conv1 = ConvLayer(in_channels, int(in_channels / 2), self.kernel_size, self.stride)
        self.conv_block = nn.Sequential(
            ConvLayer(int(in_channels / 2), int(in_channels / 2), self.kernel_size, self.stride),
            nn.Upsample(scale_factor=2),
            ConvLayer(int(in_channels / 2), int(in_channels / 4), self.kernel_size, self.stride),
            nn.Upsample(scale_factor=2),
            ConvLayer(int(in_channels / 4), int(in_channels / 4), self.kernel_size, self.stride),
            nn.Upsample(scale_factor=2),
            # ConvLayer(int(in_channels / 4), out_channels, self.kernel_size, self.stride)
        )
        last_ch = int(in_channels / 4)
        self.conv_last = ConvLayer(last_ch, out_channels, 1, self.stride)

        ks_s = 3
        ks_d = 5
        self.weight = Weight(last_ch, ks_s, in_channels, ks_d)
    def forward(self, ir_sh, vi_sh, x1):
        out = x1
        out = self.up(self.conv1(out))
        out = self.conv_block(out)

        out = self.shape_adjust(ir_sh, out)
        ws = self.weight.for_sh(ir_sh, vi_sh)  # gradient
        out = out +  0.5 * ws[0] * ir_sh + ws[1] * vi_sh
        out = self.conv_last(out)
        return out

class FuseNet(nn.Module):
    def __init__(self, img_size, patch_size, en_out_channels1, out_channels, part_out, train_flag, 
                 depth_self, depth_cross, n_heads, mlp_ratio, qkv_bias, p, attn_p):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.part_out = part_out
        self.patch_embed_tensor = PatchEmbed_tensor(img_size)
        self.recons_tensor = Recons_tensor(img_size)
        self.embed_dim = part_out * patch_size * patch_size
        self.num_patches = int(img_size / patch_size) * int(img_size / patch_size)
        self.autoencoder1 = TwoLayerConvNet()
        self.autoencoder2 = TwoLayerConvNet()
        self.cross_atten_block = cross_encoder(self.img_size, self.patch_size, self.embed_dim, self.num_patches, depth_self,
                                               depth_cross, n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.conv_gra = torch.nn.Conv2d(en_out_channels1, en_out_channels1, (3, 3), stride=1, padding=1,bias=False)
        weight = torch.from_numpy(np.array([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]))
        self.conv_gra.weight.data = weight.repeat(en_out_channels1, en_out_channels1, 1, 1).float()
        self.decoder_fusion = Decoder_fusion(part_out, out_channels, train_flag)
        self.vgg = VGG()

    def forward(self, ir, vi, shift_flag):
        ir_sh, ir_de = self.autoencoder1(ir)
        vi_sh, vi_de = self.autoencoder2(vi)

        c_f, ir_self, vi_self, vi_cross, ir_cross = self.cross_atten_block(ir_de, vi_de, shift_flag)
        in_put = c_f
        # -----------------------------------
        # decoder fusion
        out = self.decoder_fusion(ir_sh, vi_sh, in_put)
        out = utils.normalize_tensor(out)
        out = out * 255
        # -----------------------------------
        outputs = {'out': out, 'ir_self': ir_self, 'vi_self': vi_self, 'fuse_cross': c_f}

        return outputs

    # training phase
    def train_module(self, x_ir, x_vi, shift_flag, gra_loss, order_loss):
        ir_sh, ir_de = self.autoencoder1(x_ir)
        vi_sh, vi_de = self.autoencoder2(x_vi)
        out_f, ir_self, vi_self, vi_cross, ir_cross = self.cross_atten_block(ir_de, vi_de, shift_flag)
        in_put = out_f
        # -----------------------------------
        # decoder fusion
        out = self.decoder_fusion(ir_sh, vi_sh, in_put)
        out = utils.normalize_tensor(out)
        out = out * 255
        # -----------------------------------
        loss_pix, temp = order_loss(out, x_ir, x_vi)
        loss_vgg = self.vgg.vgg_loss(out, x_ir, x_vi)
        loss_gra, gp, gxir, gxvi, g_target = gra_loss(out, x_ir, x_vi)
        # -----------------------------------
        w = [1.0, 0.1, 10.0] # weight
        total_loss = w[0] * loss_pix + w[1] * loss_vgg + w[2] * loss_gra
        # -----------------------------------
        outputs = {'out': out,
                   'pix_loss': w[0] * loss_pix,
                   'vgg_loss': w[1] * loss_vgg,
                   'gra_loss': w[2] * loss_gra,
                   'total_loss': total_loss}
        return outputs

