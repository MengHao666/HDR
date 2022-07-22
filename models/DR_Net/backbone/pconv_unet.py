import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from models.DR_Net.backbone.pvt_v2 import Block

__all__ = ['PConvUNet', 'VGG16FeatureExtractor']


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


def th_upsample2x(x, mode='nearest'):
    x = F.interpolate(x, scale_factor=2, mode=mode)
    return x


def MY_SIMPLE_Embed(x):
    _, _, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)
    return x, H, W


class Transformer(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            # patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            # x, H, W = patch_embed(x)
            # print("x.shape=", x.shape)
            x, H, W = MY_SIMPLE_Embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            # print("x.shape=", x.shape)
            # if i != self.num_stages - 1:
            #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # print("*x.shape=", x.shape)

        # return x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # print("*x.shape=", x.shape)
        # x = self.head(x)

        return x


class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        self.enc_3_transformer = Transformer(embed_dims=[256], num_heads=[2], mlp_ratios=[8], qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1], sr_ratios=[4],
                                             num_stages=1)

        self.enc_5_transformer = Transformer(embed_dims=[512], num_heads=[8], mlp_ratios=[4], qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1], sr_ratios=[1],
                                             num_stages=1)

        self.dec_5_transformer = Transformer(embed_dims=[512], num_heads=[8], mlp_ratios=[4], qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1], sr_ratios=[1],
                                             num_stages=1)
        self.dec_3_transformer = Transformer(embed_dims=[128], num_heads=[2], mlp_ratios=[8], qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1], sr_ratios=[4],
                                             num_stages=1)

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, 3,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])

            ####non-symmetric transformer######
            # if i in (3, 5):
            #     print("h_dict[{}].shape={}, h_mask_dict[{}].shape={}".format(h_key, h_dict[h_key].shape, h_key,
            #                                                                  h_mask_dict[h_key].shape))

            if i == 3:
                # print("h_dict[{}].shape={}".format(h_key, h_dict[h_key].shape))
                h_dict[h_key] = self.enc_3_transformer(h_dict[h_key])
                # print("*h_dict[{}].shape={}".format(h_key, h_dict[h_key].shape))

            if i == 5:
                # print("h_dict[{}].shape={}".format(h_key, h_dict[h_key].shape))
                h_dict[h_key] = self.enc_5_transformer(h_dict[h_key])

            # if i in (3, 5):
            #     print("*h_dict[{}].shape={}, h_mask_dict[{}].shape={}".format(h_key, h_dict[h_key].shape, h_key,
            #                                                                   h_mask_dict[h_key].shape))
            ########

            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

            ####non-symmetric transformer######
            # if i in (3, 5):
            #     print("{} h.shape={}, h_mask.shape={}".format(i, h.shape, h_mask.shape))
            if i == 5:
                h = self.dec_5_transformer(h)
            if i == 3:
                h = self.dec_3_transformer(h)

            # if i in (3, 5):
            #     print("*{} h.shape={}, h_mask.shape={}".format(i, h.shape, h_mask.shape))
            ###################################

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


# if __name__ == '__main__':
#     size = (1, 3, 5, 5)
#     input = torch.ones(size)
#     input_mask = torch.ones(size)
#     input_mask[:, :, 2:, :][:, :, :, 2:] = 0
#
#     conv = PartialConv(3, 3, 3, 1, 1)
#     l1 = nn.L1Loss()
#     input.requires_grad = True
#
#     output, output_mask = conv(input, input_mask)
#     loss = l1(output, torch.randn(1, 3, 5, 5))
#     loss.backward()
#
#     assert (torch.sum(input.grad != input.grad).item() == 0)
#     assert (torch.sum(torch.isnan(conv.input_conv.weight.grad)).item() == 0)
#     assert (torch.sum(torch.isnan(conv.input_conv.bias.grad)).item() == 0)
#
#     # model = PConvUNet()
#     # output, output_mask = model(input, input_mask)

if __name__ == '__main__':
    import utils

    # params = {"input_channels": 8,"layer_size": 7}
    params = {"input_channels": 8, "layer_size": 8}

    model = PConvUNet(**params)
    utils.load_weights("../../pretrains/partialconv_input_ch8.pth", model)

    input = torch.normal(0, 1, size=(2, 8, 256, 256))
    input_mask = torch.normal(0, 1, size=(2, 8, 256, 256))
    gt = torch.normal(0, 1, size=(2, 3, 256, 256))

    l1 = nn.L1Loss()
    input.requires_grad = True

    res, res_mask = model(input, input_mask)
    # print(res.shape)
    # print(res_mask.shape)

    loss = l1(res, gt)
    loss.backward()
