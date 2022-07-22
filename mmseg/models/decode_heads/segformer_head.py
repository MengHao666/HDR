# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..losses import accuracy


@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # self.fusion_conv = ConvModule(
        #     in_channels=self.channels * num_inputs,
        #     out_channels=self.channels,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg)
        self.fusion_conv1 = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv2 = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv3 = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv4 = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.conv_seg1 = nn.Conv2d(self.channels, 2, kernel_size=1) # classify 2 categories
        if self.dropout_ratio > 0:
            self.dropout1 = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout1 = None

        self.conv_seg2 = nn.Conv2d(self.channels, 2, kernel_size=1)  # classify 2 categories
        if self.dropout_ratio > 0:
            self.dropout2 = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout2 = None

        self.conv_seg3 = nn.Conv2d(self.channels, 2, kernel_size=1)  # classify 2 categories
        if self.dropout_ratio > 0:
            self.dropout3 = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout3 = None

        self.conv_seg4 = nn.Conv2d(self.channels, 2, kernel_size=1)  # classify 2 categories
        if self.dropout_ratio > 0:
            self.dropout4 = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout4 = None



    # def forward(self, inputs):
    #     # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
    #     inputs = self._transform_inputs(inputs)
    #     outs = []
    #     for idx in range(len(inputs)):
    #         x = inputs[idx]
    #         conv = self.convs[idx]
    #         outs.append(
    #             resize(
    #                 input=conv(x),
    #                 size=inputs[0].shape[2:],
    #                 mode=self.interpolate_mode,
    #                 align_corners=self.align_corners))
    #
    #     out = self.fusion_conv(torch.cat(outs, dim=1))
    #
    #     out = self.cls_seg(out)
    #
    #     return out

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out1 = self.fusion_conv1(torch.cat(outs, dim=1))
        if self.dropout1 is not None:
            out1 = self.dropout1(out1)
        out1 = self.conv_seg1(out1)

        out2 = self.fusion_conv2(torch.cat(outs, dim=1))
        if self.dropout2 is not None:
            out2 = self.dropout2(out2)
        out2 = self.conv_seg2(out2)

        out3 = self.fusion_conv3(torch.cat(outs, dim=1))
        if self.dropout3 is not None:
            out3 = self.dropout3(out3)
        out3 = self.conv_seg3(out3)

        out4 = self.fusion_conv4(torch.cat(outs, dim=1))
        if self.dropout4 is not None:
            out4 = self.dropout4(out4)
        out4 = self.conv_seg4(out4)

        # print("out1.shape=",out1.shape)
        # print("out2.shape=",out2.shape)
        # print("out3.shape=",out3.shape)
        # print("out4.shape=",out4.shape)

        return [out1, out2, out3, out4]

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()

        for i in range(len(seg_logit)):
            _seg_logit=seg_logit[i]
            _seg_label=seg_label[i]

            _seg_logit = resize(
                input=_seg_logit,
                size=_seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            if self.sampler is not None:
                seg_weight = self.sampler.sample(_seg_logit, _seg_label)
            else:
                seg_weight = None
            _seg_label = _seg_label.squeeze(1)
            for loss_decode in self.loss_decode:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name+f'_{i}'] = loss_decode(
                        _seg_logit,
                        _seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name+f'_{i}'] += loss_decode(
                        _seg_logit,
                        _seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

            loss[f'acc_seg_{i}'] = accuracy(_seg_logit, _seg_label)


        return loss

