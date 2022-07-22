# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math


class HeatmapLoss(nn.Module):
    def __ini__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred_hm, targ_hm, meta_info):
        valid = meta_info['heatmap_vaild']  # (B,21)
        if valid.sum():
            loss = torch.norm((pred_hm - targ_hm) * valid[:, :, None, None])
            loss = loss / valid.sum()
        else:
            loss = torch.Tensor([0]).cuda()
        return loss


class DeltamapLoss(nn.Module):
    def __ini__(self):
        super(DeltamapLoss, self).__init__()

    def forward(self, pred_dm, targ_dm, targ_hm, meta_info):
        valid = meta_info['delta_valid'] * meta_info['heatmap_vaild']  # (B,21)

        if valid.sum():
            valid_tile = valid[:, :, None, None, None].repeat(1, 1, 3, 64, 64)  # (B*21*3*64*64)
            targ_hm_tile = targ_hm[:, :, None, :, :].repeat(1, 1, 3, 1, 1)  # (B*21*3*64*64)

            loss = torch.norm((pred_dm - targ_dm) * targ_hm_tile * valid_tile)
            loss = loss / valid.sum()
        else:
            loss = torch.Tensor([0]).cuda()
        return loss


class LocationmapLoss(nn.Module):
    def __ini__(self):
        super(LocationmapLoss, self).__init__()

    def forward(self, pred_lm, targ_lm, targ_hm, meta_info):
        valid = meta_info['joint_cam_valid'] * meta_info['heatmap_vaild']

        if valid.sum():
            valid_tile = valid[:, :, None, None, None].repeat(1, 1, 3, 64, 64)  # (B*21*3*64*64)
            targ_hm_tile = targ_hm[:, :, None, :, :].repeat(1, 1, 3, 1, 1)  # (B*21*3*64*64)

            loss = torch.norm((pred_lm - targ_lm) * targ_hm_tile * valid_tile)
            loss = loss / valid.sum()
        else:
            loss = torch.Tensor([0]).cuda()
        return loss
