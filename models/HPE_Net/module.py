'''
Detnet  based on PyTorch
paper reference:
code: https://github.com/MengHao666/Minimal-Hand-pytorch
'''
import sys

import torch

# sys.path.append("./")
from torch import nn
from einops import rearrange, repeat
from models.HPE_Net.resnet_helper import resnet50, conv3x3, resnet101
# from resnet_helper import resnet50, conv3x3
import numpy as np
import utils
import os.path as osp
import torch.distributed as dist
import models
import torch.backends.cudnn as cudnn


# my modification
def get_pose_tile_torch(N):
    pos_tile = np.expand_dims(
        np.stack(
            [
                np.tile(np.linspace(-1, 1, 64).reshape([1, 64]), [64, 1]),
                np.tile(np.linspace(-1, 1, 64).reshape([64, 1]), [1, 64])
            ], -1
        ), 0
    )
    pos_tile = np.tile(pos_tile, (N, 1, 1, 1))
    retv = torch.from_numpy(pos_tile).float()
    return rearrange(retv, 'b h w c -> b c h w')


class net_2d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU(inplace=True))

        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x).sigmoid()
        return x


class net_3d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21, need_norm=False):
        super().__init__()
        self.need_norm = need_norm
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU(inplace=True))
        self.prediction = nn.Conv2d(output_features, joints * 3, 1, 1, 0)

    def forward(self, x):
        x = self.prediction(self.project(x))

        dmap = rearrange(x, 'b (j l) h w -> b j l h w', l=3)

        return dmap


class DetNet(nn.Module):
    def __init__(self, stacks=1):
        super().__init__()
        self.resnet = resnet50()

        self.hmap_0 = net_2d(258, 256, 1)
        self.dmap_0 = net_3d(279, 256, 1)
        self.lmap_0 = net_3d(342, 256, 1)
        self.stacks = stacks

    def forward(self, x):
        features = self.resnet(x)

        device = x.device
        pos_tile = get_pose_tile_torch(features.shape[0]).to(device)
        x = torch.cat([features, pos_tile], dim=1)

        hmaps = []
        dmaps = []
        lmaps = []

        for _ in range(self.stacks):
            heat_map = self.hmap_0(x)
            hmaps.append(heat_map)
            x = torch.cat([x, heat_map], dim=1)

            dmap = self.dmap_0(x)
            dmaps.append(dmap)

            x = torch.cat([x, rearrange(dmap, 'b j l h w -> b (j l) h w')], dim=1)

            lmap = self.lmap_0(x)
            lmaps.append(lmap)
        hmap, dmap, lmap = hmaps[-1], dmaps[-1], lmaps[-1]

        uv, argmax = self.map_to_uv(hmap)

        delta = self.dmap_to_delta(dmap, argmax)
        xyz = self.lmap_to_xyz(lmap, argmax)

        det_result = {
            "h_map": hmap,
            "d_map": dmap,
            "l_map": lmap,
            "delta": delta,
            "xyz": xyz,
            "uv": uv
        }
        return det_result

    @property
    def pos(self):
        return self.__pos_tile

    @staticmethod
    def map_to_uv(hmap):
        b, j, h, w = hmap.shape
        hmap = rearrange(hmap, 'b j h w -> b j (h w)')
        argmax = torch.argmax(hmap, -1, keepdim=True)
        u = argmax // w
        v = argmax % w
        uv = torch.cat([u, v], dim=-1)

        return uv, argmax

    @staticmethod
    def dmap_to_delta(dmap, argmax):
        return DetNet.lmap_to_xyz(dmap, argmax)

    @staticmethod
    def lmap_to_xyz(lmap, argmax):
        lmap = rearrange(lmap, 'b j l h w -> b j (h w) l')
        index = repeat(argmax, 'b j i -> b j i c', c=3)
        xyz = torch.gather(lmap, dim=2, index=index).squeeze(2)
        return xyz


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


class HPE_Net(nn.Module):

    def __init__(self, params, load_pretrain=None, dist_model=False, demo=False):
        super(HPE_Net, self).__init__()
        # model
        self.model = DetNet(stacks=params['hpe_stacks'])
        # if load_pretrain is not None:
        #     assert load_pretrain.endswith('.pth'), "load_pretrain should end with .pth"
        #     utils.load_weights(load_pretrain, self.model)
        #
        # self.model.cuda()
        # self.params = params
        #
        # if params['initilize'] is not None and not demo:
        #     if params['initilize'] == "interhand_offfical":
        #         print("interhand_offfical_initialize")
        #         self.model.apply(init_weights)
        #     elif params['initilize'] == "kaiminghe":
        #         print("kaiminghe_initialize")
        #         for m in self.model.modules():
        #             if isinstance(m, torch.nn.Conv2d):
        #                 torch.nn.init.kaiming_normal_(m.weight)
        #     else:
        #         assert 0, "Wrong model initialization method"
        #
        # if dist_model:
        #     self.model = utils.DistModule(self.model)
        #     self.world_size = dist.get_world_size()
        # else:
        #     self.model = models.FixModule(self.model)
        #     self.world_size = 1
        #
        # self.demo = demo
        # if demo:
        #     return
        #
        # # optim
        # self.optim = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()), lr=params['lr'])
        #
        # # loss
        # self.heatmapLoss = models.HeatmapLoss()
        # self.deltamapLoss = models.DeltamapLoss()
        # self.locationmapLoss = models.LocationmapLoss()
        #
        # cudnn.benchmark = True

    def set_input(self, inputs, targets, meta_info):

        self.inputs = inputs
        self.targets = targets
        self.meta_info = meta_info

        self.rgb = inputs["rgb"].cuda()

        if targets is not None:
            self.targets_joint_img = targets['joint_img'].cuda()
            self.targets_joint_cam = targets['joint_cam'].cuda()

            self.target_hm = targets['heat_map'].cuda()
            self.target_dm = targets['delta_map'].cuda()
            self.target_lm = targets['location_map'].cuda()

        self.meta_info = {}
        for k, v in meta_info.items():
            if k not in ("jdx", "img_path"):
                self.meta_info[k] = v.cuda()
            else:
                self.meta_info[k] = v

    def forward_only(self):

        with torch.no_grad():
            det_result = self.model(self.rgb)

        # InterHand2.6M  saves ['joint_coord'],['rel_root_depth'],['hand_type'],['inv_trans']

        out = {'joint_cam': det_result['xyz'].cpu().numpy().astype(np.float32),
               'joint_img': det_result['uv'].cpu().numpy().astype(np.int32),
               # 'h_map': det_result['h_map'].cpu().numpy(),

               }

        if "jdx" in self.meta_info.keys():
            out["jdx"] = self.meta_info["jdx"].cpu().numpy().astype(np.int32),
        if "img_path" in self.meta_info.keys():
            out["img_path"]= self.meta_info['img_path']
        if 'inv_trans' in self.meta_info:
            out['inv_trans'] = self.meta_info['inv_trans'].cpu().numpy().astype(np.float32)
        return out

    def step(self):

        output = self.model(self.rgb)

        loss_dict = {}
        loss_dict["hm"] = self.heatmapLoss(output["h_map"], self.target_hm, self.meta_info)
        loss_dict["dm"] = self.deltamapLoss(output["d_map"], self.target_dm, self.target_hm, self.meta_info)
        loss_dict["lm"] = self.locationmapLoss(output["l_map"], self.target_lm, self.target_hm, self.meta_info)

        for k in loss_dict.keys():
            loss_dict[k] /= self.world_size

        total_loss = 0.
        for key, coef in self.params['lambda_dict'].items():
            value = coef * loss_dict[key]
            total_loss += value

        # update
        self.optim.zero_grad()
        total_loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()

        return loss_dict

    def load_model_demo(self, path):
        utils.load_state(path, self.model)

    def load_state(self, root, Iter, resume=False, clear_module=False):
        path = osp.join(root, "ckpt_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
        else:
            utils.load_state(path, self.model, clear_module=clear_module)

    def save_state(self, root, Iter):
        path = osp.join(root, "ckpt_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()


if __name__ == '__main__':
    mydet = DetNet()

    img_crop = torch.randn(10, 3, 256, 256)
    res = mydet(img_crop)

    hmap = res["h_map"]
    dmap = res["d_map"]
    lmap = res["l_map"]
    delta = res["delta"]
    xyz = res["xyz"]
    uv = res["uv"]

    print("hmap.shape=", hmap.shape)
    print("dmap.shape=", dmap.shape)
    print("lmap.shape=", lmap.shape)
    print("delta.shape=", delta.shape)
    print("xyz.shape=", xyz.shape)
    print("uv.shape=", uv.shape)
