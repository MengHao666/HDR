import cv2
import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
# from . import backbone, InpaintingLoss, AdversarialLoss
from models.DR_Net import backbone, InpaintingLoss, AdversarialLoss


class PartialCompletionContentCGAN(nn.Module):
    def __init__(self, params, load_pretrain=None, dist_model=False, demo=False):
        super(PartialCompletionContentCGAN, self).__init__()
        self.params = params
        self.with_modal = params.get('with_modal', False)
        self.demo=demo

        # model
        self.model = backbone.__dict__[params['backbone_arch']](**params['backbone_param'])
        # if load_pretrain is not None:
        #     assert load_pretrain.endswith('.pth'), "load_pretrain should end with .pth"
        #     utils.load_weights(load_pretrain, self.model)

        # self.model.cuda()

        # if dist_model:
        #     self.model = utils.DistModule(self.model)
        #     self.world_size = dist.get_world_size()
        # else:
        #     self.model = backbone.FixModule(self.model)
        #     self.world_size = 1

        # self.demo = demo
        # if demo:
        #     return
        #
        # # optim
        # self.optim = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()), lr=params['lr'])
        #
        # # netD
        # self.netD = backbone.__dict__[params['discriminator']](**params['discriminator_params'])
        # self.netD.cuda()
        # if dist_model:
        #     self.netD = utils.DistModule(self.netD)
        # else:
        #     self.netD = backbone.FixModule(self.netD)
        # self.optimD = torch.optim.Adam(
        #     self.netD.parameters(), lr=params['lr'] * params['d2g_lr'], betas=(0.0, 0.9))
        #
        # # loss
        # self.criterion = InpaintingLoss(backbone.VGG16FeatureExtractor()).cuda()
        # self.gan_criterion = AdversarialLoss(type=params['gan_type']).cuda()
        #
        # cudnn.benchmark = True

    # def set_input(self, rgb, visible_mask, modal, rgb_gt=None):
    #     self.rgb = rgb.cuda()
    #     if self.with_modal:
    #         self.modal = modal.cuda()
    #     self.visible_mask3 = visible_mask.repeat(
    #         1, 3, 1, 1).cuda()
    #     if self.with_modal:
    #         self.visible_mask4 = visible_mask.repeat(
    #             1, 4, 1, 1).cuda()
    #     if rgb_gt is not None:
    #         self.rgb_gt = rgb_gt.cuda()

    # def set_input(self, rgb, visible_mask, modal, rgb_gt=None):
    def set_input(self, r_rgb_erased, l_rgb_erased, r_visible_mask, l_visible_mask, r_modal, l_modal, rgb_gt=None):
        self.r_rgb_erased = r_rgb_erased.cuda()
        self.l_rgb_erased = l_rgb_erased.cuda()
        if self.with_modal:
            self.r_modal = r_modal.cuda()
            self.l_modal = l_modal.cuda()
        # self.visible_mask3 = visible_mask.repeat(
        #     1, 3, 1, 1).cuda()
        self.r_visible_mask3 = r_visible_mask.repeat(
            1, 3, 1, 1).cuda()
        self.l_visible_mask3 = l_visible_mask.repeat(
            1, 3, 1, 1).cuda()
        if self.with_modal:
            self.r_visible_mask4 = r_visible_mask.repeat(
                1, 4, 1, 1).cuda()
            self.l_visible_mask4 = l_visible_mask.repeat(
                1, 4, 1, 1).cuda()

        self.rgb_gt=rgb_gt
        if rgb_gt is not None:
            self.rgb_gt = rgb_gt.cuda()

    # def forward_only(self, ret_loss=True):
    #     with torch.no_grad():
    #         if self.with_modal:
    #             # output, _ = self.model(torch.cat([self.rgb, self.modal], dim=1),
    #             #                        self.visible_mask4)
    #             # print(f" self.r_rgb_erased.shape= {self.r_rgb_erased.shape}, self.r_modal.shape= {self.r_modal.shape}, self.l_rgb_erased.shape= {self.l_rgb_erased.shape}, self.l_modal.shape={self.l_modal.shape}")
    #             #
    #             # print("self.r_visible_mask4.shape=",self.r_visible_mask4.shape)
    #             # print("self.l_visible_mask4.shape=",self.l_visible_mask4.shape)
    #             output, _ = self.model(
    #                 torch.cat([self.r_rgb_erased, self.r_modal, self.l_rgb_erased, self.l_modal], dim=1),
    #                 torch.cat([self.r_visible_mask4, self.l_visible_mask4], dim=1))
    #         else:
    #             output, _ = self.model(self.rgb, self.visible_mask3)
    #         if output.shape[2] != self.r_rgb_erased.shape[2]:
    #             output = nn.functional.interpolate(
    #                 output, size=self.rgb.shape[2:4],
    #                 mode="bilinear", align_corners=True)
    #         # output_comp = self.visible_mask3 * self.rgb + (1 - self.visible_mask3) * output
    #         output_comp = self.r_visible_mask3 * self.l_visible_mask3 * self.r_rgb_erased + (
    #                     1 - self.r_visible_mask3 * self.l_visible_mask3) * output
    #
    #     if self.with_modal:
    #         # mask_tensors = [self.modal, self.visible_mask3]
    #         mask_tensors = [self.r_modal, self.r_visible_mask3, self.l_modal, self.l_visible_mask3]
    #     else:
    #         mask_tensors = [self.visible_mask3]
    #     ret_tensors = {'common_tensors': [self.r_rgb_erased, self.l_rgb_erased, output, output_comp, self.rgb_gt],
    #                    'mask_tensors': mask_tensors}
    #     if ret_loss:
    #         # loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
    #         loss_dict = self.criterion(self.r_rgb_erased, self.r_visible_mask3,self.l_visible_mask3, output, self.rgb_gt)
    #         for k in loss_dict.keys():
    #             loss_dict[k] /= self.world_size
    #         return ret_tensors, loss_dict
    #     else:
    #         return ret_tensors

    def forward_only(self, ret_loss=True,dilate_kernerl_size=None):
        with torch.no_grad():
            if self.with_modal:
                # output, _ = self.model(torch.cat([self.rgb, self.modal], dim=1),
                #                        self.visible_mask4)
                # print(f" self.r_rgb_erased.shape= {self.r_rgb_erased.shape}, self.r_modal.shape= {self.r_modal.shape}, self.l_rgb_erased.shape= {self.l_rgb_erased.shape}, self.l_modal.shape={self.l_modal.shape}")
                #
                # print("self.r_visible_mask4.shape=",self.r_visible_mask4.shape)
                # print("self.l_visible_mask4.shape=",self.l_visible_mask4.shape)
                output, _ = self.model(
                    torch.cat([self.r_rgb_erased, self.r_modal, self.l_rgb_erased, self.l_modal], dim=1),
                    torch.cat([self.r_visible_mask4, self.l_visible_mask4], dim=1))
            else:
                output, _ = self.model(self.rgb, self.visible_mask3)
            if output.shape[2] != self.r_rgb_erased.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.rgb.shape[2:4],
                    mode="bilinear", align_corners=True)
            # output_comp = self.visible_mask3 * self.rgb + (1 - self.visible_mask3) * output
            ori_region_mask = self.r_visible_mask3 * self.l_visible_mask3
            dr_region_mask = 1 - ori_region_mask # N*3*256,256


            # 膨胀 dr_region_mask
            if dilate_kernerl_size is not None:
                # dr_region_mask_dilated_tensor = torch.zeros([self.r_visible_mask3.shape[0], self.r_visible_mask3.shape[1], self.r_visible_mask3.shape[2], self.r_visible_mask3.shape[3]],
                #                              dtype=torch.float32, device=self.r_visible_mask3.device)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernerl_size, dilate_kernerl_size))
                dr_region_mask_imgs=dr_region_mask.clone().cpu().numpy()[:,0,:,:].astype(np.uint8) # N*3*256*256 --->N*256*256
                # print("dr_region_mask_imgs.shape=",dr_region_mask_imgs.shape)

                # dr_region_mask_dilated_list=[cv2.dilate(dr_region_mask_img,kernel) for dr_region_mask_img in  dr_region_mask_imgs] # list, every item is 256*256
                dr_region_mask_dilated_list =[]
                for dr_region_mask_img in dr_region_mask_imgs:
                    # print("type(dr_region_mask_img)=",type(dr_region_mask_img))
                    # print("dr_region_mask_img.shape=",dr_region_mask_img.shape) #256,256
                    # np.save("dr_region_mask_img.npy",dr_region_mask_img)
                    dr_region_mask_img_dilate=cv2.dilate(dr_region_mask_img, kernel)
                    # print("type(dr_region_mask_img_dilate)=", type(dr_region_mask_img_dilate))
                    # print("dr_region_mask_img_dilate.shape=", dr_region_mask_img_dilate.shape)
                    dr_region_mask_dilated_list.append(dr_region_mask_img_dilate[np.newaxis,np.newaxis, :, :]) #256*256 --》 1*1*256*256




                dr_region_mask_dilated=np.concatenate(dr_region_mask_dilated_list,axis=0) # N*1*256*256
                dr_region_mask_dilated_tensor = torch.tensor(dr_region_mask_dilated).repeat(
                    1, 3, 1, 1).cuda().float() # N*3*256*256
                output_comp = (1-dr_region_mask_dilated_tensor) * self.r_rgb_erased + dr_region_mask_dilated_tensor * output
            else:
                output_comp = self.r_visible_mask3 * self.l_visible_mask3 * self.r_rgb_erased + (
                        1 - self.r_visible_mask3 * self.l_visible_mask3) * output

        if self.with_modal:
            # mask_tensors = [self.modal, self.visible_mask3]
            mask_tensors = [self.r_modal, self.r_visible_mask3, self.l_modal, self.l_visible_mask3]
        else:
            mask_tensors = [self.visible_mask3]
        ret_tensors = {'common_tensors': [self.r_rgb_erased, self.l_rgb_erased, output, output_comp, self.rgb_gt],
                       'mask_tensors': mask_tensors}
        if ret_loss:
            # loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
            loss_dict = self.criterion(self.r_rgb_erased, self.r_visible_mask3, self.l_visible_mask3, output,
                                       self.rgb_gt)
            for k in loss_dict.keys():
                loss_dict[k] /= self.world_size
            return ret_tensors, loss_dict
        else:
            return ret_tensors


    def step(self):
        # output
        if self.with_modal:
            # output, _ = self.model(torch.cat([self.rgb, self.modal], dim=1),
            #                        self.visible_mask4)
            output, _ = self.model(
                torch.cat([self.r_rgb_erased, self.r_modal, self.l_rgb_erased, self.l_modal], dim=1),
                torch.cat([self.r_visible_mask4, self.l_visible_mask4], dim=1))
        else:
            output, _ = self.model(self.rgb, self.visible_mask3)
        if output.shape[2] != self.r_rgb_erased.shape[2]:
            print("WHY?" * 100)
            output = nn.functional.interpolate(
                output, size=self.rgb.shape[2:4],
                mode="bilinear", align_corners=True)

        # discriminator loss
        dis_input_real = self.rgb_gt
        dis_input_fake = output.detach()
        if self.with_modal:
            dis_real, _ = self.netD(torch.cat([dis_input_real, self.r_modal,  self.l_modal], dim=1))
            dis_fake, _ = self.netD(torch.cat([dis_input_fake, self.r_modal,  self.l_modal], dim=1))
        else:
            dis_real, _ = self.netD(dis_input_real)
            dis_fake, _ = self.netD(dis_input_fake)
        dis_real_loss = self.gan_criterion(dis_real, True, True) / self.world_size
        dis_fake_loss = self.gan_criterion(dis_fake, False, True) / self.world_size
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_loss = 0
        gen_input_fake = output
        if self.with_modal:
            gen_fake, _ = self.netD(torch.cat([gen_input_fake, self.r_modal,  self.l_modal], dim=1))
        else:
            gen_fake, _ = self.netD(gen_input_fake)
        gen_gan_loss = self.gan_criterion(gen_fake, True, False) * \
                       self.params['adv_loss_weight'] / self.world_size
        gen_loss += gen_gan_loss

        # other losses
        # loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
        loss_dict = self.criterion(self.r_rgb_erased, self.r_visible_mask3,self.l_visible_mask3, output, self.rgb_gt)
        for k in loss_dict.keys():
            loss_dict[k] /= self.world_size
        for key, coef in self.params['lambda_dict'].items():
            value = coef * loss_dict[key]
            gen_loss += value

        # create loss dict
        loss_dict['dis'] = dis_loss
        loss_dict['adv'] = gen_gan_loss

        # update
        self.optim.zero_grad()
        gen_loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()

        self.optimD.zero_grad()
        dis_loss.backward()
        utils.average_gradients(self.netD)
        self.optimD.step()

        return loss_dict

    def load_model_demo(self, path):
        utils.load_state(path, self.model)

    def load_state(self, root, Iter, resume=False,clear_module=False):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
            utils.load_state(netD_path, self.netD, self.optimD)
        else:
            utils.load_state(path, self.model,clear_module=clear_module)
            if self.demo:
                return
            utils.load_state(netD_path, self.netD)

    def save_state(self, root, Iter):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

        torch.save({
            'step': Iter,
            'state_dict': self.netD.state_dict(),
            'optimizer': self.optimD.state_dict()}, netD_path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
            self.netD.train()
        else:
            self.model.eval()
            if not self.demo:
                self.netD.eval()


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = {'backbone_arch': "PConvUNet",
              # "backbone_param": {"input_channels": 8,"layer_size":7},
              "backbone_param": {"input_channels": 8,"layer_size":8},
              "lr": 0.0001,
              "discriminator": "InpaintDiscriminator",
              "discriminator_params":
                  {"in_channels": 5,
                   "use_sigmoid": True},
              "d2g_lr": 0.1,
              "with_modal": True,
              "adv_loss_weight": 10,
              "dis_loss_weight": 10,
              "rec_comp_weight": 5,
              "data_mean": [0.485, 0.456, 0.406],
              "data_std": [0.229, 0.224, 0.225],
              "lambda_dict": {'rec': 2, 'cor': 1, 'style': 40},
              'comp_weight': 5.,
              "use_discriminator": False,
              'gan_type': "nsgan",
              }
    myPCCGAN = PartialCompletionContentCGAN(config, load_pretrain="../pretrains/partialconv_input_ch8.pth")
    # print("PCCGAN=", myPCCGAN)
    model_structure(myPCCGAN)
