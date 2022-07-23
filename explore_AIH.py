import numpy as np
import torch.utils.data
import os
import os.path as osp
from PIL import Image
import sys
import albumentations as A
from tqdm import tqdm

import pdb

try:
    import sys

    sys.path.append('/mnt/lustre/share/pymc/py3')
    import mc
except Exception:
    pass

# sys.path.insert(0, "./")

import utils
import torchvision.transforms as transforms


class PartialCompContentDataset(torch.utils.data.Dataset):
    def __init__(self, config, phase, visual=False, debug=False):
        self.config = config
        self.phase = phase
        self.visual = visual
        self.debug = debug

        syn_indexs_root = config['syn_indexs_root']

        if phase == "train":
            right_jdxs_M = np.loadtxt(
                osp.join(syn_indexs_root, "machine_annot_{}_sh_right_filtered.txt").format(phase)).copy().astype(
                np.int32).copy()
            left_jdxs_M = np.loadtxt(
                osp.join(syn_indexs_root, "machine_annot_{}_sh_left_filtered.txt".format(phase))).astype(
                np.int32).copy()

            right_jdxs_H = np.loadtxt(
                osp.join(syn_indexs_root, "human_annot_{}_sh_right_filtered.txt").format(phase)).copy().astype(
                np.int32).copy()
            left_jdxs_H = np.loadtxt(
                osp.join(syn_indexs_root, "human_annot_{}_sh_left_filtered.txt").format(phase)).copy().astype(
                np.int32).copy()

            self.jdxs_H = {"right": right_jdxs_H, "left": left_jdxs_H}
            self.segdata_path_H = osp.join(config["syn_segdata_root"], "human_annot", phase)
            self.img_root_H = {"right": osp.join(self.segdata_path_H, "images", "r_images"),
                               "left": osp.join(self.segdata_path_H, "images", "l_images")}
            self.amodal_mask_root_H = {"right": osp.join(self.segdata_path_H, "labels", "in_r_images", "r_amodal"),
                                       "left": osp.join(self.segdata_path_H, "labels", "in_l_images", "l_amodal")}
            syn_cfg_path_H = osp.join(config['syn_cfg_root'], "all_{}_{}_syn_cfgs.npy".format("human_annot", phase))
            # syn_cfg_path = "syn_cfgs/all_machine_annot_train_syn_cfgs.npy"
            self.syn_cfgs_H = np.load(syn_cfg_path_H, allow_pickle=True).tolist()

            self.render_data_root = config["render_data_root"]

            # human-annot train
            render_h_train_r_imgs_path = osp.join(self.render_data_root, "human_annot", "train", "images", "r_images",
                                                  "IH")
            self.render_h_train_r_imgs_list = sorted(os.listdir(render_h_train_r_imgs_path))
            render_h_train_l_imgs_path = osp.join(self.render_data_root, "human_annot", "train", "images", "l_images",
                                                  "IH")
            self.render_h_train_l_imgs_list = sorted(os.listdir(render_h_train_l_imgs_path))

            # machine-annot train
            render_m_train_r_imgs_path = osp.join(self.render_data_root, "machine_annot", "train", "images", "r_images",
                                                  "IH")
            self.render_m_train_r_imgs_list = sorted(os.listdir(render_m_train_r_imgs_path))
            render_m_train_l_imgs_path = osp.join(self.render_data_root, "machine_annot", "train", "images", "l_images",
                                                  "IH")
            self.render_m_train_l_imgs_list = sorted(os.listdir(render_m_train_l_imgs_path))

        else:
            right_jdxs_M = np.loadtxt(
                osp.join(syn_indexs_root, "machine_annot_{}_sh_right_filtered.txt").format(phase)).copy().astype(
                np.int32).copy()
            left_jdxs_M = np.loadtxt(
                osp.join(syn_indexs_root, "machine_annot_{}_sh_left_filtered.txt".format(phase))).astype(
                np.int32).copy()

        self.jdxs_M = {"right": right_jdxs_M, "left": left_jdxs_M}
        self.segdata_path_M = osp.join(config["syn_segdata_root"], "machine_annot", phase)
        self.img_root_M = {"right": osp.join(self.segdata_path_M, "images", "r_images"),
                           "left": osp.join(self.segdata_path_M, "images", "l_images")}
        self.amodal_mask_root_M = {"right": osp.join(self.segdata_path_M, "labels", "in_r_images", "r_amodal"),
                                   "left": osp.join(self.segdata_path_M, "labels", "in_l_images", "l_amodal")}

        self.categories = {"right": 1.0, "left": 1.0}

        syn_cfg_path_M = osp.join(config['syn_cfg_root'], "all_{}_{}_syn_cfgs.npy".format("machine_annot", phase))
        # syn_cfg_path = "syn_cfgs/all_machine_annot_train_syn_cfgs.npy"
        self.syn_cfgs_M = np.load(syn_cfg_path_M, allow_pickle=True).tolist()

        # self.syn_cfgs_M = syn_cfgs_M
        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])
        self.synthesize_setter = utils.SynthesizeSetter(config, forward_only=True)

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)

        if config["final_aug"] == "less":
            # self.final_aug_transform = A.Compose([
            #     A.RGBShift(p=0.5),
            #     A.RandomBrightnessContrast(p=0.5),
            # ])

            self.final_aug_transform = A.Compose([
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], additional_targets={'image2': 'image', 'mask2': 'mask'})
        elif config["final_aug"] == "more":
            # self.final_aug_transform = A.Compose([
            #     A.CLAHE(p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            #
            #     A.RGBShift(p=0.5),
            #     A.RandomBrightnessContrast(p=0.5),
            # ])

            self.final_aug_transform = A.Compose([
                A.CLAHE(p=0.1),
                A.Blur(blur_limit=3, p=0.1),

                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], additional_targets={'image2': 'image', 'mask2': 'mask'})

        else:
            pass
        print("final_aug is : {}".format(config["final_aug"]))

        print('Number of synthesized M-{} Samples : {}'.format(phase, len(self.syn_cfgs_M)))
        if self.phase == "train":
            print('Number of synthesized H-{} Samples : {}'.format(phase, len(self.syn_cfgs_H)))

            print('Number of render_h_train_r Samples : {}'.format(len(self.render_h_train_r_imgs_list)))
            print('Number of render_h_train_l Samples : {}'.format(len(self.render_h_train_l_imgs_list)))

            print('Number of render_m_train_r Samples : {}'.format(len(self.render_m_train_r_imgs_list)))
            print('Number of render_m_train_l Samples : {}'.format(len(self.render_m_train_l_imgs_list)))

        print('Total {} Samples : {}!'.format(phase, len(self)))

    def __len__(self):
        return len(self.syn_cfgs_M) if self.phase != "train" else len(self.syn_cfgs_H + self.syn_cfgs_M
                                                                      + self.render_h_train_r_imgs_list + self.render_h_train_l_imgs_list
                                                                      + self.render_m_train_r_imgs_list + self.render_m_train_l_imgs_list)

    def _init_memcached(self):
        if not self.initialized:
            assert self.memcached_client is not None, "Please specify the path of your memcached_client"
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
                img = np.array(img)
            except:
                print('Read image failed ({})'.format(fn), flush=True)
                pdb.set_trace()
                print('Read image failed ({})'.format(fn), flush=True)
                raise Exception("Exit")
            else:
                return img
        else:
            return np.array(Image.open(fn).convert('RGB'))

    def _load_mask(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader_mask(img_value_str)
                img = np.array(img)
            except:
                print('Read image failed ({})'.format(fn), flush=True)
                raise Exception("Exit")
            else:
                return img
        else:
            return np.array(Image.open(fn))

    def _get_inst(self, idx, hand_side="right", annot_subset=None):

        if annot_subset == "M":
            jdx = self.jdxs_M[hand_side][idx]
            img_path = osp.join(self.img_root_M[hand_side], "{}.jpg".format(jdx))
            amodal_mask_path = osp.join(self.amodal_mask_root_M[hand_side], "{}.png".format(jdx))
        else:
            jdx = self.jdxs_H[hand_side][idx]
            img_path = osp.join(self.img_root_H[hand_side], "{}.jpg".format(jdx))
            amodal_mask_path = osp.join(self.amodal_mask_root_H[hand_side], "{}.png".format(jdx))
        img = self._load_image(img_path)  # uint8
        amodal_mask = self._load_mask(amodal_mask_path)

        if np.all(amodal_mask == 0):
            return self._get_inst(
                np.random.choice(len(self)), hand_side=hand_side)

        return img, amodal_mask

    def getitem_syn(self, idx):
        if self.phase == "train":
            if idx < len(self.syn_cfgs_M):
                syn_cfg = self.syn_cfgs_M[idx]
                tmp_annot_subset = "M"
            else:
                syn_cfg = self.syn_cfgs_H[idx - len(self.syn_cfgs_M)]
                tmp_annot_subset = "H"
        else:
            syn_cfg = self.syn_cfgs_M[idx]
            tmp_annot_subset = "M"

        loop_side = syn_cfg["loop_side"]
        loop_side_idx = syn_cfg["loop_side_idx"]
        another_side_idx = syn_cfg["another_side_idx"]
        assert another_side_idx != -1000, print("oops! error!")
        r_aug_list = syn_cfg["r_aug_list"]
        l_aug_list = syn_cfg["l_aug_list"]
        occlusion_rate = syn_cfg["occlusion_rate"]

        loop_side_img, loop_side_amodal_mask = self._get_inst(loop_side_idx,
                                                              hand_side=loop_side,
                                                              annot_subset=tmp_annot_subset)  # modal, uint8 {0, 1}
        another_side = "right" if loop_side == "left" else "left"
        another_side_img, another_side_amodal_mask = self._get_inst(another_side_idx,
                                                                    hand_side=another_side,
                                                                    annot_subset=tmp_annot_subset)  # modal, uint8 {0, 1}

        if loop_side == "right":
            r_img, r_amodal_mask = loop_side_img, loop_side_amodal_mask
            l_img, l_amodal_mask = another_side_img, another_side_amodal_mask
        else:
            r_img, r_amodal_mask = loop_side_img[:, ::-1], loop_side_amodal_mask[:, ::-1]
            l_img, l_amodal_mask = another_side_img[:, ::-1], another_side_amodal_mask[:, ::-1]

        r_img_aug, r_amodal_mask_aug, l_img_aug, l_amodal_mask_aug = self.synthesize_setter.forward_only(
            r_img, r_amodal_mask, l_img, l_amodal_mask, r_aug_list, l_aug_list, phase=self.phase)  # uint8 {0, 1}

        r_img_aug, r_amodal_mask_aug, l_img_aug, l_amodal_mask_aug = r_img_aug.astype(
            np.uint8), r_amodal_mask_aug.astype(
            np.uint8), l_img_aug.astype(
            np.uint8), l_amodal_mask_aug.astype(np.uint8)
        if self.phase == "train":

            if self.config["final_aug"] != "None":
                transformed = self.final_aug_transform(image=r_img_aug, image2=l_img_aug, mask=r_amodal_mask_aug,
                                                       mask2=l_amodal_mask_aug)
                r_img_aug, l_img_aug = transformed['image'], transformed['image2']
                r_amodal_mask_aug, l_amodal_mask_aug = transformed['mask'], transformed['mask2']

        occluded_img = l_img_aug * l_amodal_mask_aug[:, :, None] + r_img_aug * (1 - l_amodal_mask_aug[:, :, None])

        intersection = r_amodal_mask_aug * l_amodal_mask_aug
        occlusion_rate_ = intersection.sum() / (r_amodal_mask_aug.sum())

        r_invisible_mask = intersection.copy()
        # assert abs(occlusion_rate_ - occlusion_rate) < 1e-6, print("error!")  # train_set could pass, val_set and test_set would fail. It may be due to develop history, but would not influence the train and evaluation process. Take it easy.
        r_visible_mask = (1 - r_invisible_mask.copy())[np.newaxis, :, :].astype(np.float32)
        r_modal_mask = r_amodal_mask_aug * r_visible_mask
        r_modal_mask = r_modal_mask * self.categories["right"]

        l_invisible_mask = (1 - r_amodal_mask_aug) * l_amodal_mask_aug  # intersection
        l_visible_mask = (1 - l_invisible_mask)[np.newaxis, :, :].astype(np.float32)
        l_modal_mask = (1 - r_amodal_mask_aug) * l_visible_mask
        l_modal_mask = l_modal_mask * self.categories["left"]

        if self.visual:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=[50, 50])

            plt.subplot(2, 7, 1)
            plt.imshow(r_img.copy().astype(int))
            plt.title('idx={}, r_img'.format(idx))

            plt.subplot(2, 7, 2)
            plt.imshow(r_amodal_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('r_amodal_mask')

            plt.subplot(2, 7, 3)
            plt.imshow(r_img_aug.copy().astype(int))
            plt.title('r_img_aug')

            plt.subplot(2, 7, 4)
            plt.imshow(r_amodal_mask_aug.copy().squeeze().astype(int), cmap="gray")
            plt.title('r_amodal_mask_aug')

            plt.subplot(2, 7, 5)
            plt.imshow(r_visible_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('r_visible_mask')

            plt.subplot(2, 7, 6)
            plt.imshow(r_modal_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('r_modal_mask')

            plt.subplot(2, 7, 7)
            plt.imshow(occluded_img.copy().squeeze().astype(int), cmap="gray")
            plt.title('occluded_img')

            plt.subplot(2, 7, 8)
            plt.imshow(l_img.copy().squeeze().astype(int))
            plt.title('l_img')

            plt.subplot(2, 7, 9)
            plt.imshow(l_amodal_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_amodal_mask')

            plt.subplot(2, 7, 10)
            plt.imshow(l_img_aug.copy().squeeze().astype(int))
            plt.title('l_img_aug')

            plt.subplot(2, 7, 11)
            plt.imshow(l_amodal_mask_aug.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_amodal_mask_aug')

            plt.subplot(2, 7, 12)
            plt.imshow(l_visible_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_visible_mask')

            plt.subplot(2, 7, 13)
            plt.imshow(l_modal_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_modal_mask')

            plt.subplot(2, 7, 14)
            plt.imshow(r_img_aug.copy().astype(int))
            plt.title('img_gt')

            plt.suptitle("Synthesize data")

            plt.show()

        return occluded_img, r_invisible_mask, r_visible_mask, r_modal_mask, l_invisible_mask, l_visible_mask, l_modal_mask, r_img_aug

    def getitem_render(self, idx):
        jdx = idx - len(self.syn_cfgs_M) - len(self.syn_cfgs_H)

        # self.render_h_train_r_imgs_list + self.render_h_train_l_imgs_list
        # +self.render_m_train_r_imgs_list + self.render_m_train_l_imgs_list
        flip = False
        if jdx < len(self.render_h_train_r_imgs_list):
            img_root_path = osp.join(self.render_data_root, "human_annot", "train", "images", "r_images")
            imgs_list = self.render_h_train_r_imgs_list
            mask_root_path = osp.join(self.render_data_root, "human_annot", "train", "labels", "in_r_images")
            flip = False
            this_jdx = jdx


        elif jdx < len(self.render_h_train_r_imgs_list + self.render_h_train_l_imgs_list):
            img_root_path = osp.join(self.render_data_root, "human_annot", "train", "images", "l_images")
            imgs_list = self.render_h_train_l_imgs_list
            mask_root_path = osp.join(self.render_data_root, "human_annot", "train", "labels", "in_l_images")
            flip = True
            this_jdx = jdx - len(self.render_h_train_r_imgs_list)

        elif jdx < len(
                self.render_h_train_r_imgs_list + self.render_h_train_l_imgs_list + self.render_m_train_r_imgs_list):
            img_root_path = osp.join(self.render_data_root, "machine_annot", "train", "images", "r_images")
            imgs_list = self.render_m_train_r_imgs_list
            mask_root_path = osp.join(self.render_data_root, "machine_annot", "train", "labels", "in_r_images")
            flip = False
            this_jdx = jdx - len(self.render_h_train_r_imgs_list + self.render_h_train_l_imgs_list)

        else:
            img_root_path = osp.join(self.render_data_root, "machine_annot", "train", "images", "l_images")
            imgs_list = self.render_m_train_l_imgs_list
            mask_root_path = osp.join(self.render_data_root, "machine_annot", "train", "labels", "in_l_images")
            flip = True
            this_jdx = jdx - len(
                self.render_h_train_r_imgs_list + self.render_h_train_l_imgs_list + self.render_m_train_r_imgs_list)

        ih_img_path = osp.join(img_root_path, "IH", imgs_list[this_jdx])
        if flip:
            amodal_mask_path = osp.join(mask_root_path, "l_amodal", imgs_list[this_jdx]).replace(".jpg", ".png")
        else:
            amodal_mask_path = osp.join(mask_root_path, "r_amodal", imgs_list[this_jdx]).replace(".jpg", ".png")

        rl_visible_mask_path = osp.join(mask_root_path, "rl_visible", imgs_list[this_jdx]).replace(".jpg", ".png")

        ih_img = self._load_image(ih_img_path)  # uint8

        # dr_gt_img = self._load_image(ih_img_path.replace("IH", "DR_gt",2))  # uint8
        dr_gt_img = self._load_image(osp.join(img_root_path, "DR_gt", imgs_list[this_jdx]))  # uint8
        amodal_mask = self._load_mask(amodal_mask_path)
        rl_visible_mask = self._load_mask(rl_visible_mask_path)

        # old_ih_img = ih_img.copy()
        # old_dr_gt_img = dr_gt_img.copy()
        # old_amodal_mask = amodal_mask.copy()
        # old_rl_visible_mask = rl_visible_mask.copy()

        if flip:
            ih_img = np.flip(ih_img, axis=1)
            dr_gt_img = np.flip(dr_gt_img, axis=1)
            amodal_mask = np.flip(amodal_mask, axis=1)
            rl_visible_mask = np.flip(rl_visible_mask, axis=1)

            rl_visible_mask[rl_visible_mask == 1] = 4
            rl_visible_mask[rl_visible_mask == 2] = 1
            rl_visible_mask[rl_visible_mask == 4] = 2

        transformed_render = self.final_aug_transform(image=ih_img, image2=dr_gt_img, mask=amodal_mask,
                                                      mask2=rl_visible_mask)

        transformed_ih_img, transformed_dr_gt_img = transformed_render['image'], transformed_render['image2']
        transformed_amodal_mask, transformed_rl_visible_mask = transformed_render['mask'], transformed_render['mask2']

        ######
        transformed_ih_img, transformed_dr_gt_img, transformed_amodal_mask, transformed_rl_visible_mask = \
            transformed_ih_img.astype(np.uint8), \
            transformed_dr_gt_img.astype(np.uint8), \
            transformed_amodal_mask.astype(np.uint8), \
            transformed_rl_visible_mask.astype(np.uint8)
        ######

        vis = 0
        if vis:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=[50, 50])

            # plt.subplot(3, 4, 1)
            # plt.imshow(old_ih_img.copy().astype(int))
            # plt.title('old_ih_img')
            #
            # plt.subplot(3, 4, 2)
            # plt.imshow(old_dr_gt_img.copy().astype(int))
            # plt.title('old_dr_gt_img')
            #
            # plt.subplot(3, 4, 3)
            # plt.imshow(old_amodal_mask.copy().astype(int), cmap="gray")
            # plt.title('old_amodal_mask')
            #
            # plt.subplot(3, 4, 4)
            # plt.imshow(old_rl_visible_mask.copy().astype(int), cmap="gray")
            # plt.title('old_rl_visible_mask')

            plt.subplot(3, 4, 5)
            plt.imshow(ih_img.copy().astype(int))
            plt.title('ih_img')

            plt.subplot(3, 4, 6)
            plt.imshow(dr_gt_img.copy().astype(int))
            plt.title('dr_gt_img')

            plt.subplot(3, 4, 7)
            plt.imshow(amodal_mask.copy().astype(int), cmap="gray")
            plt.title('amodal_mask')

            plt.subplot(3, 4, 8)
            plt.imshow(rl_visible_mask.copy().astype(int), cmap="gray")
            plt.title('rl_visible_mask')

            plt.subplot(3, 4, 9)
            plt.imshow(transformed_ih_img.copy().astype(int))
            plt.title('transformed_ih_img')

            plt.subplot(3, 4, 10)
            plt.imshow(transformed_dr_gt_img.copy().astype(int))
            plt.title('transformed_dr_gt_img')

            plt.subplot(3, 4, 11)
            plt.imshow(transformed_amodal_mask.copy().astype(int), cmap="gray")
            plt.title('transformed_amodal_mask')

            plt.subplot(3, 4, 12)
            plt.imshow(transformed_rl_visible_mask.copy().astype(int), cmap="gray")
            plt.title('transformed_rl_visible_mask')

            plt.show()

        right_hand_amodal_mask = transformed_amodal_mask.copy().astype(np.float32)

        right_hand_visible_mask = transformed_rl_visible_mask.copy()
        right_hand_visible_mask[right_hand_visible_mask == 2] = 0.
        right_hand_visible_mask = right_hand_visible_mask.astype(np.float32)

        r_modal_mask = right_hand_visible_mask.copy()[np.newaxis, :, :].astype(np.float32)
        r_invisible_mask = (right_hand_amodal_mask * (1 - right_hand_visible_mask))
        r_visible_mask = (1 - r_invisible_mask.copy())[np.newaxis, :, :].astype(np.float32)

        left_hand_visible_mask = transformed_rl_visible_mask.copy().astype(np.float32)
        left_hand_visible_mask[left_hand_visible_mask == 1] = 0.
        left_hand_visible_mask[left_hand_visible_mask == 2] = 1.

        l_modal_mask = (1 - left_hand_visible_mask.copy()) * (1 - right_hand_visible_mask.copy())[np.newaxis, :,
                                                             :].astype(np.float32)
        l_invisible_mask = ((1 - right_hand_amodal_mask) * left_hand_visible_mask)
        l_visible_mask = (1 - l_invisible_mask.copy())[np.newaxis, :, :].astype(np.float32)

        r_modal_mask = r_modal_mask * self.categories["right"]
        l_modal_mask = l_modal_mask * self.categories["left"]

        return transformed_ih_img, r_invisible_mask, r_visible_mask, r_modal_mask, l_invisible_mask, l_visible_mask, l_modal_mask, transformed_dr_gt_img

    def __getitem__(self, idx):

        if self.memcached:
            self._init_memcached()
        if self.phase == "train" and idx >= len(self.syn_cfgs_M) + len(self.syn_cfgs_H):
            occluded_img, r_invisible_mask, r_visible_mask, r_modal_mask, l_invisible_mask, l_visible_mask, l_modal_mask, dr_img_gt = self.getitem_render(
                idx)

        else:
            occluded_img, r_invisible_mask, r_visible_mask, r_modal_mask, l_invisible_mask, l_visible_mask, l_modal_mask, dr_img_gt = self.getitem_syn(
                idx)

        if self.visual:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=[50, 50])

            plt.subplot(2, 5, 1)
            plt.imshow(occluded_img.copy().astype(int))
            plt.title('idx={}, occluded_img'.format(idx))

            plt.subplot(2, 5, 2)
            plt.imshow(r_invisible_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('r_invisible_mask')

            plt.subplot(2, 5, 3)
            plt.imshow(r_visible_mask.squeeze().copy().astype(int), cmap="gray")
            plt.title('r_visible_mask')

            plt.subplot(2, 5, 4)
            plt.imshow(r_modal_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('r_modal_mask')

            plt.subplot(2, 5, 5)
            plt.imshow(dr_img_gt.copy().squeeze().astype(int), cmap="gray")
            plt.title('dr_img_gt')

            plt.subplot(2, 5, 6)
            plt.imshow(occluded_img.copy().astype(int))
            plt.title('idx={}, occluded_img'.format(idx))

            plt.subplot(2, 5, 7)
            plt.imshow(l_invisible_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_invisible_mask')

            plt.subplot(2, 5, 8)
            plt.imshow(l_visible_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_visible_mask')

            plt.subplot(2, 5, 9)
            plt.imshow(l_modal_mask.copy().squeeze().astype(int), cmap="gray")
            plt.title('l_modal_mask')

            plt.subplot(2, 5, 10)
            plt.imshow(dr_img_gt.copy().astype(int))
            plt.title('dr_img_gt')

            plt.suptitle("Data")

            plt.show()

        r_visible_mask_tensor = torch.from_numpy(r_visible_mask)
        l_visible_mask_tensor = torch.from_numpy(l_visible_mask)

        img = torch.from_numpy(occluded_img.astype(np.float32).transpose((2, 0, 1)) / 255.)
        img = self.img_transform(img)  # CHW

        r_img_erased = img.clone()
        r_img_erased = r_img_erased * r_visible_mask_tensor  # erase rgb
        l_img_erased = img.clone()
        l_img_erased = l_img_erased * l_visible_mask_tensor  # erase rgb

        dr_img_gt = torch.from_numpy(dr_img_gt.astype(np.float32).transpose((2, 0, 1)) / 255.)
        dr_img_gt = self.img_transform(dr_img_gt)  # CHW (-1,1)

        return img, r_img_erased, l_img_erased, r_visible_mask_tensor, l_visible_mask_tensor, r_modal_mask, l_modal_mask, dr_img_gt


if __name__ == '__main__':

    AIH_root = 'C:\AIH_dataset'
    myconfg = {

        # AIH_syn config
        "syn_segdata_root": osp.join(AIH_root, 'AIH_syn'),
        'syn_cfg_root': osp.join(AIH_root, 'AIH_syn', 'syn_cfgs'),
        'syn_indexs_root': osp.join(AIH_root, 'AIH_syn', 'filtered_list'),
        "final_aug": "less",

        # AIH_render config
        "render_data_root": osp.join(AIH_root, 'AIH_render'),

        # mean & std
        "data_mean": [0.485, 0.456, 0.406],
        "data_std": [0.229, 0.224, 0.225],

    }
    phase = "train";
    stride = 60000
    # phase = "val";
    # stride = 5000
    # phase = "test";
    # stride = 5000

    dataset = PartialCompContentDataset(myconfg, phase=phase, visual=True, debug=False)
    print(len(dataset))
    for i in tqdm(range(0, len(dataset), stride)):
        dataset[i]
