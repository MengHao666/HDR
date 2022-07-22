# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
# from main.config import cfg
# from config import cfg
import random
import math

import torch


def load_img(path, order='RGB'):
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def load_skeleton(path, joint_num):
    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id

    return skeleton


def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2

    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5  # 用于生成一个0到1的随机符点数: 0 <= n < 1.0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, do_flip, color_scale


# all params in padding image cooridinate
# img_padding, bboxs_padding,
#                                                                                          masks_padding,
#                                                                                          joint_cam, joint_img_padding,
#                                                                                          joint_cam_valid,
#                                                                                          joint_img_valid,
#                                                                                          self.joint_type,
#                                                                                          self.config


# img_padding,
#             bboxs_padding,
#             joint_cam,
#             joint_img_padding,
#             joint_cam_valid,
#             joint_img_valid,
#             self.joint_type,
#             self.config, masks

# def augmentation(img, bboxs, joint_cam, joint_img, joint_cam_valid, joint_img_valid, joint_type,
#                  config, hand_side, mode, hand_type):
#     # hand_side = config['hand_side']
#     img = img.copy();
#     joint_cam = joint_cam.copy();
#     # hand_type = hand_type.copy();
#
#     original_img_shape = img.shape
#     joint_num = len(joint_cam)
#
#     if mode == 'train':
#         translation, scale, rot, do_flip, color_scale = get_aug_config()
#         if hand_type == hand_side:
#             do_flip = False
#     else:
#         translation, scale, rot, do_flip, color_scale = [0, 0], 1.0, 0.0, False, np.array([1, 1, 1])
#
#     # print("hand_type=",hand_type)
#     # print("hand_side=",hand_side)
#
#     # color_scale = np.array([1, 1, 1])  # menghao 0811, 这个应该是需要做的，但是之前没做不知道为啥,0909订正了这个bug
#
#     rot_rad = -np.pi * rot / 180
#     rot_mat = np.array([
#         [np.cos(rot_rad), -np.sin(rot_rad), 0],
#         [np.sin(rot_rad), np.cos(rot_rad), 0],
#         [0, 0, 1],
#     ]).astype(np.float32)
#
#     bbx_r, bbx_l = bboxs["right"], bboxs["left"]
#     # mask_r, mask_l = masks["right"]["all"], masks["left"]["all"]
#     #
#     # mask_img_r = None
#     # if mask_r is not None:
#     #     mask_img_r = np.zeros((original_img_shape[0], original_img_shape[1]))
#     #     mask_img_r[mask_r[:, 0], mask_r[:, 1]] = 1
#     #
#     # mask_img_l = None
#     # if mask_l is not None:
#     #     mask_img_l = np.zeros((original_img_shape[0], original_img_shape[1]))
#     #     mask_img_l[mask_l[:, 0], mask_l[:, 1]] = 1
#
#     if mode == 'test':
#         bbx_r = np.array([img.shape[1] / 4, img.shape[0] / 4, img.shape[1] / 2, img.shape[0] / 2]).astype(
#             np.float32) if bbx_r is None else bbx_r
#         bbx_l = np.array([img.shape[1] / 4, img.shape[0] / 4, img.shape[1] / 2, img.shape[0] / 2]).astype(
#             np.float32) if bbx_l is None else bbx_l
#
#     # Check  datasets before affine
#     visual_check = 0
#     if visual_check:
#         import matplotlib
#         matplotlib.use('TkAgg')
#         import matplotlib.pyplot as plt
#
#         fig = plt.figure(figsize=[50, 50])
#
#         fig.add_subplot(1, 3, 1)
#         plt.imshow(img.copy().astype(int))
#         plt.title('ori img')
#
#         # fig.add_subplot(1, 3, 2)
#         # if mask_img_r is not None:
#         #     plt.imshow(mask_img_r.astype(np.int32), cmap="gray")
#         # plt.title('mask_img_r')
#         #
#         # fig.add_subplot(1, 3, 3)
#         # if mask_img_l is not None:
#         #     plt.imshow(mask_img_l.astype(np.int32), cmap="gray")
#         # plt.title('mask_img_l')
#
#         plt.suptitle("Check  datasets before affine")
#         plt.show()
#
#     if bbx_r is not None:
#         bbx_r[0] = bbx_r[0] + bbx_r[2] * translation[0]
#         bbx_r[1] = bbx_r[1] + bbx_r[3] * translation[1]
#
#     if bbx_l is not None:
#         bbx_l[0] = bbx_l[0] + bbx_l[2] * translation[0]
#         bbx_l[1] = bbx_l[1] + bbx_l[3] * translation[1]
#
#     if do_flip:  # flip
#         bbx_r, bbx_l = bbx_l, bbx_r  # preprocessing
#         # mask_img_r, mask_img_l = mask_img_l, mask_img_r
#
#     img_r, trans_r = None, None
#     inv_trans_r = None
#     mask_img_r_crop = None
#     if bbx_r is not None:
#         img_r, trans_r, inv_trans_r = generate_patch_image(img, bbx_r, do_flip, scale, rot,
#                                                            config['input_img_shape'])
#         img_r = np.clip(img_r * color_scale[None, None, :], 0, 255)
#         # if mask_img_r is not None:
#         #     mask_img_r_crop, _, _ = generate_patch_mask(mask_img_r, bbx_r, do_flip, scale, rot,
#         #                                                 config['input_img_shape'])
#
#     img_l, trans_l = None, None
#     inv_trans_l = None
#     mask_img_l_crop = None
#     if bbx_l is not None:
#         img_l, trans_l, inv_trans_l = generate_patch_image(img, bbx_l, do_flip, scale, rot,
#                                                            config['input_img_shape'])
#         img_l = np.clip(img_l * color_scale[None, None, :], 0, 255)
#         # if mask_img_l is not None:
#         #     mask_img_l_crop, _, _ = generate_patch_mask(mask_img_l, bbx_l, do_flip, scale, rot,
#         #                                                 config['input_img_shape'])
#
#     if do_flip:
#         # print("do flip here !")
#
#         joint_img[:, 0] = original_img_shape[1] - joint_img[:, 0] - 1
#         joint_img[joint_type['right']], joint_img[joint_type['left']] = joint_img[joint_type['left']].copy(), \
#                                                                         joint_img[joint_type['right']].copy()
#
#         joint_cam[:, 0] = - joint_cam[:, 0]
#         joint_cam[joint_type['right']], joint_cam[joint_type['left']] = joint_cam[joint_type['left']].copy(), \
#                                                                         joint_cam[
#                                                                             joint_type['right']].copy()
#
#         joint_cam_valid[joint_type['right']], joint_cam_valid[joint_type['left']] = joint_cam_valid[
#                                                                                         joint_type['left']].copy(), \
#                                                                                     joint_cam_valid[
#                                                                                         joint_type['right']].copy()
#         joint_img_valid[joint_type['right']], joint_img_valid[joint_type['left']] = joint_img_valid[
#                                                                                         joint_type['left']].copy(), \
#                                                                                     joint_img_valid[
#                                                                                         joint_type['right']].copy()
#         # hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
#
#     for i in range(joint_num // 2):
#         joint_img[:21][i, :2] = trans_point2d(joint_img[:21][i, :2], trans_r) if trans_r is not None else joint_img[
#                                                                                                           :21][i,
#                                                                                                           :2]
#         joint_img[21:][i, :2] = trans_point2d(joint_img[21:][i, :2], trans_l) if trans_l is not None else joint_img[
#                                                                                                           21:][i,
#                                                                                                           :2]
#
#     joint_cam = rot_mat.dot(
#         joint_cam.transpose(1, 0)
#     ).transpose()
#
#     # Check Affined datasets
#     visual_check = 0
#     if visual_check:
#
#         import matplotlib
#         matplotlib.use('TkAgg')
#         import matplotlib.pyplot as plt
#
#         fig = plt.figure(figsize=[50, 50])
#
#         fig.add_subplot(2, 5, 1)
#         if img_r is not None:
#             plt.imshow(img_r.copy().astype(int))
#         plt.title('right img')
#
#         fig.add_subplot(2, 5, 2)
#         joint_img_r = joint_img[:21].copy()
#         joint_cam_r = joint_cam[:21].copy()
#         img_valid_r = joint_img_valid[:21]
#         if img_r is not None:
#             plt.imshow(img_r.copy().astype(int))
#             for p in range(joint_img_r.shape[0]):
#                 if not img_valid_r[p]:
#                     continue
#                 plt.plot(joint_img_r[p][0], joint_img_r[p][1], 'bo', markersize=5)
#                 plt.text(joint_img_r[p][0], joint_img_r[p][1], '{0}'.format(p), color="w", fontsize=7.5)
#         plt.title('right img + 2D label')
#
#         plt.subplot(2, 5, 3)
#         if mask_img_r_crop is not None:
#             plt.imshow(mask_img_r_crop.astype(np.int32), cmap="gray")
#         plt.title('right_mask_crop')
#
#         # plt.subplot(2, 5, 4)
#         # if mask_img_r is not None:
#         #     plt.imshow(mask_img_r.astype(np.int32), cmap="gray")
#         # plt.title('right_padding_mask')
#
#         ax = fig.add_subplot(2, 5, 5, projection='3d')
#         big_size = 50
#         hand_3d = joint_cam_r
#         cam_valid_r = joint_cam_valid[:21]
#         for i in range(1, 20):
#             if cam_valid_r[i]:
#                 ax.scatter(hand_3d[i, 0], hand_3d[i, 1], hand_3d[i, 2])
#         if cam_valid_r[-1]:
#             ax.scatter(hand_3d[-1, 0], hand_3d[-1, 1], hand_3d[-1, 2], s=big_size * 1.5, c=config["colors"]["AERO"],
#                        label='right')
#         if cam_valid_r[0]:
#             ax.scatter(hand_3d[0, 0], hand_3d[0, 1], hand_3d[0, 2], s=big_size, c="red", label='right')
#
#         for li, color in zip(config['interhand_link_hand'], config['colors_hand']):
#             if cam_valid_r[li][0] and cam_valid_r[li][1]:
#                 ax.plot(hand_3d[li][:, 0], hand_3d[li][:, 1], hand_3d[li][:, 2], c=color)
#
#         # ax.set_xlabel('x')
#         # ax.set_ylabel('y')
#         # ax.set_zlabel('z')
#         ax.grid(False)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_zticks([])
#         ax.view_init(-90, -90)
#         ax.set_title('right-hand 3D label', fontsize=12, color='r')
#
#         plt.subplot(2, 5, 6)
#         if img_l is not None:
#             plt.imshow(img_l.copy().astype(int))
#         plt.title('left img')
#
#         fig.add_subplot(2, 5, 7)
#         joint_img_l = joint_img[21:]
#         joint_cam_l = joint_cam[21:].copy()
#         img_valid_l = joint_img_valid[21:]
#         cam_valid_l = joint_cam_valid[21:]
#
#         if img_l is not None:
#             plt.imshow(img_l.copy().astype(int))
#             for p in range(joint_img_l.shape[0]):
#                 if not img_valid_l[p]:
#                     continue
#                 plt.plot(joint_img_l[p][0], joint_img_l[p][1], 'bo', markersize=5)
#                 plt.text(joint_img_l[p][0], joint_img_l[p][1], '{0}'.format(p), color="w", fontsize=7.5)
#         plt.title('left img + 2D label')
#
#         plt.subplot(2, 5, 8)
#         if mask_img_l_crop is not None:
#             plt.imshow(mask_img_l_crop.astype(np.int32), cmap="gray")
#         plt.title('left_mask_crop')
#
#         # plt.subplot(2, 5, 9)
#         # if mask_img_l is not None:
#         #     plt.imshow(mask_img_l.astype(np.int32), cmap="gray")
#         # plt.title('left_padding_mask')
#
#         ax = fig.add_subplot(2, 5, 10, projection='3d')
#         hand_3d = joint_cam_l
#         for i in range(1, 20):
#             if cam_valid_l[i]:
#                 ax.scatter(hand_3d[i, 0], hand_3d[i, 1], hand_3d[i, 2])
#         if cam_valid_l[-1]:
#             ax.scatter(hand_3d[-1, 0], hand_3d[-1, 1], hand_3d[-1, 2], s=big_size * 1.5, c=config["colors"]["AERO"],
#                        label='right')
#         if cam_valid_l[0]:
#             ax.scatter(hand_3d[0, 0], hand_3d[0, 1], hand_3d[0, 2], s=big_size, c="red", label='right')
#
#         for li, color in zip(config['interhand_link_hand'], config['colors_hand']):
#             if cam_valid_l[li][0] and cam_valid_l[li][1]:
#                 ax.plot(hand_3d[li][:, 0], hand_3d[li][:, 1], hand_3d[li][:, 2], c=color)
#
#         # ax.set_xlabel('x')
#         # ax.set_ylabel('y')
#         # ax.set_zlabel('z')
#         ax.grid(False)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_zticks([])
#         ax.view_init(-90, -90)
#         ax.set_title('left-hand 3D label', fontsize=12, color='r')
#
#         plt.suptitle("Check Affined datasets")
#         plt.show()
#
#     if hand_side == "right" and img_r is not None:
#         return img_r, joint_cam[joint_type[hand_side]], joint_img[joint_type[hand_side]], \
#                joint_cam_valid[
#                    joint_type[hand_side]], joint_img_valid[joint_type[hand_side]], inv_trans_r
#     elif hand_side == "left" and img_l is not None:
#         return img_l, joint_cam[joint_type[hand_side]], joint_img[joint_type[hand_side]], \
#                joint_cam_valid[
#                    joint_type[hand_side]], joint_img_valid[joint_type[hand_side]], inv_trans_l
#     else:
#
#         # print("another side is used!")
#         another_side = "right" if hand_side == "left" else "left"
#
#         another_img = img_r if hand_side == "left" else img_l
#         another_inv_trans = inv_trans_r if hand_side == "left" else inv_trans_l
#         assert another_img is not None, print('Another side is also None ')
#
#         another_joint_cam = joint_cam[joint_type[another_side]].copy()
#         another_joint_img = joint_img[joint_type[another_side]].copy()
#         # another_mask_crop = mask_img_l_crop if another_side == "left" else mask_img_r_crop
#
#         want_img = another_img.copy()[:, ::-1, :]
#         want_joint_cam = another_joint_cam.copy()
#         want_joint_cam[:, 0] = -want_joint_cam[:, 0]
#
#         # if another_mask_crop is not None:
#         #     want_mask_crop = another_mask_crop.copy()[:, ::-1]
#         # else:
#         #     want_mask_crop = None
#
#         want_joint_img = another_joint_img.copy()
#         want_joint_img[:, 0] = want_img.shape[1] - want_joint_img[:, 0] - 1
#
#         want_joint_cam_valid = joint_cam_valid[joint_type[another_side]]
#         want_joint_img_valid = joint_img_valid[joint_type[another_side]]
#
#         # return want_img, want_mask_crop, want_joint_cam, want_joint_img, want_joint_cam_valid, want_joint_img_valid
#         return want_img, want_joint_cam, want_joint_img, want_joint_cam_valid, want_joint_img_valid, another_inv_trans




# img_padding,
#             bboxs_padding,
#             joint_cam,
#             joint_img_padding,
#             joint_cam_valid,
#             joint_img_valid,
#             self.joint_type,
#             self.config, masks,self.hand_side,self.mode,seg_hand_type


def augmentation_demo(img, bboxs,
                 config, masks_padding):
    hand_side = config['hand_side']
    img = img.copy();
    masks_padding=masks_padding.copy()


    original_img_shape = img.shape

    # if config['mode'] == 'train':
    #     translation, scale, rot, do_flip, color_scale = get_aug_config()
    # else:
    translation, scale, rot, do_flip, color_scale = [0, 0], 1.0, 0.0, False, np.array([1, 1, 1])

    # rot_rad = -np.pi * rot / 180
    # rot_mat = np.array([
    #     [np.cos(rot_rad), -np.sin(rot_rad), 0],
    #     [np.sin(rot_rad), np.cos(rot_rad), 0],
    #     [0, 0, 1],
    # ]).astype(np.float32)

    bbx = bboxs[hand_side]
    # print("bbx=",bbx)

    if bbx is None:
        bbx=np.array([img.shape[1] / 4, img.shape[0] / 4, img.shape[1] / 2, img.shape[0] / 2]).astype(
            np.float32)

    amodal_loc = masks_padding[hand_side]["all"]
    visible_loc = masks_padding[hand_side]["visible"]

    another_side = "right" if hand_side == "left" else "left"
    another_visible_loc = masks_padding[another_side]["visible"]
    another_amodal_loc = masks_padding[another_side]["all"]

    amodal_mask = np.zeros((original_img_shape[0], original_img_shape[1]))
    amodal_mask[amodal_loc[:, 0], amodal_loc[:, 1]] = 1

    visible_mask = np.zeros((original_img_shape[0], original_img_shape[1]))
    visible_mask[visible_loc[:, 0], visible_loc[:, 1]] = 1

    another_visible_mask = np.zeros((original_img_shape[0], original_img_shape[1]))
    if another_visible_loc is not None:
        another_visible_mask[another_visible_loc[:, 0], another_visible_loc[:, 1]] = 1

    another_amodal_mask = np.zeros((original_img_shape[0], original_img_shape[1]))
    if another_amodal_loc is not None:
        another_amodal_mask[another_amodal_loc[:, 0], another_amodal_loc[:, 1]] = 1

    # Check  datasets before affine
    visual_check = 0
    if visual_check:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[50, 50])

        fig.add_subplot(1, 5, 1)
        plt.imshow(img.copy().astype(int))
        for p in range(joint_img.shape[0]):
            if not joint_img_valid[p]:
                continue
            plt.plot(joint_img[p][0], joint_img[p][1], 'bo', markersize=5)
            plt.text(joint_img[p][0], joint_img[p][1], '{0}'.format(p), color="w", fontsize=7.5)
        plt.title('{} img + 2D label'.format(hand_side))
        plt.title('ori img')

        fig.add_subplot(1, 5, 2)
        plt.imshow(amodal_mask.astype(np.int32), cmap="gray")
        plt.title('amodal_mask')

        fig.add_subplot(1, 5, 3)
        plt.imshow(visible_mask.astype(np.int32), cmap="gray")
        plt.title('visible_mask')

        fig.add_subplot(1, 5, 4)
        plt.imshow(another_amodal_mask.astype(np.int32), cmap="gray")
        plt.title('another_amodal_mask')

        fig.add_subplot(1, 5, 5)
        plt.imshow(another_visible_mask.astype(np.int32), cmap="gray")
        plt.title('another_visible_mask')

        plt.suptitle("Check  datasets before affine")
        plt.show()

    bbx[0] = bbx[0] + bbx[2] * translation[0]
    bbx[1] = bbx[1] + bbx[3] * translation[1]

    img_crop, trans, inv_trans = generate_patch_image(img, bbx, do_flip, scale, rot,
                                                      config['input_img_shape'])
    img_crop = np.clip(img_crop * color_scale[None, None, :], 0, 255)
    amodal_mask_crop, _, _ = generate_patch_mask(amodal_mask.copy(), bbx, do_flip, scale, rot,
                                                 config['input_img_shape'])

    visible_mask_crop, _, _ = generate_patch_mask(visible_mask.copy(), bbx, do_flip, scale, rot,
                                                  config['input_img_shape'])

    another_visible_mask_crop, _, _ = generate_patch_mask(another_visible_mask.copy(), bbx, do_flip, scale, rot,
                                                          config['input_img_shape'])
    another_amodal_mask_crop, _, _ = generate_patch_mask(another_amodal_mask.copy(), bbx, do_flip, scale, rot,
                                                         config['input_img_shape'])

    # for i in range(joint_num // 2):
    #     joint_img[i, :2] = trans_point2d(joint_img[i, :2], trans)

    # joint_cam = rot_mat.dot(
    #     joint_cam.transpose(1, 0)
    # ).transpose()

    # Check Affined datasets
    visual_check = 0
    if visual_check:

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[50, 50])

        fig.add_subplot(1, 5, 1)
        plt.imshow(img_crop.copy().astype(int))
        plt.title('{} img'.format(hand_side))

        fig.add_subplot(1, 5, 2)
        plt.imshow(img_crop.copy().astype(int))
        for p in range(joint_img.shape[0]):
            if not joint_img_valid[p]:
                continue
            plt.plot(joint_img[p][0], joint_img[p][1], 'bo', markersize=5)
            plt.text(joint_img[p][0], joint_img[p][1], '{0}'.format(p), color="w", fontsize=7.5)
        plt.title('{} img + 2D label'.format(hand_side))

        plt.subplot(1, 5, 3)
        plt.imshow(amodal_mask_crop.astype(np.int32), cmap="gray")
        plt.title('{}_amodal_mask_crop'.format(hand_side))

        plt.subplot(1, 5, 4)
        plt.imshow(visible_mask_crop.astype(np.int32), cmap="gray")
        plt.title('{}_visible_mask_crop'.format(hand_side))

        ax = fig.add_subplot(1, 5, 5, projection='3d')
        big_size = 50
        hand_3d = joint_cam
        cam_valid_r = joint_cam_valid[:21]
        for i in range(1, 20):
            if cam_valid_r[i]:
                ax.scatter(hand_3d[i, 0], hand_3d[i, 1], hand_3d[i, 2])
        if cam_valid_r[-1]:
            ax.scatter(hand_3d[-1, 0], hand_3d[-1, 1], hand_3d[-1, 2], s=big_size * 1.5, c=config["colors"]["AERO"],
                       label=hand_side)
        if cam_valid_r[0]:
            ax.scatter(hand_3d[0, 0], hand_3d[0, 1], hand_3d[0, 2], s=big_size, c="red", label='right')

        for li, color in zip(config['interhand_link_hand'], config['colors_hand']):
            if cam_valid_r[li][0] and cam_valid_r[li][1]:
                ax.plot(hand_3d[li][:, 0], hand_3d[li][:, 1], hand_3d[li][:, 2], c=color)

        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(-90, -90)
        ax.set_title('{}-hand 3D label'.format(hand_side), fontsize=12, color='r')

        plt.suptitle("Check Affined datasets")
        plt.show()

    return img_crop,  inv_trans, amodal_mask_crop, visible_mask_crop, another_amodal_mask_crop, another_visible_mask_crop

def augmentation(img, bboxs, joint_cam, joint_img, joint_cam_valid, joint_img_valid, joint_type,
                 config, masks, mode, hand_type):
    hand_side = config['hand_side']

    img = img.copy();
    joint_cam = joint_cam.copy();

    original_img_shape = img.shape
    joint_num = len(joint_cam)


    if mode == 'train':
        translation, scale, rot, do_flip, color_scale = get_aug_config()
        if hand_type == hand_side:
            do_flip = False
    else:
        translation, scale, rot, do_flip, color_scale = [0, 0], 1.0, 0.0, False, np.array([1, 1, 1])


    # print(translation, scale, rot, do_flip, color_scale)

    rot_rad = -np.pi * rot / 180
    rot_mat = np.array([
        [np.cos(rot_rad), -np.sin(rot_rad), 0],
        [np.sin(rot_rad), np.cos(rot_rad), 0],
        [0, 0, 1],
    ]).astype(np.float32)

    bbx_r, bbx_l = bboxs["right"], bboxs["left"]

    mask_r_amodal, mask_l_amodal = masks["right"]["all"], masks["left"]["all"]
    mask_r_visible, mask_l_visible = masks["right"]["visible"], masks["left"]["visible"]

    mask_img_r_amodal = np.zeros((original_img_shape[0], original_img_shape[1]))
    if mask_r_amodal is not None:
        mask_img_r_amodal[mask_r_amodal[:, 0], mask_r_amodal[:, 1]] = 1

    mask_img_r_visible = np.zeros((original_img_shape[0], original_img_shape[1]))
    if mask_r_visible is not None:
        mask_img_r_visible[mask_r_visible[:, 0], mask_r_visible[:, 1]] = 1

    mask_img_l_amodal = np.zeros((original_img_shape[0], original_img_shape[1]))
    if mask_l_amodal is not None:
        mask_img_l_amodal[mask_l_amodal[:, 0], mask_l_amodal[:, 1]] = 1

    mask_img_l_visible = np.zeros((original_img_shape[0], original_img_shape[1]))
    if mask_l_visible is not None:
        mask_img_l_visible[mask_l_visible[:, 0], mask_l_visible[:, 1]] = 1


    if mode == 'test':
        bbx_r = np.array([img.shape[1] / 4, img.shape[0] / 4, img.shape[1] / 2, img.shape[0] / 2]).astype(
            np.float32) if bbx_r is None else bbx_r
        bbx_l = np.array([img.shape[1] / 4, img.shape[0] / 4, img.shape[1] / 2, img.shape[0] / 2]).astype(
            np.float32) if bbx_l is None else bbx_l


    if bbx_r is not None:
        bbx_r[0] = bbx_r[0] + bbx_r[2] * translation[0]
        bbx_r[1] = bbx_r[1] + bbx_r[3] * translation[1]

    if bbx_l is not None:
        bbx_l[0] = bbx_l[0] + bbx_l[2] * translation[0]
        bbx_l[1] = bbx_l[1] + bbx_l[3] * translation[1]

    if do_flip:  # flip
        bbx_r, bbx_l = bbx_l, bbx_r  # preprocessing
        mask_img_r_amodal, mask_img_l_amodal = mask_img_l_amodal, mask_img_r_amodal
        mask_img_r_visible, mask_img_l_visible = mask_img_l_visible, mask_img_r_visible

    img_r, trans_r = None, None
    inv_trans_r = None
    mask_img_r_amodal_crop = np.zeros([256, 256])
    mask_img_r_visible_crop = np.zeros([256, 256])
    if bbx_r is not None:
        img_r, trans_r, inv_trans_r = generate_patch_image(img, bbx_r, do_flip, scale, rot,
                                                           config['input_img_shape'])
        img_r = np.clip(img_r * color_scale[None, None, :], 0, 255)

        if mask_img_r_amodal is not None:
            mask_img_r_amodal_crop, _, _ = generate_patch_mask(mask_img_r_amodal, bbx_r, do_flip, scale, rot,
                                                        config['input_img_shape'])
        else:
            mask_img_r_amodal_crop = np.zeros([256, 256])

        if mask_img_r_visible is not None:
            mask_img_r_visible_crop, _, _ = generate_patch_mask(mask_img_r_visible, bbx_r, do_flip, scale, rot,
                                                        config['input_img_shape'])
        else:
            mask_img_r_visible_crop = np.zeros([256, 256])

    img_l, trans_l = None, None
    inv_trans_l = None
    mask_img_l_amodal_crop = np.zeros([256, 256])
    mask_img_l_visible_crop = np.zeros([256, 256])
    if bbx_l is not None:
        img_l, trans_l, inv_trans_l = generate_patch_image(img, bbx_l, do_flip, scale, rot,
                                                           config['input_img_shape'])
        img_l = np.clip(img_l * color_scale[None, None, :], 0, 255)

        if mask_img_l_amodal is not None:
            mask_img_l_amodal_crop, _, _ = generate_patch_mask(mask_img_l_amodal, bbx_l, do_flip, scale, rot,
                                                               config['input_img_shape'])
        else:
            mask_img_l_amodal_crop = np.zeros([256, 256])

        if mask_img_l_visible is not None:
            mask_img_l_visible_crop, _, _ = generate_patch_mask(mask_img_l_visible, bbx_l, do_flip, scale, rot,
                                                                config['input_img_shape'])
        else:
            mask_img_l_visible_crop = np.zeros([256, 256])

    if do_flip:
        # print("do flip here !")

        joint_img[:, 0] = original_img_shape[1] - joint_img[:, 0] - 1
        joint_img[joint_type['right']], joint_img[joint_type['left']] = joint_img[joint_type['left']].copy(), \
                                                                        joint_img[joint_type['right']].copy()

        joint_cam[:, 0] = - joint_cam[:, 0]
        joint_cam[joint_type['right']], joint_cam[joint_type['left']] = joint_cam[joint_type['left']].copy(), \
                                                                        joint_cam[
                                                                            joint_type['right']].copy()

        joint_cam_valid[joint_type['right']], joint_cam_valid[joint_type['left']] = joint_cam_valid[
                                                                                        joint_type['left']].copy(), \
                                                                                    joint_cam_valid[
                                                                                        joint_type['right']].copy()
        joint_img_valid[joint_type['right']], joint_img_valid[joint_type['left']] = joint_img_valid[
                                                                                        joint_type['left']].copy(), \
                                                                                    joint_img_valid[
                                                                                        joint_type['right']].copy()
        # hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()

    for i in range(joint_num // 2):
        joint_img[:21][i, :2] = trans_point2d(joint_img[:21][i, :2], trans_r) if trans_r is not None else joint_img[
                                                                                                          :21][i,
                                                                                                          :2]
        joint_img[21:][i, :2] = trans_point2d(joint_img[21:][i, :2], trans_l) if trans_l is not None else joint_img[
                                                                                                          21:][i,
                                                                                                          :2]

    joint_cam = rot_mat.dot(
        joint_cam.transpose(1, 0)
    ).transpose()



    if hand_side == "right" and img_r is not None:

        mask_img_l_amodal_crop = np.zeros([256, 256])
        mask_img_l_visible_crop = np.zeros([256, 256])
        if bbx_l is not None:

            if mask_img_l_amodal is not None:
                mask_img_l_amodal_crop, _, _ = generate_patch_mask(mask_img_l_amodal, bbx_r, do_flip, scale, rot,
                                                                   config['input_img_shape'])
            if mask_img_l_visible is not None:
                mask_img_l_visible_crop, _, _ = generate_patch_mask(mask_img_l_visible, bbx_r, do_flip, scale, rot,
                                                                    config['input_img_shape'])


        # print("here")
        # for data in [img_r, joint_cam[joint_type[hand_side]], joint_img[joint_type[hand_side]], \
        #        joint_cam_valid[
        #            joint_type[hand_side]], joint_img_valid[joint_type[hand_side]], inv_trans_r,mask_img_r_amodal_crop,mask_img_r_visible_crop,mask_img_l_amodal_crop,mask_img_l_visible_crop]:
        #     if data is None:
        #         print("woc")
        return img_r, joint_cam[joint_type[hand_side]], joint_img[joint_type[hand_side]], \
               joint_cam_valid[
                   joint_type[hand_side]], joint_img_valid[joint_type[hand_side]], inv_trans_r,mask_img_r_amodal_crop,mask_img_r_visible_crop,mask_img_l_amodal_crop,mask_img_l_visible_crop
    # img, joint_cam, joint_img, joint_cam_valid, joint_img_valid, inv_trans, right_amodal_mask, right_visible_mask, left_amodal_mask, left_visible_mask
    elif hand_side == "left" and img_l is not None:
        print("woc")
        assert 0,"error"
    else:

        # print("another side is used!")
        another_side = "right" if hand_side == "left" else "left"

        another_img = img_r if hand_side == "left" else img_l


        another_inv_trans = inv_trans_r if hand_side == "left" else inv_trans_l
        assert another_img is not None, print('Another side is also None ')

        another_joint_cam = joint_cam[joint_type[another_side]].copy()
        another_joint_img = joint_img[joint_type[another_side]].copy()

        another_amodal_mask_crop = mask_img_l_amodal_crop if another_side == "left" else mask_img_r_amodal_crop # to be right
        another_visible_mask_crop = mask_img_l_visible_crop if another_side == "left" else mask_img_r_visible_crop

        # the_other_amodal_mask_crop = mask_img_r_amodal_crop if another_side == "left" else mask_img_l_amodal_crop # to be left
        # the_other_visible_mask_crop = mask_img_r_visible_crop if another_side == "left" else mask_img_l_visible_crop


        want_img = another_img.copy()[:, ::-1, :]
        want_joint_cam = another_joint_cam.copy()
        want_joint_cam[:, 0] = -want_joint_cam[:, 0]

        want_amodal_mask_crop = another_amodal_mask_crop.copy()[:, ::-1] #  right
        want_visible_mask_crop = another_visible_mask_crop.copy()[:, ::-1]

        not_want_amodal_mask_crop = np.zeros([256, 256])#  left
        not_want_visible_mask_crop = np.zeros([256, 256])

        # not_want_amodal_mask_crop = the_other_amodal_mask_crop.copy()[:, ::-1]  # left
        # not_want_visible_mask_crop = the_other_visible_mask_crop.copy()[:, ::-1]

        want_joint_img = another_joint_img.copy()
        want_joint_img[:, 0] = want_img.shape[1] - want_joint_img[:, 0] - 1

        want_joint_cam_valid = joint_cam_valid[joint_type[another_side]]
        want_joint_img_valid = joint_img_valid[joint_type[another_side]]

        return want_img, want_joint_cam, want_joint_img, want_joint_cam_valid, want_joint_img_valid, another_inv_trans,want_amodal_mask_crop,want_visible_mask_crop,not_want_amodal_mask_crop, not_want_visible_mask_crop,

    # img, joint_cam, joint_img, joint_cam_valid, joint_img_valid, inv_trans, right_amodal_mask, right_visible_mask, left_amodal_mask, left_visible_mask











def transform_input_to_output_space(joint_img, joint_cam,
                                    joint_cam_valid, joint_img_valid,  config
                                    ):
    joint_cam = joint_cam.copy()[config['IH2SNAP']];
    joint_img = joint_img.copy()[config['IH2SNAP']];
    joint_cam_valid = joint_cam_valid.copy()[config['IH2SNAP']]
    joint_img_valid = joint_img_valid.copy()[config['IH2SNAP']]

    delta_valid = [
        joint_cam_valid[i] * joint_cam_valid[config['SNAP_PARENT'][i]]
        for i in range(21)
    ]
    delta_valid = np.array(delta_valid)

    njoints = joint_cam.shape[0]
    ''' prepare heat maps H '''
    hm = np.zeros(
        (njoints, config['output_hm_shape'][0], config['output_hm_shape'][1]),
        dtype='float32'
    )  # (CHW)
    hm_vaild = np.ones(njoints, dtype='float32')
    # '''joint_img.shape= (21, 2) to locationmap(21,64,64)'''
    for i in range(njoints):
        kp = (
                (joint_img[i] / config['input_img_shape'][0]) * config['output_hm_shape'][0]
        ).astype(np.int32)  # kp uv: [0~256] -> [0~64]
        hm[i], aval = gen_heatmap(hm[i], kp, config['sigma'])
        hm_vaild[i] *= aval

    joint_root = joint_cam[config['SNAP_ROOT_ID']]
    joint_ref = joint_cam[config['SNAP_REF_ID']]
    ref_bone = np.linalg.norm(joint_root - joint_ref) + 1e-6
    ref_bone = np.atleast_1d(ref_bone)

    '''prepare location maps L'''
    jointR = joint_cam - joint_root[np.newaxis, :]  # root relative
    jointRS = jointR / ref_bone  # scale invariant
    # '''jointRS.shape= (21, 3) to locationmap(21,3,64,64)'''
    location_map = jointRS[:, :, np.newaxis, np.newaxis].repeat(64, axis=-2).repeat(64, axis=-1)

    '''prepare delta maps D'''
    kin_chain = [
        jointRS[i] - jointRS[config['SNAP_PARENT'][i]]
        for i in range(21)
    ]
    kin_chain = np.array(kin_chain)  # id 0's parent is itself #21*3
    kin_len = np.linalg.norm(
        kin_chain, ord=2, axis=-1, keepdims=True  # 21*1
    ) + 1e-6
    kin_chain[1:] = kin_chain[1:] / kin_len[1:]
    # '''kin_chain(21, 3) to delta_map(21,3,64,64)'''
    delta_map = kin_chain[:, :, np.newaxis, np.newaxis].repeat(64, axis=-2).repeat(64, axis=-1)

    return joint_img, joint_cam, joint_cam_valid, joint_img_valid, delta_valid, hm, hm_vaild, delta_map, location_map


def render_gaussian_heatmap(joint_coord):
    x = torch.arange(cfg.output_hm_shape[0])
    y = torch.arange(cfg.output_hm_shape[1])
    # z = torch.arange(cfg.output_hm_shape[0])
    # zz, yy, xx = torch.meshgrid(z, y, x)
    yy, xx = torch.meshgrid(y, x)
    xx = xx[None, None, :, :].cuda().float();
    yy = yy[None, None, :, :].cuda().float();
    # zz = zz[None, None, :, :, :].cuda().float();

    x = joint_coord[:, :, 0, None, None];
    y = joint_coord[:, :, 1, None, None];
    # z = joint_coord[:, :, 2, None, None, None];
    # heatmap = torch.exp(
    #     -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
    heatmap = torch.exp(
        -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
    heatmap = heatmap * 255
    return heatmap


def ori_render_gaussian_heatmap(joint_coord):
    # x = torch.arange(cfg.output_hm_shape[2])
    # y = torch.arange(cfg.output_hm_shape[1])
    # z = torch.arange(cfg.output_hm_shape[0])

    x = torch.arange(64)
    y = torch.arange(64)
    z = torch.arange(64)

    zz, yy, xx = torch.meshgrid(z, y, x)
    xx = xx[None, None, :, :, :].cuda().float();
    yy = yy[None, None, :, :, :].cuda().float();
    zz = zz[None, None, :, :, :].cuda().float();

    x = joint_coord[:, :, 0, None, None, None];
    y = joint_coord[:, :, 1, None, None, None];
    z = joint_coord[:, :, 2, None, None, None];
    heatmap = torch.exp(
        -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
    heatmap = heatmap * 255
    return heatmap


if __name__ == '__main__':
    joint = np.arange(63).reshape(21, 3)
    joint = np.expand_dims(joint, axis=0)
    joint = torch.from_numpy(joint).cuda()

    ori_heatmap = ori_render_gaussian_heatmap(joint)
    print("ori_hm=", ori_heatmap.shape)

    kp = np.arange(42).reshape(21, -1)
    kp = np.expand_dims(kp, axis=0)
    kp = torch.from_numpy(kp).cuda()
    kp_heatmap = render_gaussian_heatmap(kp)
    print("kp_heatmap=", kp_heatmap.shape)

    kp_hm_np = kp_heatmap.squeeze(0).cpu().numpy()
    tmp = np.zeros((64, 64))
    for k in range(kp_hm_np.shape[0]):
        tmp += kp_hm_np[k] * 64
    plt.imshow(tmp)
    plt.title('2D heatmap')
    plt.show()


def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1


def get_bbox(joint_img, joint_valid):
    x_img = joint_img[:, 0][joint_valid == 1];
    y_img = joint_img[:, 1][joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, input_img_shape, factor=1.25):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * factor
    bbox[3] = h * factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


# my modification
def mask2bbox(mask):
    assert len(mask.shape) == 2, "Mask's shape is wrong!"

    rmin, cmin = np.min(mask, axis=0).astype(np.float32)
    rmax, cmax = np.max(mask, axis=0).astype(np.float32)

    w = cmax - cmin
    h = rmax - rmin

    return np.array([cmin, rmin, w, h])  # format:(x,y,w,h)

    # # aspect ratio preserving bbox
    # w = bbox[2]
    # h = bbox[3]
    # c_x = bbox[0] + w/2.
    # c_y = bbox[1] + h/2.
    # aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    # if w > aspect_ratio * h:
    #     h = w / aspect_ratio
    # elif w < aspect_ratio * h:
    #     w = h * aspect_ratio
    # bbox[2] = w*1.25
    # bbox[3] = h*1.25
    # bbox[0] = c_x - bbox[2]/2.
    # bbox[1] = c_y - bbox[3]/2.
    #
    # return bbox


def corner2bbox(corner_):
    corner = corner_.copy()
    assert len(corner.shape) == 1, "corner's shape is wrong!"
    top, left, bottom, right = corner

    x = left
    y = top
    w = right - left
    h = bottom - top

    return np.array([x, y, w, h])  # format:(x,y,w,h)


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bbox = bbox.copy()

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:  # 若flip，先flip，后旋转平移
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans


def generate_patch_mask(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width = img.shape

    bbox = bbox.copy()

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:  # 若flip，先flip，后旋转平移
        img = img[:, ::-1]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]
