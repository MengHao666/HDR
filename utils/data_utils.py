import random

import numpy as np
from PIL import Image
import io
import cv2
import torch


class SynthesizeSetter(object):

    def __init__(self, config, forward_only=True):
        if not forward_only:
            self.min_cut_ratio = config["syn_setter"]['min_cut_ratio']
            self.max_cut_ratio = config["syn_setter"].get('max_cut_ratio', 1.0)
            self.base_aug = config['base_aug']

    def get_aug_config(self):
        # trans_factor = 0.15
        # scale_factor = 0.25
        # rot_factor = 45
        # color_factor = 0.2

        trans_factor = self.base_aug["trans"]
        scale_factor = self.base_aug["scale"]
        rot_factor = self.base_aug["rot"]

        trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
        scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
        rot = np.clip(np.random.randn(), -2.0,
                      2.0) * rot_factor if random.random() <= 0.6 else 0
        do_flip = random.random() <= 0.5
        # c_up = 1.0 + color_factor
        # c_low = 1.0 - color_factor
        #
        # color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
        # color_scale = random.uniform(c_low, c_up)

        return trans, scale, rot, do_flip

    def augmentation(self, img, mask, bbox, hand_side, color_scale):
        img = img.copy();

        # if mode == 'train':
        #     translation, scale, rot, do_flip, color_scale = self.get_aug_config()
        # else:
        #     translation, scale, rot, do_flip, color_scale = [0., 0.], 1.0, 0.0, False, 1.0

        translation, scale, rot, do_flip = self.get_aug_config()

        if hand_side == "left":
            # print("**translation=",translation)
            # print("****translation=",type(translation))
            translation = [item * 0.2 for item in translation]
            # print("*****translation=", translation)
        do_flip = False  # NO flip here!
        aug_list = [translation, scale, rot]

        bbox[0] = bbox[0] + bbox[2] * translation[0]
        bbox[1] = bbox[1] + bbox[3] * translation[1]
        img, trans_matrix, inv_trans_matrix = generate_patch_image(img, bbox, do_flip, scale, rot, [256, 256])
        # img = np.clip(img * color_scale, 0, 255)
        img = np.clip(img * color_scale[None, None, :], 0, 255)

        mask, _, _ = generate_patch_image(mask, bbox, do_flip, scale, rot, [256, 256])

        return img, mask, aug_list

    def augmentation_with_auglist(self, img, mask, bbox, aug_list):

        translation, scale, rot, color_scale = aug_list

        img = img.copy();

        # if mode == 'train':
        #     translation, scale, rot, do_flip, color_scale = self.get_aug_config()
        # else:
        #     translation, scale, rot, do_flip, color_scale = [0., 0.], 1.0, 0.0, False, 1.0

        # translation, scale, rot, do_flip = self.get_aug_config()

        # if hand_side == "left":
        #     # print("**translation=",translation)
        #     # print("****translation=",type(translation))
        #     translation = [item * 0.2 for item in translation]
        #     # print("*****translation=", translation)
        do_flip = False  # NO flip here!
        # aug_list = [translation, scale, rot]

        bbox[0] = bbox[0] + bbox[2] * translation[0]
        bbox[1] = bbox[1] + bbox[3] * translation[1]
        img, trans_matrix, inv_trans_matrix = generate_patch_image(img, bbox, do_flip, scale, rot, [256, 256])
        # img = np.clip(img * color_scale, 0, 255)
        img = np.clip(img * color_scale[None, None, :], 0, 255)

        mask, _, _ = generate_patch_image(mask, bbox, do_flip, scale, rot, [256, 256])

        return img, mask, aug_list

    def __call__(self, r_img, r_amodal_mask, l_img, l_amodal_mask, phase, max_iter=100):
        # # flip
        # change_side = False
        # if self.base_aug['flip'] and np.random.rand() > 0.5:
        #     r_img, l_img = l_img[:, ::-1], r_img[:, ::-1]
        #     r_amodal_mask, l_amodal_mask = l_amodal_mask[:, ::-1], r_amodal_mask[:, ::-1]
        #     change_side = True

        r_img_aug, r_amodal_mask_aug, l_img_aug, l_amodal_mask_aug, intersection, occlusion_ratio = [None for _ in
                                                                                                     range(6)]

        # color_factor = self.base_aug["color"]
        # c_up = 1.0 + color_factor
        # c_low = 1.0 - color_factor
        # color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
        color_scale = self.get_color_scale()

        for i in range(max_iter):
            r_img_aug, r_amodal_mask_aug, r_aug_list = self.augmentation(r_img, r_amodal_mask, [0, 0, 255, 255],
                                                                         hand_side="right", color_scale=color_scale)
            l_img_aug, l_amodal_mask_aug, l_aug_list = self.augmentation(l_img, l_amodal_mask, [0, 0, 255, 255],
                                                                         hand_side="left", color_scale=color_scale)

            r_aug_list.append(color_scale)
            r_aug_list.append(change_side)

            l_aug_list.append(color_scale)
            l_aug_list.append(change_side)

            intersection = r_amodal_mask_aug * l_amodal_mask_aug
            occlusion_ratio = intersection.sum() / (l_amodal_mask_aug.sum())
            if self.min_cut_ratio <= occlusion_ratio < self.max_cut_ratio:
                # print("I=",i)
                # print("occlusion_ratio=",occlusion_ratio)
                break
        return r_img_aug, r_amodal_mask_aug, l_img_aug, l_amodal_mask_aug, intersection, occlusion_ratio, r_aug_list, l_aug_list

    def get_color_scale(self):
        color_factor = 0.2

        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

        return color_scale

    def forward_only(self, r_img, r_amodal_mask, l_img, l_amodal_mask, r_aug_list, l_aug_list,phase):

        # r_translation, r_scale, r_rot, r_color_scale = r_aug_list
        # l_translation, l_scale, l_rot, l_color_scale = l_aug_list

        # assert (r_color_scale == l_color_scale).all()
        if phase=="train":
            r_color_scale = self.get_color_scale()
            l_color_scale = r_color_scale
        else:
            r_color_scale =np.array([1.0, 1.0, 1.0])
            l_color_scale =np.array([1.0, 1.0, 1.0])

        r_aug_list[-1] = r_color_scale
        l_aug_list[-1] = l_color_scale

        r_img_aug, r_amodal_mask_aug, r_aug_list = self.augmentation_with_auglist(r_img, r_amodal_mask,
                                                                                  [0, 0, 255, 255],
                                                                                  r_aug_list)
        l_img_aug, l_amodal_mask_aug, l_aug_list = self.augmentation_with_auglist(l_img, l_amodal_mask,
                                                                                  [0, 0, 255, 255],
                                                                                  l_aug_list)
        return r_img_aug, r_amodal_mask_aug, l_img_aug, l_amodal_mask_aug


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()

    if len(img.shape) == 3:
        img_height, img_width, img_channels = img.shape
    else:
        assert len(img.shape) == 2
        img_height, img_width = img.shape  # for mask use

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        if len(img.shape) == 3:
            img = img[:, ::-1, :]
        else:
            assert len(img.shape) == 2
            img = img[:, ::-1]  # for mask use
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans


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


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff).convert('RGB')


def pil_loader_mask(mask_str):
    buff = io.BytesIO(mask_str)
    return Image.open(buff)


def combine_bbox(bboxes):
    '''
    bboxes: Nx4, xywh
    '''
    l = bboxes[:, 0].min()
    u = bboxes[:, 1].min()
    r = (bboxes[:, 0] + bboxes[:, 2]).max()
    b = (bboxes[:, 1] + bboxes[:, 2]).max()
    w = r - l
    h = b - u
    return np.array([l, u, w, h])


def mask_to_bbox(mask):
    mask = (mask == 1)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()]  # xywh


def bbox_iou(b1, b2):
    '''
    b: (x1,y1,x2,y2)
    '''
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.
    else:
        interArea = (rx - lx) * (dy - uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)


def crop_padding(img, roi, pad_value):
    '''
    img: HxW or HxWxC np.ndarray
    roi: (x,y,w,h)
    pad_value: [b,g,r]
    '''
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x, y, w, h = roi
    x, y, w, h = int(x), int(y), int(w), int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x, y, x + w, y + h), (0, 0, W, H)) > 0:
        output[max(-y, 0):min(H - y, h), max(-x, 0):min(W - x, w), :] = img[max(y, 0):min(y + h, H),
                                                                        max(x, 0):min(x + w, W), :]
    if need_squeeze:
        output = np.squeeze(output)
    return output


def place_eraser(inst, eraser, min_overlap, max_overlap):
    assert len(inst.shape) == 2
    assert len(eraser.shape) == 2
    assert min_overlap <= max_overlap
    h, w = inst.shape
    overlap = np.random.uniform(low=min_overlap, high=max_overlap)
    offx = np.random.uniform(overlap - 1, 1 - overlap)
    if offx < 0:
        over_y = overlap / (offx + 1)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    else:
        over_y = overlap / (1 - offx)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    assert offy > -1 and offy < 1
    bbox = (int(offx * w), int(offy * h), w, h)
    shift_eraser = crop_padding(eraser, bbox, pad_value=(0,))
    assert inst.max() <= 1, "inst max: {}".format(inst.max())
    assert shift_eraser.max() <= 1, "eraser max: {}".format(eraser.max())
    ratio = ((inst == 1) & (shift_eraser == 1)).sum() / float((inst == 1).sum() + 1e-5)
    return shift_eraser, ratio


def place_eraser_in_ratio(inst, eraser, min_overlap, max_overlap, min_ratio, max_ratio, max_iter):
    for i in range(max_iter):
        shift_eraser, ratio = place_eraser(inst, eraser, min_overlap, max_overlap)
        if ratio >= min_ratio and ratio < max_ratio:
            break
    return shift_eraser


def scissor_mask(inst, eraser, min_overlap, max_overlap):
    assert len(inst.shape) == 2
    assert len(eraser.shape) == 2
    assert min_overlap <= max_overlap
    h, w = inst.shape
    overlap = np.random.uniform(low=min_overlap, high=max_overlap)
    offx = np.random.uniform(overlap - 1, 1 - overlap)
    if offx < 0:
        over_y = overlap / (offx + 1)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    else:
        over_y = overlap / (1 - offx)
        if np.random.rand() > 0.5:
            offy = over_y - 1
        else:
            offy = 1 - over_y
    assert offy > -1 and offy < 1
    bbox = (int(offx * h), int(offy * h), w, h)
    shift_eraser = crop_padding(eraser, bbox, pad_value=(0,)) > 0.5  # bool
    ratio = ((inst > 0.5) & shift_eraser).sum() / float((inst > 0.5).sum())
    inst_erased = inst.copy()
    inst_erased[shift_eraser] = 0
    return inst_erased, shift_eraser, ratio


def scissor_mask_force(inst, eraser, min_overlap, max_overlap, min_ratio, max_ratio, max_iter):
    for i in range(max_iter):
        inst_erased, shift_eraser, ratio = scissor_mask(inst, eraser, min_overlap, max_overlap)
        if ratio >= min_ratio and ratio < max_ratio:
            break
    return inst_erased, shift_eraser


def mask_aug(mask, config):
    '''
    mask: uint8 (HxW), 0 (bg), 128 (ignore), 255 (fg)
    '''
    oldh, oldw = mask.shape
    if config['flip'] and np.random.rand() > 0.5:
        mask = mask[:, ::-1]
    assert config['scale'][0] <= config['scale'][1]
    if not (config['scale'][0] == 1 and config['scale'][0] == 1):
        scale = np.random.uniform(config['scale'][0], config['scale'][1])
        newh, neww = int(scale * oldh), int(scale * oldw)
        mask = cv2.resize(mask, (neww, newh), interpolation=cv2.INTER_NEAREST)
        bbox = [(neww - oldw) // 2, (newh - oldh) // 2, oldw, oldh]
        mask = crop_padding(mask, bbox, pad_value=(0,))
    return mask


def base_aug(img, scis_img, config):
    '''
    img, scis_img: HW
    '''
    oldh, oldw = img.shape
    if config['flip'] and np.random.rand() > 0.5:
        img = img[:, ::-1]
        scis_img = scis_img[:, ::-1]
    assert config['scale'][0] <= config['scale'][1]
    scale = np.random.uniform(config['scale'][0], config['scale'][1])
    newh, neww = int(scale * oldh), int(scale * oldw)
    offx = int(oldw * np.random.uniform(config['shift'][0], config['shift'][1]))
    offy = int(oldh * np.random.uniform(config['shift'][0], config['shift'][1]))

    bbox = [(neww - oldw) // 2 - offx, (newh - oldh) // 2 - offy, oldw, oldh]
    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_NEAREST)
    img = crop_padding(img, bbox, pad_value=(0,))
    scis_img = cv2.resize(scis_img, (neww, newh), interpolation=cv2.INTER_NEAREST)
    scis_img = crop_padding(scis_img, bbox, pad_value=(0,))
    return img, scis_img


class EraserSetter(object):

    def __init__(self, config):
        self.min_overlap = config['min_overlap']
        self.max_overlap = config['max_overlap']
        self.min_cut_ratio = config['min_cut_ratio']
        self.max_cut_ratio = config.get('max_cut_ratio', 1.0)

    def __call__(self, inst, eraser):
        return place_eraser_in_ratio(inst, eraser, self.min_overlap,
                                     self.max_overlap, self.min_cut_ratio,
                                     self.max_cut_ratio, 100)


def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:, c, :, :].mul_(d).add_(m)
    return tensor


def th2np(th, dtype='image', transpose=False, rgb_cyclic=False, cfgs=None):
    assert dtype in ['image', 'mask']
    if dtype == 'image':
        assert cfgs is not None
        if cfgs is not None:
            th = unormalize(th.detach().cpu(), cfgs["data_mean"], cfgs["data_std"])
        else:
            print(cfgs is None)
            th = (th + 1.0) / 2.0
        th = th * 255
        th = torch.clamp(th, 0, 255)
        npdata = th.detach().cpu().numpy()  # NCHW
        if rgb_cyclic:
            npdata = npdata[:, ::-1, :, :]
        if transpose:
            npdata = npdata.transpose((0, 2, 3, 1))  # NHWC
    else:
        if th.ndim == 3:
            th = th.unsqueeze(1)
        if th.size(1) == 1:
            # npdata = th.detach().cpu().repeat(1, 3, 1, 1).numpy()  # NCHW
            npdata = th.detach().cpu().repeat(1, 3, 1, 1).numpy()  # NCHW
        else:
            npdata = th.detach().cpu().numpy()
        if transpose:
            npdata = npdata.transpose((0, 2, 3, 1))
    return npdata
