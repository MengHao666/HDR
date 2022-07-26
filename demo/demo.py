import cv2
import yaml
import os.path as osp
import sys
sys.path.insert(0,'E:\PycharmProjects\HDR')
from mmseg.apis import inference_segmentor, init_segmentor
import models
import numpy as np
import torch
from utils.preprocessing import process_bbox, generate_patch_image, mask2bbox, augmentation_demo
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image



def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:, c, :, :].mul_(d).add_(m)

    th = tensor * 255
    th = torch.clamp(th, 0, 255)
    npdata = th.detach().cpu().numpy().squeeze()  # NCHW

    # print(npdata.shape)
    npdata = np.transpose(npdata, (1, 2, 0))
    return npdata


# demo config file
cfg_file = './experiments/demo.yaml'
with open(cfg_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# segmentor
seg_cfg_file = "./configs/segformer/segformer_mit-b5_256x256_interhand_1101.py"
seg_model = init_segmentor(
    config=seg_cfg_file,
    checkpoint='./demo_work_dirs/Interhand_seg/iter_237500.pth',
)

# DRNet
exp_path_dr = './demo_work_dirs/TDR_fintune/'
load_iter_dr = 138000
dr_model = models.__dict__[config['dr_model']['algo']](config['dr_model'],
                                                       dist_model=False,
                                                       demo=True)

dr_model.load_state(osp.join(exp_path_dr, "checkpoints"),
                    load_iter_dr, resume=False, clear_module=True)
dr_model.switch_to('eval')
dr_model.cuda()

# Hand Pose Estimator
exp_path_hpe = './demo_work_dirs/All_train_SingleRightHand'
load_iter_hpe = 261000
hpe_model = models.__dict__[config['hpe_model']['algo']](config['hpe_model'],
                                                         dist_model=False,
                                                         demo=True, )
hpe_model.load_state(osp.join(exp_path_hpe, "checkpoints"),
                     load_iter_hpe, resume=False, clear_module=True)
hpe_model.switch_to('eval')
hpe_model.cuda()

# Prepare one sample (img and joints_2D_GT), they are from Tzionas dataset
joint_num = 28
PADDING_FACTOR = 0.5
PROCESSED_BBOX_FACTOR = 2
reslu = [480, 640]

both_kp_bbox = [reslu[1], reslu[0], 0, 0]
both_kp_bbox_padding = both_kp_bbox.copy()
process_both_kp_bbox_paddings = []
joints_2D_GT = np.loadtxt('./demo/Tzionas_dataset-02-1-joints_2D_GT-100.txt')[:, 1:]
joints_2D_GT_valid = np.sum(joints_2D_GT[:, 1:], axis=1) > 0

for i in range(joint_num):
    if joints_2D_GT_valid[i]:
        both_kp_bbox[0] = min(both_kp_bbox[0], joints_2D_GT[i][0])
        both_kp_bbox[1] = min(both_kp_bbox[1], joints_2D_GT[i][1])
        both_kp_bbox[2] = max(both_kp_bbox[2], joints_2D_GT[i][0])
        both_kp_bbox[3] = max(both_kp_bbox[3], joints_2D_GT[i][1])
both_kp_bbox[2] -= both_kp_bbox[0]
both_kp_bbox[3] -= both_kp_bbox[1]

both_kp_bbox_padding = both_kp_bbox.copy()
both_kp_bbox_padding[0] += reslu[1] * PADDING_FACTOR  # for image padding
both_kp_bbox_padding[1] += reslu[0] * PADDING_FACTOR  # for image padding
process_both_kp_bbox_padding = process_bbox(both_kp_bbox_padding.copy(), input_img_shape=(256, 256),
                                            factor=PROCESSED_BBOX_FACTOR)
process_both_kp_bbox_padding = np.array(process_both_kp_bbox_padding).astype(np.int32)

img_path = './demo/Tzionas_dataset-02-1-rgb-100.png'
img_ = np.array(Image.open(img_path).convert('RGB'))
img_height, img_width, _ = img_.shape

img_padding = cv2.copyMakeBorder(img_.copy(), int(img_height * PADDING_FACTOR),
                                 int(img_height * PADDING_FACTOR),
                                 int(img_width * PADDING_FACTOR),
                                 int(img_width * PADDING_FACTOR), cv2.BORDER_CONSTANT, value=[0, 0, 0])
crop_img_padding, trans, inv_trans = generate_patch_image(img_padding, process_both_kp_bbox_padding,
                                                          do_flip=False, scale=1,
                                                          rot=0,
                                                          out_shape=[256, 256])

# visual check
vis = 0
if vis:
    fig = plt.figure(figsize=[50, 50])

    plt.subplot(1, 5, 1)
    plt.imshow(img_)
    plt.title('ori_Color')

    plt.subplot(1, 5, 2)
    plt.imshow(img_)
    plt.title('ori_Color+2D annotations')
    plt.text(joints_2D_GT[0][0], joints_2D_GT[0][1], '0', color="w", fontsize=1)
    for p in range(28):
        if joints_2D_GT_valid[p]:
            plt.plot(joints_2D_GT[p][0], joints_2D_GT[p][1], 'bo', markersize=0.5)
            plt.text(joints_2D_GT[p][0], joints_2D_GT[p][1], '{0}'.format(p), color="w", fontsize=5)
    plt.gca().add_patch(
        plt.Rectangle((both_kp_bbox[0], both_kp_bbox[1]), both_kp_bbox[2], both_kp_bbox[3], linewidth=0.5,
                      edgecolor="yellow",
                      facecolor='none', label="both_kp_bbox"))

    plt.subplot(1, 5, 3)
    plt.imshow(img_padding)
    plt.title(' img_padding')

    plt.subplot(1, 5, 4)
    plt.imshow(img_padding)
    plt.title(' img_padding+2D bbox(padding)')
    plt.gca().add_patch(
        plt.Rectangle((both_kp_bbox_padding[0], both_kp_bbox_padding[1]), both_kp_bbox_padding[2],
                      both_kp_bbox_padding[3], linewidth=1,
                      edgecolor="yellow",
                      facecolor='none', label="both_kp_bbox_padding"))
    plt.gca().add_patch(
        Rectangle((process_both_kp_bbox_padding[0], process_both_kp_bbox_padding[1]),
                  process_both_kp_bbox_padding[2],
                  process_both_kp_bbox_padding[3],
                  linewidth=1,
                  edgecolor='r',
                  facecolor='none', label="processed_both_kp_bbox"))

    plt.subplot(1, 5, 5)
    plt.imshow(crop_img_padding.copy().astype(np.int32))
    plt.title('crop_img_padding')

    plt.show()

rgb = torch.from_numpy(crop_img_padding.astype(np.float32).transpose((2, 0, 1)) / 255.)
img_transform = transforms.Compose([
    transforms.Normalize(config['data']['data_mean'], config['data']['data_std'])
])
rgb = img_transform(rgb).unsqueeze(0)  # CHW
seg_result = inference_segmentor(seg_model, rgb.cuda())

# save mask
l_amodal, l_visible, r_amodal, r_visible = seg_result
l_amodal, l_visible, r_amodal, r_visible = l_amodal.squeeze().astype(np.float32), l_visible.squeeze().astype(
    np.float32), r_amodal.squeeze().astype(np.float32), r_visible.squeeze().astype(np.float32)

# print(l_amodal.shape, l_visible.shape, r_amodal.shape, r_visible.shape)

x, y, w, h = process_both_kp_bbox_padding.astype(np.int32)
l_amodal_resize = cv2.resize(l_amodal, (w, h), interpolation=cv2.INTER_NEAREST)
l_visible_resize = cv2.resize(l_visible, (w, h), interpolation=cv2.INTER_NEAREST)
r_amodal_resize = cv2.resize(r_amodal, (w, h), interpolation=cv2.INTER_NEAREST)
r_visible_resize = cv2.resize(r_visible, (w, h), interpolation=cv2.INTER_NEAREST)

#### visual check segmentation result
vis = 0
if vis:
    fig = plt.figure(figsize=[20, 10])
    plt.suptitle('visual check segmentation result')

    plt.subplot(1, 5, 1)
    plt.imshow(crop_img_padding.astype(np.int32))
    plt.title('crop_img_padding')

    plt.subplot(1, 5, 2)
    plt.imshow(l_amodal_resize, cmap='gray')
    plt.title('l_amodal_resize')

    plt.subplot(1, 5, 3)
    plt.imshow(l_visible_resize, cmap='gray')
    plt.title('l_visible_resize')

    plt.subplot(1, 5, 4)
    plt.imshow(r_amodal_resize, cmap='gray')
    plt.title('r_amodal_resize')

    plt.subplot(1, 5, 5)
    plt.imshow(r_visible_resize, cmap='gray')
    plt.title('r_visible_resize')

    plt.show()

padding_mask_data = {"right":
                         {"all": None, "visible": None, "occluded": None},
                     "left":
                         {"all": None, "visible": None, "occluded": None}}
padding_mask_data["right"]["all"] = r_amodal_resize.copy()
padding_mask_data["left"]["all"] = l_amodal_resize.copy()
padding_mask_data["right"]["visible"] = r_visible_resize.copy()
padding_mask_data["left"]["visible"] = l_visible_resize.copy()

padding_mask_loc = {"right":
                        {"all": None, "visible": None, },
                    "left":
                        {"all": None, "visible": None}}
# get mask location from mask
for which_hand in ("right", "left"):
    for save_type in ["visible", "all"]:
        if padding_mask_data[which_hand][save_type] is not None:
            l_tuple = np.where(padding_mask_data[which_hand][save_type] == 1)
            location = np.concatenate((l_tuple[0][:, None], l_tuple[1][:, None]), axis=1).astype(np.int32)
            location[:, 0] += y
            location[:, 1] += x
            padding_mask_loc[which_hand][save_type] = location.copy()

# get boundingbox form mask
mask_bbxs = {}
for which_hand in ['right', 'left']:
    if (padding_mask_loc[which_hand]["all"] is not None) and (padding_mask_loc[which_hand][
                                                                  "all"].size != 0):
        mask_bbxs[which_hand] = mask2bbox(padding_mask_loc[which_hand]["all"].copy())
    else:
        mask_bbxs[which_hand] = None

final_bboxs = mask_bbxs.copy()
for k, v in final_bboxs.items():
    if v is not None:
        final_bboxs[k] = process_bbox(v, input_img_shape=[256, 256], factor=2.0, )

####
# set which hand side
hand_side = 'right'
# hand_side = 'left'
config['hand_side'] = hand_side
config['input_img_shape'] = [256, 256]
used_dr_output = "combination"
dilate_kernerl_size = 9

### get data for DRNet
img_padding = cv2.copyMakeBorder(img_.copy(), int(img_height * PADDING_FACTOR),
                                 int(img_height * PADDING_FACTOR),
                                 int(img_width * PADDING_FACTOR),
                                 int(img_width * PADDING_FACTOR), cv2.BORDER_CONSTANT, value=[0, 0, 0])

img, inv_trans, amodal_mask, visible_mask, another_amodal_mask, another_visible_mask = augmentation_demo(
    img_padding,
    final_bboxs,
    config, padding_mask_loc)

ori_img = img.copy()

amodal_mask, visible_mask, another_amodal_mask, another_visible_mask = amodal_mask.astype(
    np.int32), visible_mask.astype(np.int32), another_amodal_mask.astype(np.int32), another_visible_mask.astype(
    np.int32)

######
##### this
this_invisible_mask = amodal_mask * (1 - visible_mask)  # intersection
this_invisible_mask = this_invisible_mask.astype(np.float32)

this_visible_mask_ = np.clip(visible_mask + (1 - amodal_mask), 0, 1)
this_visible_mask = this_visible_mask_[np.newaxis, :, :]
this_visible_mask = this_visible_mask.astype(np.float32)

this_modal_mask = visible_mask[np.newaxis, :, :]
this_modal_mask = this_modal_mask.astype(np.float32)

###### another
another_invisible_mask = (1 - amodal_mask) * another_amodal_mask  # intersection
another_invisible_mask = another_invisible_mask.astype(np.float32)

another_visible_mask_ = 1 - another_invisible_mask
another_visible_mask = another_visible_mask_[np.newaxis, :, :]
another_visible_mask = another_visible_mask.astype(np.float32)

another_modal_mask = (1 - amodal_mask) * (1 - another_amodal_mask)[np.newaxis, :, :]
another_modal_mask = another_modal_mask.astype(np.float32)

# flip the data related for DRNet
if hand_side == "left":
    img = img.copy()[:, ::-1, :].copy()  # flipped

    r_invisible_mask = this_invisible_mask.copy()[:, ::-1].copy()
    r_visible_mask = this_visible_mask.copy()[:, :, ::-1].copy()
    r_modal_mask = this_modal_mask.copy()[:, :, ::-1].copy()

    l_invisible_mask = another_invisible_mask.copy()[:, ::-1].copy()
    l_visible_mask = another_visible_mask.copy()[:, :, ::-1].copy()
    l_modal_mask = another_modal_mask.copy()[:, :, ::-1].copy()

else:
    r_invisible_mask = this_invisible_mask
    r_visible_mask = this_visible_mask
    r_modal_mask = this_modal_mask

    l_invisible_mask = another_invisible_mask
    l_visible_mask = another_visible_mask
    l_modal_mask = another_modal_mask

r_visible_mask_tensor = torch.from_numpy(r_visible_mask)
l_visible_mask_tensor = torch.from_numpy(l_visible_mask)

rgb = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.)
rgb = img_transform(rgb)  # CHW

r_rgb_erased = rgb.clone()
r_rgb_erased = r_rgb_erased * r_visible_mask_tensor  # erase r rgb
l_rgb_erased = rgb.clone()
l_rgb_erased = l_rgb_erased * l_visible_mask_tensor  # erase l rgb

# print(type(r_rgb_erased))
# print(type(l_rgb_erased))
# print(type(r_visible_mask_tensor))
# print(type(l_visible_mask_tensor))
# print(type(r_modal_mask))
# print(type(l_modal_mask))

r_modal_mask = torch.from_numpy(r_modal_mask).float()
l_modal_mask = torch.from_numpy(l_modal_mask).float()

# inputs = {'rgb': rgb,
#           "r_rgb_erased": r_rgb_erased,
#           "l_rgb_erased": l_rgb_erased,
#           "r_visible_mask": r_visible_mask_tensor,
#           "l_visible_mask": l_visible_mask_tensor,
#           "r_modal_mask": r_modal_mask,
#           "l_modal_mask": l_modal_mask
#           }

inputs = {'rgb': rgb.unsqueeze(0),
          "r_rgb_erased": r_rgb_erased.unsqueeze(0),
          "l_rgb_erased": l_rgb_erased.unsqueeze(0),
          "r_visible_mask": r_visible_mask_tensor.unsqueeze(0),
          "l_visible_mask": l_visible_mask_tensor.unsqueeze(0),
          "r_modal_mask": r_modal_mask.unsqueeze(0),
          "l_modal_mask": l_modal_mask.unsqueeze(0)
          }

dr_model.set_input(inputs["r_rgb_erased"], inputs["l_rgb_erased"],
                   inputs["r_visible_mask"],
                   inputs["l_visible_mask"], inputs["r_modal_mask"],
                   inputs["l_modal_mask"])

deocclu_result = dr_model.forward_only(ret_loss=False, dilate_kernerl_size=dilate_kernerl_size)

if used_dr_output == "combination":
    deocclu_output_comp = deocclu_result['common_tensors'][3].clone()

elif used_dr_output == "direct":
    deocclu_output_comp = deocclu_result['common_tensors'][2].clone()

else:
    assert 0, "error"

deocclu_output_comp_img = unormalize(deocclu_output_comp.detach().cpu(), config['data']["data_mean"],
                                     config['data']["data_std"]).squeeze()
if hand_side == 'left':
    deocclu_output_comp_img = deocclu_output_comp_img[:, ::-1, :]

hpe_model.set_input({"rgb": deocclu_output_comp}, targets=None, meta_info={})
hpe_result = hpe_model.forward_only()

if hand_side != 'right':
    print("filp here !")
    hpe_result['joint_cam'][:, :, 0] = -hpe_result['joint_cam'][:, :, 0]
    hpe_result['joint_img'] = hpe_result['joint_img'] * 4.0

# visual check
vis = 1
if vis:
    fig = plt.figure(figsize=[20, 10])
    plt.suptitle(f'visual DRNet {hand_side}_hand result')

    plt.subplot(1, 3, 1)
    plt.imshow(ori_img.astype(np.int32))
    plt.title('ori_img')

    plt.subplot(1, 3, 2)
    plt.imshow(deocclu_output_comp_img.astype(np.int32))
    plt.title('deocclu_output_comp_img')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    big_size = 50
    hand_3d = hpe_result['joint_cam'].copy().squeeze()
    for i in range(1, 4):
        ax.scatter(hand_3d[i, 0], hand_3d[i, 1], hand_3d[i, 2])
    for i in range(5, 21):
        ax.scatter(hand_3d[i, 0], hand_3d[i, 1], hand_3d[i, 2])

    ax.scatter(hand_3d[0, 0], hand_3d[0, 1], hand_3d[0, 2], s=big_size * 1.5,
               c=config['data']["colors"]['AERO'], label='right')

    ax.scatter(hand_3d[4, 0], hand_3d[4, 1], hand_3d[4, 2], s=big_size, c="red", label='right')
    for li, color in zip(config['data']['snap_link_hand'], config['data']['colors_hand']):
        ax.plot(hand_3d[li][:, 0], hand_3d[li][:, 1], hand_3d[li][:, 2], c=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(-90, -90)
    ax.set_title('{}_hand 3D prediction'.format(hand_side), fontsize=12, color='r')

    plt.show()
