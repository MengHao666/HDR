import numpy as np


def global_align(gtj0, prj0, cfg):
    gtj = gtj0.copy().astype(np.float32)
    prj = prj0.copy().astype(np.float32)

    # gtj :21*3
    # prj 21*3
    pred_align = prj.copy()

    pred_ref_bone_len = np.linalg.norm(prj[cfg['SNAP_REF_ID']] - prj[cfg['SNAP_ROOT_ID']]) + 1e-6
    gt_ref_bone_len = np.linalg.norm(gtj[cfg['SNAP_REF_ID']] - gtj[cfg['SNAP_ROOT_ID']]) + 1e-6

    scale = gt_ref_bone_len / pred_ref_bone_len

    for j in range(21):
        pred_align[j] = gtj[cfg['SNAP_ROOT_ID']] + scale * (prj[j] - prj[cfg['SNAP_ROOT_ID']])

    return pred_align
