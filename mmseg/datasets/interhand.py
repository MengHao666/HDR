import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset



@DATASETS.register_module()
class InterHand_Dataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'foreground')

    PALETTE = [[255, 0, 0], [0, 255, 0]]

    def __init__(self, **kwargs):
        # print(kwargs["ann_dir"])
        # exit(0)
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         **kwargs)
        assert osp.exists(self.img_dir)