from pycocotools.coco import COCO
import os
import argparse
import sys
import shutil
from collections import OrderedDict
from SPcv.split_and_merge_image import COCOMergeResult


def merge_det_result(json_result_file, corner_gt_file, merged_gt_file, merge_nms_th=1.0):

    print('merge result from sub image', json_result_file, merged_gt_file)
    if merge_nms_th >= 1.0 - 1e-6:
        use_nms = False
    else:
        use_nms = True
    _, merged_json_result_file = COCOMergeResult(use_nms=use_nms, nms_th=merge_nms_th)(
        corner_gt_file,
        json_result_file,
        os.path.split(json_result_file)[0],  # dir
        merged_gt_file
    )
    coco_gt = COCO(merged_gt_file)
    return coco_gt, merged_json_result_file


