# Copyright (c) OpenMMLab. All rights reserved.
from .auto_augment import (AutoAugment, BrightnessTransform, ColorTransform,
                           ContrastTransform, EqualizeTransform, Rotate, Shear,
                           Translate)
from .compose import Compose
from .formatting import (Collect, DefaultFormatBundle, ImageToTensor,
                         ToDataContainer, ToTensor, Transpose, to_tensor,
                         DefaultFormatBundle_sp)  # add by hui
from .instaboost import InstaBoost
from .loading import (FilterAnnotations, LoadAnnotations, LoadImageFromFile,
                      LoadImageFromWebcam, LoadMultiChannelImageFromFiles,
                      LoadPanopticAnnotations, LoadProposals)
from .test_time_aug import MultiScaleFlipAug
from .transforms import (Albu, CopyPaste, CutOut, Expand, MinIoURandomCrop,
                         MixUp, Mosaic, Normalize, Pad, PhotoMetricDistortion,
                         RandomAffine, RandomCenterCropPad, RandomCrop,
                         RandomFlip, RandomShift, Resize, SegRescale,
                         YOLOXHSVRandomAug)
from .transforms import Resize_sp  # add by hui
from .rtest_time_aug import CroppedTilesFlipAug  # add by hui
from .loading import LoadAnnotations_sp  # add by hui
from .loading import LoadImageFromFile_sp  # add by hui


__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'DefaultFormatBundle', 'LoadAnnotations',
    'LoadImageFromFile', 'LoadImageFromWebcam', 'LoadPanopticAnnotations',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'FilterAnnotations',
    'MultiScaleFlipAug', 'Resize', 'RandomFlip', 'Pad', 'RandomCrop',
    'Normalize', 'SegRescale', 'MinIoURandomCrop', 'Expand',
    'PhotoMetricDistortion', 'Albu', 'InstaBoost', 'RandomCenterCropPad',
    'AutoAugment', 'CutOut', 'Shear', 'Rotate', 'ColorTransform',
    'EqualizeTransform', 'BrightnessTransform', 'ContrastTransform',
    'Translate', 'RandomShift', 'Mosaic', 'MixUp', 'RandomAffine',
    'YOLOXHSVRandomAug', 'CopyPaste',
    'Resize_sp',
    'LoadAnnotations_sp',
    'LoadImageFromFile_sp',
    'DefaultFormatBundle_sp',
    'CroppedTilesFlipAug'
]
