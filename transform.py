from albumentations.core.composition import BboxParams
from albumentations.core.composition import Compose as compo
from albumentations.augmentations.transforms import *
from albumentations import *
import albumentations as A

# (
#     LongestMaxSize,
#     PadIfNeeded,
#     RandomSizedBBoxSafeCrop,
#     ShiftScaleRotate,
#     RandomRotate90,
#     HorizontalFlip,
#     RandomBrightnessContrast,
#     RandomGamma,
#     HueSaturationValue,
#     MotionBlur,
#     JpegCompression,
#     Normalize,
# )


def get_albumentations_transforms(mode ):
    """
    Composes albumentations transforms.
    Returns the full list of transforms when mode is "train".
    mode should be one of "train", "val".
    """
    # compose validation transforms
    if mode == "val":
        transforms = compo(
            [],
            bbox_params=BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["category_id"],
            ),
        )
    # compose train transforms
    # TODO: make transformation parameters configurable from yml
    elif mode == "train":
        transforms = compo(
            [
        # A.Normalize(),
        # A.Blur(p=0.5),
        # A.ColorJitter(p=0.5),
        # A.Downscale(p=0.3),
        # A.Superpixels(p=0.3),
        A.RandomContrast(p=0.5),
        A.ShiftScaleRotate(p=0.8),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Sharpen(p = 0.5),

        # A.RGBShift(p=0.5),
        # A.RandomRain(p=0.3),
        # A.RandomFog(p=0.3)
            ],
            bbox_params=BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["category_id"],
            ),
        )
    return transforms
