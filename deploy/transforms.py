import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

BORDER_CONSTANT = 0
BORDER_REFLECT = 2
PATCH_SIZE = 512


def pre_transforms(image_size=512):
    result = [
        albu.Resize(height=256, width=512, always_apply=True),
    ]

    return result


def post_transforms():
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_valid_transforms():
    return compose([
        pre_transforms(),
        post_transforms()
    ])