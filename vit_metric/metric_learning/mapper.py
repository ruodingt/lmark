import albumentations
import cv2
import numpy as np
import torch


class ClsDatasetMapper:
    def __init__(self, cfg, is_train):
        self.is_train = is_train
        self.image_size = cfg.INPUT.IMAGE_SIZE_MSQ
        self.transform = self.get_transforms(is_train)
        pass

    def __call__(self, dataset_dict):
        filepath = dataset_dict['fp']
        label = dataset_dict['label']

        # FIXME: RGB or BGR? Change this to PIL load for more precise control of format
        #  MOST D2 config use BGR format
        image = cv2.imread(filepath)  # [:,:,::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if not self.is_train:
            return {"image": torch.tensor(image), "label": torch.tensor(label)}
        else:
            return {"image": torch.tensor(image), "label": torch.tensor(label)}

    def get_transforms(self, is_train):
        image_size = self.image_size
        if is_train:
            transforms = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ImageCompression(quality_lower=99, quality_upper=100),
                albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0,
                                                p=0.7),
                albumentations.Resize(image_size, image_size),
                albumentations.Cutout(max_h_size=int(image_size * 0.3), max_w_size=int(image_size * 0.3), num_holes=1,
                                      p=0.5),
                albumentations.Normalize()  # default to imageNet params
            ])

        else:
            transforms = albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize()  # default to imageNet params
            ])

        return transforms
