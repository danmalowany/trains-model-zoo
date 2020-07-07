import os
from operator import add

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CocoDetection

from transforms import ToImgaug, ImgaugToTensor


class CocoMask(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, use_mask=True):
        super(CocoMask, self).__init__(root, annFile, transforms, target_transform, transform)
        self.transforms = transforms
        self.use_mask = use_mask

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # From boxes [x, y, w, h] to [x1, y1, x2, y2]
        new_target = {"image_id": np.array(img_id, dtype=np.int64),
                      "area": np.array([obj['area'] for obj in target], dtype=np.float32),
                      "iscrowd": np.array([obj['iscrowd'] for obj in target], dtype=np.int64),
                      "boxes": np.array([obj['bbox'][:2] + list(map(add, obj['bbox'][:2], obj['bbox'][2:]))
                                         for obj in target], dtype=np.float32),
                      "labels": np.array([obj['category_id'] for obj in target], dtype=np.int64)}

        if self.use_mask:
            mask = [coco.annToMask(ann) for ann in target]
            if len(mask) > 1:
                mask = np.stack(tuple(mask), axis=0)
            new_target["masks"] = torch.as_tensor(mask, dtype=torch.uint8)

        if self.transforms is not None:
            img, new_target = ToImgaug(img.convert('RGB'), new_target)
            img, new_target['boxes'] = self.transforms(image=img,
                                                       bounding_boxes=new_target['boxes'])
            img, new_target = ImgaugToTensor(img, new_target)

        return img, new_target