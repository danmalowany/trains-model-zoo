import random
import cv2
cv2.setNumThreads(0)

import imgaug as ia
import numpy as np
import torch
from PIL import Image
from trains import Task
from imgaug import augmenters as iaa
from torchvision.transforms import functional as F
from torchvision.transforms import transforms


def get_transform(train, image_size):
    transforms = [Resize(size=(image_size, image_size)),ToTensor()]
    if train:
        transforms= [Resize(size=(image_size, image_size)),
                     RandomGrayscale(p=0.05),
                     RandomApply([ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15)], p=0.25),
                     ToTensor(),
                     RandomApply([AddGaussianNoise(0., 0.1)], p=0.25),
                     RandomHorizontalFlip(),
                     ]
    return Compose(transforms)


def get_augmentations(train, image_size):
    augmentation = iaa.Sequential([
        iaa.Resize({"height": image_size, "width": image_size})
    ])
    if train:
        brightness, contrast, saturation, hue, rotate, shear = get_transform_values(Task.current_task())
        augmentation = iaa.Sequential([
            iaa.Resize({"height": image_size, "width": image_size}),
            iaa.Fliplr(0.5),  # horizontal flips
            # iaa.Crop(percent=(0, 0.1)),  # random crops
            # iaa.RemoveCBAsByOutOfImageFraction(0.6),
            iaa.Sometimes(0.25, iaa.OneOf([iaa.SaltAndPepper(p=(0.0, 0.04), per_channel=0.5),
                                           iaa.AdditiveGaussianNoise(scale=(0, 15), per_channel=0.5)])),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.75))),
            iaa.Sometimes(0.25, iaa.PerspectiveTransform(scale=(0.0, 0.05))),
            iaa.ChannelShuffle(0.1),
            iaa.LinearContrast((1-contrast, 1+contrast)),  # Strengthen or weaken the contrast in each image.
            iaa.Multiply((1-brightness, 1+brightness)),  # Make some images brighter and some darker.
            iaa.MultiplyHueAndSaturation(mul_hue=(1-hue, 1+hue), mul_saturation=(1-saturation, 1+saturation)),
            # Apply affine transformations to each image.
            iaa.Affine(
                scale={"x": (0.8, 1.15), "y": (0.8, 1.15)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-rotate, rotate),
                shear=(-shear, shear)),
            # iaa.RemoveCBAsByOutOfImageFraction(0.6),
        ])
    return augmentation


def get_transform_values(task):
    configuration_data = task.get_model_config_dict()
    configuration_data['brightness'] = configuration_data.get('brightness', 0.3)
    configuration_data['contrast'] = configuration_data.get('contrast', 0.3)
    configuration_data['saturation'] = configuration_data.get('saturation', 0.3)
    configuration_data['hue'] = configuration_data.get('hue', 0.3)
    configuration_data['rotate'] = configuration_data.get('rotate', 30)
    configuration_data['shear'] = configuration_data.get('shear', 15)
    for key, val in configuration_data.items():
        configuration_data[key] = val
    task.set_model_config(config_dict=configuration_data)
    return configuration_data['brightness'], configuration_data['contrast'], configuration_data['saturation'], \
           configuration_data['hue'], configuration_data['rotate'], configuration_data['shear']


def ToImgaug(image, target):
    image = np.array(image)
    target['boxes'] = ia.BoundingBoxesOnImage(
        [ia.BoundingBox(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]), label=label)
        if len(box)>0 else [] for box, label in zip(target['boxes'], target['labels'].tolist())], shape=image.shape)
    return image, target


def ImgaugToTensor(image, target):
    # image = F.to_tensor(F.to_pil_image(image))
    def safe_negative_frame(target):
        target['labels'] = torch.zeros((0, 1), dtype=torch.int64) # torch.as_tensor([], dtype=torch.int64)
        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32) # torch.as_tensor([])
        return target

    if len(target['boxes']) == 0:
        target = safe_negative_frame(target)
        return image, target
    else:
        target['labels'] = np.array([box.label for box in
                                            target['boxes'].remove_out_of_image().clip_out_of_image()], dtype=np.int64)
        target['boxes'] = np.array([[box.x1, box.y1, box.x2, box.y2] for box in
                                           target['boxes'].remove_out_of_image().clip_out_of_image()], dtype=np.float32)
        if len(target['boxes']) == 0:
            target = safe_negative_frame(target)
    return image, target


class RandomGeneral:
    def __call__(self,  img, target):
        NotImplementedError()


class RandomApply(RandomGeneral, transforms.RandomApply):
    def __call__(self, img, target):
        if self.p < random.random():
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomGrayscale(RandomGeneral, transforms.RandomGrayscale):
    def __call__(self, img, target):
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels), target
        return img, target


class ColorJitter(RandomGeneral, transforms.ColorJitter):
    def __call__(self, img, target):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), target


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, image, target):
        return (image + torch.randn(image.size()) * self.std + self.mean).clamp(0, 1), target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "boxes" in target and len(target["boxes"]) > 0:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class Resize(object):
    """Resize the input PIL image to given size.
    If boxes is not None, resize boxes accordingly.
    Args:
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
      random_interpolation: (bool) randomly choose a resize interpolation method.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    Example:
    >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
    >> img, boxes = resize(img, boxes, (500,600))  # resize image size to (500,600)
    >> img, _ = resize(img, None, (500,600))  # resize image only
    """
    def __init__(self, size, max_size=1000, random_interpolation=False):
        self.size = size
        self.max_size = max_size
        self.random_interpolation = random_interpolation

    def __call__(self, image, target):
        """Resize the input PIL image to given size.
        If boxes is not None, resize boxes accordingly.
        Args:
          image: (PIL.Image) image to be resized.
          target: (tensor) object boxes, sized [#obj,4].
        """
        w, h = image.size
        if isinstance(self.size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(self.size) / size_min
            if sw * size_max > self.max_size:
                sw = sh = float(self.max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = self.size
            sw = float(ow) / w
            sh = float(oh) / h

        method = random.choice([
            Image.BOX,
            Image.NEAREST,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
            Image.BILINEAR]) if self.random_interpolation else Image.BILINEAR
        image = image.resize((ow, oh), method)
        if target is not None and "masks" in target:
            resized_masks = torch.nn.functional.interpolate(
                input=target["masks"][None].float(),
                size=(512, 512),
                mode="nearest",
            )[0].type_as(target["masks"])
            target["masks"] = resized_masks
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            resized_boxes = target["boxes"] * torch.tensor([sw, sh, sw, sh])
            target["boxes"] = resized_boxes
        return image, target

#
# import cv2
# import numpy as np
# class RandomRotate(object):
#     """Randomly rotates an image
#
#     Bounding boxes which have an area of less than 25% in the remaining in the
#     transformed image is dropped. The resolution is maintained, and the remaining
#     area if any is filled by black color.
#
#     Parameters
#     ----------
#     angle: float or tuple(float)
#         if **float**, the image is rotated by a factor drawn
#         randomly from a range (-`angle`, `angle`). If **tuple**,
#         the `angle` is drawn randomly from values specified by the
#         tuple
#
#     Returns
#     -------
#
#     numpy.ndaaray
#         Rotated image in the numpy format of shape `HxWxC`
#
#     numpy.ndarray
#         Tranformed bounding box co-ordinates of the format `n x 4` where n is
#         number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
#
#     """
#
#     def __init__(self, angle=10):
#         self.angle = angle
#
#         if type(self.angle) == tuple:
#             assert len(self.angle) == 2, "Invalid range"
#
#         else:
#             self.angle = (-self.angle, self.angle)
#
#     def __call__(self, img, bboxes):
#
#         angle = random.uniform(*self.angle)
#
#         w, h = img.shape[1], img.shape[0]
#         cx, cy = w // 2, h // 2
#
#         img = rotate_im(img, angle)
#
#         corners = get_corners(bboxes)
#
#         corners = np.hstack((corners, bboxes[:, 4:]))
#
#         corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
#
#         new_bbox = get_enclosing_box(corners)
#
#         scale_factor_x = img.shape[1] / w
#
#         scale_factor_y = img.shape[0] / h
#
#         img = cv2.resize(img, (w, h))
#
#         new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
#
#         bboxes = new_bbox
#
#         bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)
#
#         return img, bboxes
#
#
# class RandomShear(object):
#     """Randomly shears an image in horizontal direction
#
#
#     Bounding boxes which have an area of less than 25% in the remaining in the
#     transformed image is dropped. The resolution is maintained, and the remaining
#     area if any is filled by black color.
#
#     Parameters
#     ----------
#     shear_factor: float or tuple(float)
#         if **float**, the image is sheared horizontally by a factor drawn
#         randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
#         the `shear_factor` is drawn randomly from values specified by the
#         tuple
#
#     Returns
#     -------
#
#     numpy.ndaaray
#         Sheared image in the numpy format of shape `HxWxC`
#
#     numpy.ndarray
#         Tranformed bounding box co-ordinates of the format `n x 4` where n is
#         number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
#
#     """
#
#     def __init__(self, shear_factor=0.2):
#         self.shear_factor = shear_factor
#
#         if type(self.shear_factor) == tuple:
#             assert len(self.shear_factor) == 2, "Invalid range for scaling factor"
#         else:
#             # self.shear_factor = (-self.shear_factor, self.shear_factor)
#             self.shear_factor = (0, self.shear_factor)
#
#         shear_factor = random.uniform(*self.shear_factor)
#
#     def __call__(self, img, bboxes):
#
#         shear_factor = random.uniform(*self.shear_factor)
#
#         w, h = img.shape[1], img.shape[0]
#
#         # if shear_factor < 0:
#         #     img, bboxes = HorizontalFlip()(img, bboxes)
#
#         M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
#
#         nW = img.shape[1] + abs(shear_factor * img.shape[0])
#
#         bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)
#
#         img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
#
#         # if shear_factor < 0:
#         #     img, bboxes = HorizontalFlip()(img, bboxes)
#
#         img = cv2.resize(img, (w, h))
#
#         scale_factor_x = nW / w
#
#         bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]
#
#         return img, bboxes
#
#
# class RandomScale(object):
#     """Randomly scales an image
#
#
#     Bounding boxes which have an area of less than 25% in the remaining in the
#     transformed image is dropped. The resolution is maintained, and the remaining
#     area if any is filled by black color.
#
#     Parameters
#     ----------
#     scale: float or tuple(float)
#         if **float**, the image is scaled by a factor drawn
#         randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
#         the `scale` is drawn randomly from values specified by the
#         tuple
#
#     Returns
#     -------
#
#     numpy.ndaaray
#         Scaled image in the numpy format of shape `HxWxC`
#
#     numpy.ndarray
#         Tranformed bounding box co-ordinates of the format `n x 4` where n is
#         number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
#
#     """
#
#     def __init__(self, scale=0.2, diff=False):
#         self.scale = scale
#
#         if type(self.scale) == tuple:
#             assert len(self.scale) == 2, "Invalid range"
#             assert self.scale[0] > -1, "Scale factor can't be less than -1"
#             assert self.scale[1] > -1, "Scale factor can't be less than -1"
#         else:
#             assert self.scale > 0, "Please input a positive float"
#             self.scale = (max(-1, -self.scale), self.scale)
#
#         self.diff = diff
#
#     def __call__(self, img, bboxes):
#
#         # Chose a random digit to scale by
#
#         img_shape = img.shape
#
#         if self.diff:
#             scale_x = random.uniform(*self.scale)
#             scale_y = random.uniform(*self.scale)
#         else:
#             scale_x = random.uniform(*self.scale)
#             scale_y = scale_x
#
#         resize_scale_x = 1 + scale_x
#         resize_scale_y = 1 + scale_y
#
#         img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)
#
#         bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
#
#         canvas = np.zeros(img_shape, dtype=np.uint8)
#
#         y_lim = int(min(resize_scale_y, 1) * img_shape[0])
#         x_lim = int(min(resize_scale_x, 1) * img_shape[1])
#
#         canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]
#
#         img = canvas
#         bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)
#
#         return img, bboxes
#
#
# def bbox_area(bbox):
#     return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
#
#
# def clip_box(bbox, clip_box, alpha):
#     """Clip the bounding boxes to the borders of an image
#
#     Parameters
#     ----------
#
#     bbox: numpy.ndarray
#         Numpy array containing bounding boxes of shape `N X 4` where N is the
#         number of bounding boxes and the bounding boxes are represented in the
#         format `x1 y1 x2 y2`
#
#     clip_box: numpy.ndarray
#         An array of shape (4,) specifying the diagonal co-ordinates of the image
#         The coordinates are represented in the format `x1 y1 x2 y2`
#
#     alpha: float
#         If the fraction of a bounding box left in the image after being clipped is
#         less than `alpha` the bounding box is dropped.
#
#     Returns
#     -------
#
#     numpy.ndarray
#         Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
#         number of bounding boxes left are being clipped and the bounding boxes are represented in the
#         format `x1 y1 x2 y2`
#
#     """
#     ar_ = (bbox_area(bbox))
#     x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
#     y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
#     x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
#     y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)
#
#     bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))
#
#     delta_area = ((ar_ - bbox_area(bbox)) / ar_)
#
#     mask = (delta_area < (1 - alpha)).astype(int)
#
#     bbox = bbox[mask == 1, :]
#
#     return bbox
#
#
# def rotate_im(image, angle):
#     """Rotate the image.
#
#     Rotate the image such that the rotated image is enclosed inside the tightest
#     rectangle. The area not occupied by the pixels of the original image is colored
#     black.
#
#     Parameters
#     ----------
#
#     image : numpy.ndarray
#         numpy image
#
#     angle : float
#         angle by which the image is to be rotated
#
#     Returns
#     -------
#
#     numpy.ndarray
#         Rotated Image
#
#     """
#     # grab the dimensions of the image and then determine the
#     # centre
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     # perform the actual rotation and return the image
#     image = cv2.warpAffine(image, M, (nW, nH))
#
#     #    image = cv2.resize(image, (w,h))
#     return image
#
#
# def get_corners(bboxes):
#     """Get corners of bounding boxes
#
#     Parameters
#     ----------
#
#     bboxes: numpy.ndarray
#         Numpy array containing bounding boxes of shape `N X 4` where N is the
#         number of bounding boxes and the bounding boxes are represented in the
#         format `x1 y1 x2 y2`
#
#     returns
#     -------
#
#     numpy.ndarray
#         Numpy array of shape `N x 8` containing N bounding boxes each described by their
#         corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
#
#     """
#     width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
#     height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)
#
#     x1 = bboxes[:, 0].reshape(-1, 1)
#     y1 = bboxes[:, 1].reshape(-1, 1)
#
#     x2 = x1 + width
#     y2 = y1
#
#     x3 = x1
#     y3 = y1 + height
#
#     x4 = bboxes[:, 2].reshape(-1, 1)
#     y4 = bboxes[:, 3].reshape(-1, 1)
#
#     corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
#
#     return corners
#
#
# def rotate_box(corners, angle, cx, cy, h, w):
#     """Rotate the bounding box.
#
#
#     Parameters
#     ----------
#
#     corners : numpy.ndarray
#         Numpy array of shape `N x 8` containing N bounding boxes each described by their
#         corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
#
#     angle : float
#         angle by which the image is to be rotated
#
#     cx : int
#         x coordinate of the center of image (about which the box will be rotated)
#
#     cy : int
#         y coordinate of the center of image (about which the box will be rotated)
#
#     h : int
#         height of the image
#
#     w : int
#         width of the image
#
#     Returns
#     -------
#
#     numpy.ndarray
#         Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
#         corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
#     """
#
#     corners = corners.reshape(-1, 2)
#     corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
#
#     M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
#
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cx
#     M[1, 2] += (nH / 2) - cy
#     # Prepare the vector to be transformed
#     calculated = np.dot(M, corners.T).T
#
#     calculated = calculated.reshape(-1, 8)
#
#     return calculated
#
#
# def get_enclosing_box(corners):
#     """Get an enclosing box for ratated corners of a bounding box
#
#     Parameters
#     ----------
#
#     corners : numpy.ndarray
#         Numpy array of shape `N x 8` containing N bounding boxes each described by their
#         corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
#
#     Returns
#     -------
#
#     numpy.ndarray
#         Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
#         number of bounding boxes and the bounding boxes are represented in the
#         format `x1 y1 x2 y2`
#
#     """
#     x_ = corners[:, [0, 2, 4, 6]]
#     y_ = corners[:, [1, 3, 5, 7]]
#
#     xmin = np.min(x_, 1).reshape(-1, 1)
#     ymin = np.min(y_, 1).reshape(-1, 1)
#     xmax = np.max(x_, 1).reshape(-1, 1)
#     ymax = np.max(y_, 1).reshape(-1, 1)
#
#     final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))
#
#     return final