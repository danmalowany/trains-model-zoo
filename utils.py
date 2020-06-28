import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist

from torchvision import transforms

from allegroai import Task


MEAN_IMAGE = [0.56927896, 0.5088081, 0.48382497]
STD_IMAGE = [0.27686155, 0.27230453, 0.2761136]


class AllegroDataset(Dataset):
    def __init__(self, dataview, transforms=None, train=True, annotate=False):
        super(AllegroDataset, self).__init__()
        self.dataview = dataview
        self.dataset = self.dataview.to_list(allow_repetition=train)
        self.transforms = transforms
        self.annotate = annotate
        self.versions_stat = SupportVector(report_interval=500, title='Versions support vector')
        if not self.annotate:
            self.labels_stat = SupportVector(report_interval=500, title='Labels support vector')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frame = self.dataset[index]
        frame_annotation = frame.get_annotations()
        path = frame.get_local_source()
        if path is None:
            return None

        img = safe_pil_imread(path)
        if img is None:
            return None

        image_tensor = self.transforms(img.convert('RGB'))
        if self.annotate:
            return image_tensor, frame

        cls = frame_annotation[0].label_enum

        self.versions_stat.update(frame._api_object.dataset.version)
        self.labels_stat.update('_'.join(frame.annotations[0].labels))

        return image_tensor, cls


class SupportVector(object):
    def __init__(self, report_interval=100, title='Support vector'):
        self.frames_data = []
        self.report_interval = report_interval
        self.title = title
        self.logger = Task.current_task().get_logger()
        self._history = (None, None)

    def add_history(self, new_data):
        history_values, history_counts = self._history
        (values, counts) = (x.tolist() for x in np.unique(new_data, return_counts=True))
        if history_values is None:
            return values, counts

        all_values, all_counts = ([], [])
        for val, count in zip(values, counts):
            if val not in history_values:
                all_values.append(val)
                all_counts.append(count)
            else:
                history_index = history_values.index(val)
                all_values.append(val)
                all_counts.append(history_counts[history_index] + count)
        return all_values, all_counts

    def update(self, new_data):
        self.frames_data.append(new_data)
        if len(self.frames_data) % self.report_interval == 0:
            (values, counts) = self.add_history(self.frames_data)
            for val, count in zip(values, counts):
                self.logger.report_scalar(title=self.title, series=val, value=float(count)/sum(counts),
                                          iteration=Task.current_task().get_last_iteration())
            self._history = (values, counts)
            self.frames_data = []


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def safe_annotator_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return [default_collate([x[0] for x in batch]), [x[1] for x in batch]]


def safe_pil_imread(path):
    try:
        img = Image.open(path)
    except:
        print('Corrupted file: {}'.format(path))
        return None
    if np.prod(img.size) > 89478485:
        print('Image exceeds PIL size limit: {}'.format(path))
        return None
    return img


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def get_rank():
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def prepare_imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(MEAN_IMAGE)
    std = np.array(STD_IMAGE)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_transforms(train, image_size):
    brightness, contrast, saturation, hue, degrees, shear = get_transform_values(Task.current_task())
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                           saturation=saturation, hue=hue)],
                                   p=0.25),
            transforms.RandomApply([transforms.RandomAffine(degrees=degrees, shear=shear, scale=(0.8, 1.2))], p=0.25),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.25),
            transforms.Normalize(MEAN_IMAGE, STD_IMAGE),
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMAGE, STD_IMAGE)
        ]),
    }
    if train:
        return data_transforms['train']
    else:
        return data_transforms['val']


def get_transform_values(task):
    configuration_data = task.get_model_config_dict()
    configuration_data['brightness'] = configuration_data.get('brightness', 0.4)
    configuration_data['contrast'] = configuration_data.get('contrast', 0.4)
    configuration_data['saturation'] = configuration_data.get('saturation', 0.4)
    configuration_data['hue'] = configuration_data.get('hue', 0.2)
    configuration_data['degrees'] = configuration_data.get('degrees', 45)
    configuration_data['shear'] = configuration_data.get('shear', 45)
    for key, val in configuration_data.items():
        configuration_data[key] = val
    task.set_model_config(config_dict=configuration_data)
    return configuration_data['brightness'], configuration_data['contrast'], configuration_data['saturation'], \
           configuration_data['hue'], configuration_data['degrees'], configuration_data['shear']


def calculate_dataview_stat(dataview):
    it = dataview.get_iterator()
    mean = [0, 0, 0]
    meansq = [0, 0, 0]
    transform = transforms.ToTensor()
    for m, frame in enumerate(it):
        path = frame.get_local_source()
        img = safe_pil_imread(path)
        if img:
            tensor = transform(img)
            for n, channel in enumerate(tensor):
                mean[n] += channel.mean()
                meansq[n] += (channel**2).mean()

    mean_all = np.array(mean)/m
    std_all = np.sqrt(np.array(meansq)/m - (np.array(mean)/m)**2)
    return mean_all, std_all
