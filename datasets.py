import hashlib
from math import ceil
from operator import add

import attr
import numpy as np
from PIL import Image, ImageFile
from trains import Task
from torch.utils.data import Dataset, IterableDataset

from transforms import ToImgaug, ImgaugToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


@attr.s()
class BaseDataset(object):
    dataview = attr.ib(type=DataView)
    transforms_func = attr.ib(default=None)
    use_mask = attr.ib(default=False)
    train = attr.ib(default=True)
    annotate = attr.ib(default=False)

    def __attrs_post_init__(self):
        self.transforms = None
        self.versions_stat = SupportVector(report_interval=500, title='Versions support vector')
        if not self.annotate:
            self.labels_stat = SupportVector(report_interval=500, title='Labels support vector')

    def get_data_from_frame(self, frame):
        frame_annotations = frame.get_annotations()
        path = frame.get_local_source()
        if path is None:
            return None

        img = safe_pil_imread(path)
        if img is None:
            return None

        if self.annotate:
            frame.width = img.width
            frame.height = img.height
            img = self.transforms(image=np.array(img.convert('RGB')))
            return img, frame

        # ignore negative labels
        frame_annotations = [ann for ann in frame_annotations if ann.label_enum >= 0]

        # From boxes [x, y, w, h] to [x1, y1, x2, y2]
        new_target = {"image_id": np.array(string_to_8digits_hash(frame.id), dtype=np.int64),
                      "area": np.array([np.prod(obj.get_bounding_box()[2:]) for obj in frame_annotations], dtype=np.float32),
                      "iscrowd": np.array(['crowd' in obj.labels for obj in frame_annotations], dtype=np.int64),
                      "boxes": np.array([list(obj.get_bounding_box()[:2]) +
                                                list(map(add, obj.get_bounding_box()[:2], obj.get_bounding_box()[2:]))
                                                for obj in frame_annotations], dtype=np.float32),
                      "labels": np.array([obj.label_enum for obj in frame_annotations], dtype=np.int64)}

        # if self.use_mask: TODO: Add support for masks
        #     mask = []
        #     for i in range(len(frame_annotations)):
        #         mask_path = frame.get_local_mask_source()
        #         mask.append(coco.annToMask(target[i]))
        #     if len(mask) > 1:
        #         mask = np.stack(tuple(mask), axis=0)
        #     new_target["masks"] = torch.as_tensor(mask, dtype=torch.uint8)

        if self.transforms is not None:
            img, new_target = ToImgaug(img.convert('RGB'), new_target)
            img, new_target['boxes'] = self.transforms(image=img,
                                                       bounding_boxes=new_target['boxes'])
            img, new_target = ImgaugToTensor(img, new_target)

        self.versions_stat.update(frame._api_object.dataset.version)
        for obj in frame_annotations:
            self.labels_stat.update('_'.join(obj.labels))

        return img, new_target


class AllegroDataset(BaseDataset, Dataset):
    def __init__(self, dataview, transforms_func=None, use_mask=True, train=True, annotate=False):
        super(AllegroDataset, self).__init__(dataview, transforms_func=transforms_func, use_mask=use_mask,
                                             train=train, annotate=annotate)
        self.dataset = self.dataview.to_list(allow_repetition=train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frame = self.dataset[index]
        if (self.transforms is None) and (self.transforms_func is not None):
            self.transforms = self.transforms_func(train=self.train)
            import cv2
            print('cv2 default num threads = {}'.format(cv2.getNumThreads()))
            cv2.setNumThreads(0)
            print('cv2 new num threads = {}'.format(cv2.getNumThreads()))
        return self.get_data_from_frame(frame)


class AllegroIterableDataset(BaseDataset, IterableDataset):
    def __init__(self, dataview, transforms=None, use_mask=True, train=True):
        super(AllegroIterableDataset, self).__init__(dataview, transforms=transforms, use_mask=use_mask, train=train)
        self.dataset = self.dataview.get_iterator()

    def __len__(self):
        return get_dataview_len(self.dataview)

    def __iter__(self):
        return iter(self.AllegroDataview(self.dataset, self.get_data_from_frame))

    class AllegroDataview():
        def __init__(self, dataset, get_data_func):
            self.dataset = dataset
            self.get_data_func = get_data_func

        def __iter__(self):
            return self

        def __next__(self):
            frame = next(self.dataset)
            return self.get_data_func(frame)


def string_to_8digits_hash(string):
    return int(hashlib.sha1(string.encode()).hexdigest(), 16) % (10 ** 8)


def get_dataview_len(dataview):
    queries = dataview.get_queries()
    _, unique_count = dataview.get_count()
    # normalize the weights / ratios
    weights = [float(q.weight) for q in queries]
    # skip rules with zero results, there weight is irrelevant
    sum_weights = sum([w for c, w in zip(unique_count, weights) if c > 0])
    weights = [w / sum_weights for w in weights]
    max_count = max(unique_count)
    larges_rule_idx = [i for i, c in enumerate(unique_count) if c == max_count][0]
    override_iterator_len = int(ceil(unique_count[larges_rule_idx] / weights[larges_rule_idx]))
    return override_iterator_len


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
