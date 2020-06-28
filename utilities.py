import json
from collections import OrderedDict

import attr
import cv2
import numpy as np
from torchvision.transforms import functional as F
from trains.utilities.plotly_reporter import SeriesInfo

from torchvision_references import utils
from torchvision_references.coco_eval import CocoEvaluator

from datetime import datetime


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return utils.collate_fn(batch)


def draw_boxes(im, boxes, labels, color=(150, 0, 0)):
    for box, draw_label in zip(boxes, labels):
        draw_box = box.astype('int')
        im = cv2.rectangle(im, tuple(draw_box[:2]), tuple(draw_box[2:]), color, 2)
        im = cv2.putText(im, str(draw_label), (draw_box[0], max(0, draw_box[1]-5)),
                         cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
    return im


def draw_debug_images(images, targets, predictions=None, score_thr=0.4):
    debug_images = []
    for n, (image, target) in enumerate(zip(images, targets)):
        img = draw_boxes(np.array(F.to_pil_image(image.cpu())),
                         [box.cpu().numpy() for box in target['boxes']],
                         [label.item() for label in target['labels']])
        if predictions and predictions[target['image_id'].item()]:
            img = draw_boxes(img,
                             [box.cpu().numpy() for box, score in
                              zip(predictions[target['image_id'].item()]['boxes'],
                                  predictions[target['image_id'].item()]['scores']) if score >= score_thr],
                             [label.item() for label, score in
                              zip(predictions[target['image_id'].item()]['labels'],
                                  predictions[target['image_id'].item()]['scores']) if score >= score_thr],
                             color=(0, 150, 0))
        debug_images.append(img)
    return debug_images


def draw_mask(target):
    masks = [channel*label for channel, label in zip(target['masks'].cpu().numpy(), target['labels'].cpu().numpy())]
    masks_sum = sum(masks)
    masks_out = masks_sum + 25*(masks_sum > 0)
    return (masks_out*int(255/masks_out.max())).astype('uint8')


@attr.s(auto_attribs=True)
class CocoLikeAnnotations():
    def __attrs_post_init__(self):
        self.coco_like_json: dict = {'images': [], 'annotations': []}
        self._ann_id: int = 0

    def update_images(self, file_name, height, width, id):
        self.coco_like_json['images'].append({'file_name': file_name,
                                         'height': height, 'width': width,
                                         'id': id})

    def update_annotations(self, box, label_id, image_id, is_crowd=0):
        segmentation, bbox, area = self.extract_coco_info(box)
        self.coco_like_json['annotations'].append({'segmentation': segmentation, 'bbox': bbox, 'area': area,
                                              'category_id': int(label_id), 'id': self._ann_id, 'iscrowd': is_crowd,
                                              'image_id': image_id})
        self._ann_id += 1

    @staticmethod
    def extract_coco_info(box):
        segmentation = list(map(int, [box[0], box[1], box[0], box[3], box[2], box[3], box[2], box[1]]))
        bbox = list(map(int, np.append(box[:2], (box[2:] - box[:2]))))
        area = int(bbox[2] * bbox[3])
        return segmentation, bbox, area

    def dump_to_json(self, path_to_json='/tmp/inference_results/inference_results.json'):
        with open(path_to_json, "w") as write_file:
            json.dump(self.coco_like_json, write_file)


class COCOResults(object):
    # pycocotools results structure:
    # T = len(p.iouThrs) | R = len(p.recThrs) | K = len(p.catIds) | A = len(p.areaRng) | M = len(p.maxDets)
    # precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    # recall = -np.ones((T, K, A, M))
    # scores = -np.ones((T, R, K, A, M))
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR_1", "AR_10", "AR", "ARs", "ARm", "ARl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR_1", "AR_10", "AR", "ARs", "ARm", "ARl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, coco_evaluator: CocoEvaluator, labels_enum: dict=None):
        iou_types = coco_evaluator.iou_types
        allowed_types = ("bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)

        self.iou_thresh = 0.5
        self.labels_enum = {val: key for key, val in labels_enum.items()}
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self._results = results

        self.full_results = OrderedDict()
        for iou_type in iou_types:
            self.full_results[iou_type] = None

        for iou_type in iou_types:
            self._update(coco_evaluator.coco_eval[iou_type])

    def _update(self, coco_eval):
        if coco_eval is None:
            return

        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self._results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

        self.full_results[iou_type] = {'precision': coco_eval.eval['precision'],
                                       'recall': coco_eval.eval['recall'],
                                       'scores': coco_eval.eval['scores'],
                                       'Average Precision': {k:v for k, v in self._results[iou_type].items() if k.startswith("AP")},
                                       'Average Recall': {k:v for k, v in self._results[iou_type].items() if k.startswith("AR")}}

        self.recThrs = coco_eval.params.recThrs
        curr_recall = coco_eval.params.recThrs
        iou_thresholds = coco_eval.params.iouThrs
        prec_recall_series = []
        conf_fscore_series = []
        AP50 = {}
        iou_idx = np.where(iou_thresholds == self.iou_thresh)[0][0]
        for m in range(self.full_results[iou_type]['recall'].shape[1]):
            label_name = self.labels_enum[m+1] if self.labels_enum is not None else str(m+1)
            if (self.full_results[iou_type]['precision'][iou_idx, :, m, 0, -1] >= 0).all():
                curr_precision = self.full_results[iou_type]['precision'][iou_idx, :, m, 0, -1].reshape(-1)
                curr_conf = self.full_results[iou_type]['scores'][iou_idx, :, m, 0, -1].reshape(-1)
                prec_recall_graph = [(recall, precision) for precision, recall
                                     in zip(curr_precision, curr_recall)]
                conf_fscore_graph = [(conf, ((2 * recall * precision) /
                                             (np.finfo(float).eps + recall + precision)))
                                     for conf, precision, recall in zip(curr_conf, curr_precision, curr_recall)]
                prec_recall_series.append(SeriesInfo(
                    name=str(label_name),
                    data=prec_recall_graph,
                    labels=['Recall=%3.2f, Precision=%3.2f' % (x[0], x[1]) for x in prec_recall_graph]
                ))
                conf_fscore_series.append(SeriesInfo(
                    name=str(label_name),
                    data=conf_fscore_graph,
                    labels=['conf_thr=%3.2f' % x[0] for x in conf_fscore_graph]
                ))
                AP50_curr = self.full_results[iou_type]['precision'][iou_idx, :, m, 0, -1]
                AP50[label_name] = np.mean(AP50_curr[AP50_curr > -1])
        self.full_results[iou_type]['Precision-Recall Curve'] = prec_recall_series
        self.full_results[iou_type]['Conf-F1-score Curve'] = conf_fscore_series
        self.full_results[iou_type]['AP50'] = AP50

    def get_results(self):
        return self.full_results
