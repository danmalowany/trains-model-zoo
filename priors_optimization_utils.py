from itertools import product

import numpy as np
import pandas as pd
import torch
from trains import Task

from SSD.priorbox_optimization import PriorOptimizationInput, ImageSizeTuple
from SSD.priorbox_optimization.bbox_clustering import get_box_pairwise_iou


def collect_ground_truth_stats(ground_truth_loader):
    def just_meta_iter(loader):
        for gt in loader:
            yield gt[-1]

    gt = list(just_meta_iter(ground_truth_loader))
    gt_df = get_gt_df_from_gt(gt)

    return gt_df


def get_gt_df_from_gt(gt):
    # removing all "crowd" labels

    def process_meta_element(element):
        boxes = element['boxes']
        iscrowd = element['iscrowd']
        labels = element['labels']

        orig_boxes = [box for box, crowd in zip(boxes, iscrowd) if not crowd]
        orig_labels = [label for label, crowd in zip(labels, iscrowd) if not crowd]

        orig_boxes = np.around(orig_boxes)
        width = np.around(orig_boxes[:, 2] - orig_boxes[:, 0])
        height = np.around(orig_boxes[:, 3] - orig_boxes[:, 1])

        area = width * height
        good_boxes = np.where(area > 0)[0]
        if len(good_boxes) != len(orig_boxes):
            boxes = orig_boxes[good_boxes]
            labels = np.array(orig_labels)[good_boxes].tolist()
            height = height[good_boxes]
            width = width[good_boxes]
        else:
            boxes = orig_boxes
            labels = orig_labels

        pairwise_iou = get_box_pairwise_iou(boxes)
        score = np.around(pairwise_iou.sum(axis=0) - 1, decimals=2)

        return [(w, h, label, q) for w, h, label, q in zip(width, height, labels, score)]

    processed_gt = [process_meta_element(el) for elem in gt for el in elem if len(el['boxes']) > 0]
    all_gt = [elem for elements in processed_gt for elem in elements]
    column_names = ['width', 'height', 'label', 'overlap_score']

    return pd.DataFrame(all_gt, columns=column_names)


def get_optimization_input(ground_truth_df, fmap_sizes, input_priors, image_size):
    def fmap_to_pixel_fov(fmap_sizes):
        # fm = [np.array([fmap, fmap]) for fmap in fmap_sizes]
        # fm_np = np.vstack(fm)
        # fm_in_pixels = np.array(image_size) / fm_np
        fm_in_pixels = np.array(image_size) * \
                       np.array([3/fmap_sizes[-7], 3/fmap_sizes[-6], 3/(fmap_sizes[-5]+2), 3/(fmap_sizes[-4]+2),
                                 3/(fmap_sizes[-3]+2), 3/(fmap_sizes[-2]+2), 1])
        fm_in_pixels = [np.array([fmap, fmap]) for fmap in fm_in_pixels]
        fm_in_pixels = np.vstack(fm_in_pixels)
        return pd.DataFrame(fm_in_pixels, columns=['width', 'height'])

    task = Task.current_task()
    fmap = [np.array([fmap, fmap]) for fmap in fmap_sizes]
    task.upload_artifact('feature_maps_sizes', pd.DataFrame(np.vstack(fmap), columns=['width', 'height']))

    fmap_df = fmap_to_pixel_fov(fmap_sizes)
    task.upload_artifact('feature_maps_pixel_fov', fmap_df)

    in_priors_df = pd.DataFrame(input_priors.numpy(), columns=['match_group', 'width', 'height'])
    target_image_size = ImageSizeTuple(w=image_size, h=image_size)

    return PriorOptimizationInput(
        target_image_size=target_image_size,
        gt_bbox=ground_truth_df,
        fmap_sizes=fmap_df,
        in_priors=in_priors_df,
    )


def convert_optimization_result_to_priors(fm_sizes, steps, opt_result):
    priors_output = opt_result.out_priors
    by_resolution = list(priors_output.groupby('match_group'))
    num_anchors_per_resolution = [len(priors[-1]) for priors in by_resolution]
    if len(num_anchors_per_resolution) < len(fm_sizes):
        print('Some resolution were empty - setting default prior per empty resolution')
        curr_match_groups = opt_result.out_priors.match_group.to_list()
        curr_prior_number = len(curr_match_groups)
        empty_match_groups = list(set(range(len(fm_sizes))) - set(np.unique(curr_match_groups)))
        for empty_match_group in empty_match_groups:
            prior_size = opt_result.target_image_size.w / fm_sizes[empty_match_group]
            new_prior = pd.DataFrame(np.array([empty_match_group, prior_size**2, 1, prior_size, prior_size]).reshape(1, 5),
                                     columns=['match_group', 'area', 'aspect_ratio', 'width', 'height'])
            new_prior['index'] = 'prior_{}'.format(curr_prior_number)
            new_prior = new_prior.set_index('index')
            priors_output = priors_output.append(new_prior)
            curr_prior_number += 1
            by_resolution.append((empty_match_group, new_prior))
            num_anchors_per_resolution.append(1)
        Task.current_task().register_artifact('priors_output', priors_output.sort_values('match_group'))
        by_resolution = list(priors_output.groupby('match_group'))

    boxes = []
    priors = []
    for i, (fm_size, new_priors) in enumerate(zip(fm_sizes, by_resolution)):
        for h, w in product(range(fm_size), repeat=2):
            cx = (w + 0.5) * steps[i]
            cy = (h + 0.5) * steps[i]

            for prior in new_priors[-1].iterrows():
                w = prior[-1].width
                h = prior[-1].height
                boxes.append((cx, cy, w, h))
                priors.append((i, w, h))

    return torch.Tensor(boxes), torch.Tensor(np.unique(np.array(priors), axis=0)), num_anchors_per_resolution
