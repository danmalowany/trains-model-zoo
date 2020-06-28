from logging import warning

import numpy as np
import pandas as pd
from collections import OrderedDict


def filter_outliers(feature_sizes, gt_bbox_df, inplace=True,
                    drop_too_big=False, drop_too_small=False):
    minimum_object_scale, maximum_object_scale = get_min_max_object_scales_from_feature_size_array(feature_sizes)
    s_0 = (gt_bbox_df['width'] < minimum_object_scale) & (gt_bbox_df['height'] < minimum_object_scale)
    s_inf = (gt_bbox_df['width'] > maximum_object_scale) | (gt_bbox_df['height'] > maximum_object_scale)
    excluded = gt_bbox_df['width'] < 0
    if any(s_0):
        warning("There are {s0} objects smaller than the minimum object scale".format(s0=sum(s_0)))
        if drop_too_small:
            excluded |= s_0
    if any(s_inf):
        warning("There are {sinf} objects with at least one spatial dimension"
                " larger than the maximum possible scale to detect".format(sinf=sum(s_inf)))
        if drop_too_big:
            excluded |= s_inf
        warning("removing objects from dataframe")

    if inplace:
        gt_bbox_df.drop(gt_bbox_df[excluded].index, inplace=True)
    else:
        gt_bbox_df = gt_bbox_df.drop(gt_bbox_df[excluded].index, inplace=False)
    return gt_bbox_df


def get_min_max_object_scales_from_feature_size_array(feature_sizes):
    feature_size_to_matcher = feature_sizes.min(axis=1)
    minimum_object_scale = feature_size_to_matcher.min()
    maximum_object_scale = feature_size_to_matcher.max()
    return int(minimum_object_scale), int(maximum_object_scale)


def create_match_scales_logic(feature_sizes, gt_bbox_df, max_aspect_ratio=5, enable_graphics=False):
    MAX_AR = max_aspect_ratio

    if enable_graphics:
        from matplotlib import pyplot as plt
        fig = plt.figure(frameon=False)
    else:
        fig = []

    feature_size_to_matcher = feature_sizes.min(axis=1)
    minimum_object_scale = feature_size_to_matcher.min()
    # TODO - missing resolution - should be 7 and looking at 6
    min_sizes = feature_size_to_matcher[:-1]
    max_sizes = feature_size_to_matcher[1:]
    # fuzzy boundaries mkI
    # TODO - calculate the shared boundary sizes
    # min_sizes = [ms - minimum_object_scale for ms in min_sizes]
    # max_sizes = [ms + minimum_object_scale for ms in max_sizes]
    min_sizes = [max(ms, minimum_object_scale) for ms in min_sizes]
    max_sizes = [max(ms, minimum_object_scale) for ms in max_sizes]
    # split last range
    max_size_last = max_sizes[-1]
    min_sizes.append(min_sizes[-1] + (max_sizes[-1] - min_sizes[-1]) // 2)
    max_sizes[-1] = min_sizes[-1]
    max_sizes.append(max_size_last)
    low_bound = []
    for l_bound, r_bound in zip(min_sizes, max_sizes):
        w_bound = gt_bbox_df['width'].between(l_bound, r_bound, inclusive=True) & \
                  gt_bbox_df['height'].between(r_bound / MAX_AR, r_bound, inclusive=True)
        h_bound = gt_bbox_df['height'].between(l_bound, r_bound, inclusive=True) & \
                  gt_bbox_df['width'].between(l_bound / MAX_AR, r_bound, inclusive=True)
        low_bound.append(w_bound | h_bound)

        if enable_graphics:
            plotme = gt_bbox_df[low_bound[-1]][['width', 'height']].values
            plt.scatter(*plotme.T)

    # extra debug - show outliers
    if enable_graphics > 1:
        matched = low_bound[0].copy()
        for bound in low_bound:
            matched |= bound
        unmatched = ~matched
        values = gt_bbox_df[unmatched][['width', 'height']].values
        plt.scatter(*values.T, marker='x')

    return low_bound, fig


def xywh_to_ttbb(xywh_box: np.ndarray):
    whalf = np.floor_divide(xywh_box[:, 2], 2)
    hhalf = np.floor_divide(xywh_box[:, 3], 2)
    return np.array([
        # tx =
        xywh_box[:, 0] - whalf,
        # ty =
        xywh_box[:, 1] - hhalf,
        # bx =
        xywh_box[:, 0] + whalf,
        # by =
        xywh_box[:, 1] + hhalf]).T.astype(np.int)


def ttbb_box_area(ttbb_box: np.ndarray):
    if len(ttbb_box.shape) < 2:
        raise ValueError('expected atleast 2d array here')
    return (ttbb_box[:, 2] - ttbb_box[:, 0]) * (ttbb_box[:, 3] - ttbb_box[:, 1])


def compute_intersection_with_single_box(single_box: np.ndarray, boxes: np.ndarray):
    txy = np.maximum(single_box[:, 0:2], boxes[:, 0:2])
    bxy = np.minimum(single_box[:, 2:4], boxes[:, 2:4])
    return np.hstack((txy, bxy))


def fast_iou_one_many(ttbb_box, ttbb_boxes):
    one_box_area = ttbb_box_area(ttbb_box)
    many_boxes_area = ttbb_box_area(ttbb_boxes)
    overlap_boxes = compute_intersection_with_single_box(ttbb_box, ttbb_boxes)
    overlap_area = ttbb_box_area(overlap_boxes)
    # broadcasting ;)
    return overlap_area / ((one_box_area + many_boxes_area).astype(np.float32) - overlap_area)


def one_to_many_iou(box: np.ndarray, many_box: np.ndarray, box_aligner=None):
    shifts = box_aligner(box, many_box)
    ttbb_box = xywh_to_ttbb(np.atleast_2d(np.hstack((np.zeros(2), box))))
    ttbb_other_boxes = xywh_to_ttbb(np.hstack((shifts, many_box)))
    return fast_iou_one_many(ttbb_box, ttbb_other_boxes)


def index_to_prior_name_map(column_name_pattern: str, prior_table: pd.DataFrame) -> OrderedDict:
    mapping = OrderedDict()
    for prior_idx in prior_table.index:
        column_name = column_name_pattern.format(prior_idx)
        mapping[prior_idx] = column_name
    return mapping


def get_box_pairwise_iou(boxes):
    # TODO - rewrite to be more maintainable
    def _get_boxes_area(box_array):
        return np.prod(np.maximum(np.diff(box_array[:, [0, 2, 1, 3]], axis=1)[:, [0, 2]], 0), axis=1)

    n_boxes = len(boxes)
    left_ = boxes.repeat(n_boxes, 0)  # repeat boxes on x-array
    right_ = boxes.reshape(-1, 1).repeat(n_boxes, 1).T.reshape(-1, 4)  # repeat boxes on y-array
    intersection_boxes = np.zeros_like(left_)
    for k in [2, 3]:
        intersection_boxes[:, k] = np.hstack((left_[:, k:k+1], right_[:, k:k+1])).min(1)
    for k in [0, 1]:
        intersection_boxes[:, k] = np.hstack((left_[:, k:k + 1], right_[:, k:k + 1])).max(1)
    inter_area = _get_boxes_area(intersection_boxes).reshape(n_boxes, n_boxes)
    union_area = (_get_boxes_area(left_) + _get_boxes_area(right_)).reshape(n_boxes, n_boxes) - inter_area
    iou = inter_area / union_area
    return iou


def aspect_ratio_from_df(df: pd.DataFrame, columns=('width', 'height')):
    assert len(columns) == 2
    return np.divide(df[columns[0]].values, df[columns[-1]].values)


def humanized_aspect_ratio_from_df(df: pd.DataFrame, columns=('width', 'height')):
    assert len(columns) == 2
    ar = np.divide(df[columns[0]].values, df[columns[-1]].values)
    ar_m1 = ar - 1
    ra = np.divide(df[columns[-1]].values, df[columns[0]].values)
    m_ra_m1 = -(ra - 1)
    return np.where(ar >= 1, ar_m1, m_ra_m1)


def area_from_df(df: pd.DataFrame, columns=('width', 'height')):
    return np.prod(df[list(columns)].values, axis=1)


def bbox_log_distance(boxes, axis=0):
    return np.exp(np.median(np.log(boxes), axis=axis))


def bbox_wh_relative_metric(box_many_box_tuple):
    box, many_box = box_many_box_tuple
    return np.sqrt(np.sum(np.power(np.log(many_box / box), 2), axis=1))


def target_iou(target_box_wh: np.ndarray, incoming_box_wh: np.ndarray, center_deltas_xy: np.ndarray) -> np.ndarray:
    # todo raise exceptions instead
    # assert target_box_wh.size == 2, "single target box supported"
    # assert incoming_box_wh.size == 2, "single incoming box"
    # assert center_deltas_xy.shape[-1] == 2

    center_deltas_xy = np.atleast_2d(center_deltas_xy)

    incoming_box_wh_replica = np.tile(incoming_box_wh, len(center_deltas_xy)).reshape(-1, 2)
    incoming_boxes = np.hstack((center_deltas_xy, incoming_box_wh_replica))
    boxes_with_shift_ttbb = xywh_to_ttbb(incoming_boxes)

    target_box_ttbb = xywh_to_ttbb(np.array([[0, 0, *target_box_wh]]))
    overlap_boxes = compute_intersection_with_single_box(target_box_ttbb, boxes_with_shift_ttbb).astype(np.float32)

    target_box_area = target_box_wh[0] * target_box_wh[1]
    incoming_box_area = incoming_box_wh[0] * incoming_box_wh[1]
    overlap_area = ttbb_box_area(overlap_boxes).astype(np.float32)
    # some broadcasting...
    iou = overlap_area / ((target_box_area + incoming_box_area).astype(np.float32) - overlap_area)

    return iou


def threshold_clusters(cluster_centers, cluster_weights, thresh=None, logger=None):
    def _pick_clusters_based_on_weights(c_weights, k=0.5):
        """
        We are going to pick all the clusters that amount to >K percent of weight
        :param cluster_centers: (num_clusters, 2) center coordinates
        :param cluster_weights: (num_clusters, 1) weights, e.g. sum of some score for all the boxes in the cluster
        :return: IDXs of selected clusters
        """
        idxs = np.argsort(c_weights)[::-1]
        if (c_weights > 0).sum() <= 1:
            return idxs[:6]
        below_threshold = (np.cumsum(c_weights[idxs]) / c_weights.sum()) <= k
        selected_clusters = idxs[below_threshold]
        return selected_clusters

    def merge_centers(base_centers, centers_to_merge, base_weights, merge_weights):
        merged_centers = base_centers.copy()
        for center, weight in zip(centers_to_merge, merge_weights):
            dist = np.linalg.norm(center - base_centers, axis=1)
            closeset_center_idx = np.argsort(dist)[0]
            merged_center = np.average([base_centers[closeset_center_idx], center],
                                       weights=[base_weights[closeset_center_idx], weight],
                                       axis=0)
            merged_centers[closeset_center_idx] = merged_center
        return merged_centers

    new_cluster_idx = _pick_clusters_based_on_weights(cluster_weights, k=thresh)
    new_clusters = np.around(cluster_centers[new_cluster_idx, :])
    removed_clusters_idx = list(set.difference(set(range(len(cluster_centers))) , set(new_cluster_idx)))
    removed_clusters = np.around(cluster_centers[removed_clusters_idx, :])
    final_clusters = merge_centers(new_clusters, removed_clusters,
                                   cluster_weights[new_cluster_idx], cluster_weights[removed_clusters_idx])

    if logger:
        for i, cluster in enumerate(final_clusters):
            print(' prior-{} : : [{} , {}]'.format(i, *cluster))
    return [cluster for cluster in final_clusters]