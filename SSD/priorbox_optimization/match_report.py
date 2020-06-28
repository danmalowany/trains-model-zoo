from collections import OrderedDict
from itertools import zip_longest
from logging import warning
from typing import Tuple

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import AlignTypeEnum, BoxAlignCalculator
from .bbox_clustering import index_to_prior_name_map, target_iou

DEBUG_COMPARISON_KEEP_DATA = False


def match_report_by_label(priors_guess, gt_bbox, match_iou, box_aligner, label_dict, unique_str):
    hashed = hash(priors_guess.to_csv())
    np_labels = gt_bbox.label.unique()
    if not np.any(np_labels > 0):
        raise RuntimeError('Expected at least one label that has positive id')
    new_mx_iou_w_prior, new_percent_matched_n = intersect_prior_with_gt_bbox_df(
        box_aligner=box_aligner,
        iou_min_threshold=match_iou,
        priors_df=priors_guess,
        gt_bbox_df=gt_bbox,
        column_name_map=index_to_prior_name_map('{}_iou_%s' % (str(match_iou)), priors_guess),
        precalc_score_df=None,
        precalc_score_name_map=None,
        debug_result_data=Path('iou_pass_opt_dataview-{}_priors-{}_iou-{}.pkl'.format(unique_str, hashed, match_iou))
    )
    new_mx_iou_w_prior = new_mx_iou_w_prior.set_index(gt_bbox.index)
    new_stats = {}
    label_name_map = {_id: _name for _name, _id in label_dict.items()}
    for label in np_labels:
        label_group = gt_bbox.label == label
        label_name = label_name_map[label]
        new_stats[label_name] = new_mx_iou_w_prior[label_group].apply(lambda t: np.count_nonzero(t > match_iou), axis=0)
    match_stats = pd.DataFrame(new_stats)
    return match_stats


def intersect_prior_with_gt_bbox_df(
        box_aligner: BoxAlignCalculator,
        iou_min_threshold: float,
        priors_df: pd.DataFrame,
        gt_bbox_df: pd.DataFrame,
        column_name_map: dict = None,
        precalc_score_df: pd.DataFrame = None,
        precalc_score_name_map: dict = None,
        override_pd_columns: list = None,
        debug_result_data: Path = None,
        no_match_fill_value=np.NaN,
) -> Tuple[pd.DataFrame, pd.Series]:

    def calc_percent_match_per_column(df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda t: np.sum(~np.isnan(t)) / float(len(t)) * 100, axis=0)

    if debug_result_data and debug_result_data.exists():
        print('using pre-calculated data in {}'.format(debug_result_data))
        result = pd.read_pickle(path=str(debug_result_data))
        matched = calc_percent_match_per_column(result)
        return result, matched

    only_priors = list(column_name_map.keys()) if column_name_map else priors_df.index
    if precalc_score_name_map is not None:
        if column_name_map is not None:
            only_priors = {k: 'selected' for k in column_name_map if k in precalc_score_name_map.keys()}
        else:
            try:
                only_priors = only_priors.intersection(list(precalc_score_name_map.keys()))
            except Exception as ex:
                raise IndexError('without supplying column_name_map, precalc_score assumed the same index and failed')

    result = OrderedDict()
    col_query = ['width', 'height'] if not override_pd_columns else override_pd_columns

    if precalc_score_df is not None:
        assert type(precalc_score_df) is pd.DataFrame, "sorry, no documentation yet"
        if precalc_score_name_map is None:
            warning('Precalculated score dataframe without column name mapping, using fallback - column_name_map')
            precalc_score_name_map = column_name_map

    def create_query(gt_bbox: pd.DataFrame, prior_name_in_df: str):
        precalc_col = []
        if precalc_score_df is not None:
            col_name = precalc_score_name_map[prior_name_in_df]
            precalc_col = precalc_score_df[col_name].values
        return zip_longest(gt_bbox[col_query].values, precalc_col)

    def prior_tqdm(priors: pd.DataFrame, action_msg: str = None):
        msg = action_msg if action_msg else 'Intersecting'
        select = priors.index.isin(only_priors)
        return tqdm(zip(priors[select].index, priors[select][col_query].values),
                    total=len(only_priors), desc=msg + ' with priors...')

    for prior_name, prior in prior_tqdm(priors_df):
        metric = []
        for bbox, precalc in create_query(gt_bbox_df, prior_name):
            if precalc and np.isnan(precalc):
                metric.append(no_match_fill_value)
                continue
            shifts = box_aligner(prior, bbox)
            score = np.max(target_iou(prior, bbox, shifts))
            score = score if score > iou_min_threshold else no_match_fill_value
            metric.append(score)
        column_name = column_name_map.get(prior_name) if column_name_map else prior_name
        result[column_name] = metric

    result = pd.DataFrame(result)
    matched = calc_percent_match_per_column(result)

    if DEBUG_COMPARISON_KEEP_DATA and debug_result_data:
        result.to_pickle(path=str(debug_result_data))

    return result, matched


def generate_match_report(gt_bbox: pd.DataFrame, prev_priors: pd.DataFrame, new_priors: pd.DataFrame,
                          label_dict=None, match_iou=None, unique_str='', use_random_box_align=False):

    use_random_box_align = use_random_box_align if match_iou and match_iou > 0 else False
    match_iou = match_iou if match_iou else 0.4
    label_dict = {str(k): k for k in gt_bbox.label.unique()} if label_dict is None else label_dict

    align_strat = AlignTypeEnum.BottomLeft if not use_random_box_align else AlignTypeEnum.MultiRandom
    box_aligner = BoxAlignCalculator(align_strategy=align_strat)

    if not prev_priors.empty:
        print("Intersecting input priors with ground truth:")
        prev_stats = match_report_by_label(prev_priors, gt_bbox, match_iou, box_aligner=box_aligner,
                                           label_dict=label_dict, unique_str=unique_str)
    else:
        prev_stats = None
    print("Intersecting new priors with ground truth:")
    new_stats = match_report_by_label(new_priors, gt_bbox, match_iou, box_aligner=box_aligner,
                                      label_dict=label_dict, unique_str=unique_str)

    if prev_stats is not None:
        print('\n')
        print("Original prior match report:")
        print('\n'+prev_stats.to_string())

    print('\n')
    print("Optimized prior match report:")
    print('\n'+new_stats.to_string())

    return prev_stats, new_stats
