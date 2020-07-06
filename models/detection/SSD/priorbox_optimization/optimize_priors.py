import numpy as np
import pandas
from trains import Task

from . import PriorOptimizationOptions, PriorOptimizationInput, PriorOptimizationOutput
from .bbox_clustering import filter_outliers, area_from_df, aspect_ratio_from_df
from .match_report import generate_match_report
from .optimization_methods import global_optimization, kmeans_per_fmap

DEBUG = False


def optimize_priors(input_data: PriorOptimizationInput,
                    options: PriorOptimizationOptions = None):
    task = Task.current_task()
    options = options if options else PriorOptimizationOptions()
    # setup graph style:
    if options.plot_results:
        from matplotlib import pyplot as plt
        ax = input_data.gt_bbox[['width', 'height']].plot.hist(bins=50, alpha=0.5)
        ax.set_title("Dataset's box statistics")
        plt.show()
        plt.close()

    # grab any task labels if present
    label_dict = task.get_labels_enumeration() or None

    gt_bbox_df = input_data.gt_bbox
    fmap_sizes = input_data.fmap_sizes
    target_image_size = input_data.target_image_size

    gt_bbox_df = filter_outliers(fmap_sizes, gt_bbox_df, drop_too_big=True, drop_too_small=True)

    # save data to Trains task
    task.upload_artifact(input_data.gt_artifact_name, gt_bbox_df)
    task.upload_artifact('priors_input', input_data.in_priors)

    clusters = []
    var1 = 'width'
    var2 = 'height'
    pvars = [var1, var2]

    optimization_method = options.optimization_method
    if optimization_method == "Kmeans_global":
        clusters = global_optimization(gt_bbox_df, pvars, fmap_sizes, options, target_image_size)
    elif optimization_method == "Kmeans_per_feature_map":
        clusters = kmeans_per_fmap(gt_bbox_df, pvars, fmap_sizes, options, target_image_size)


    all_clusters = [cluster_group for _, cluster_group in clusters if len(cluster_group)]
    if not len(all_clusters):
        raise ValueError("Seems like no clusters were found. Are you sure the ground truth data is OK?")
    all_clusters = np.vstack(all_clusters)
    match_groups = np.hstack([[mg] * len(cluster_group) for mg, cluster_group in clusters if len(cluster_group)])
    new_priors_guess_df = create_prior_df(all_clusters, match_groups)
    task.upload_artifact('priors_output', new_priors_guess_df)

    if options.gen_match_report:
        # if label_dict is None:
        #     error("cannot generate match report if labels enumeration is not supplied")
        # else:
        print("Generating match report (intersecting priors with all bboxes sampled earlier...)")
        iou_thresh = options.match_report_overlap
        before, after = generate_match_report(gt_bbox=gt_bbox_df, prev_priors=input_data.in_priors,
                                              new_priors=new_priors_guess_df,
                                              label_dict=label_dict, unique_str=input_data.gt_artifact_name,
                                              match_iou=iou_thresh)
        if before is not None:
            task.upload_artifact('match_report_before', before, metadata=dict(match_iou=iou_thresh))
        task.upload_artifact('match_report_after', after, metadata=dict(match_iou=iou_thresh))
    return PriorOptimizationOutput(target_image_size, new_priors_guess_df)


def create_prior_df(priors, match_group):
    priors_df = pandas.DataFrame(priors, columns=['width', 'height'])
    priors_df['match_group'] = match_group
    # add more data
    priors_df['area'] = area_from_df(priors_df)
    priors_df['aspect_ratio'] = aspect_ratio_from_df(priors_df)
    priors_df = priors_df[['match_group', 'area', 'aspect_ratio', 'width', 'height']]
    priors_df = priors_df.sort_values(['match_group', 'area', 'aspect_ratio']).reset_index(drop=True)
    priors_df['index'] = list('prior_{}'.format(i) for i in range(len(priors)))
    priors_df = priors_df.set_index('index')
    return priors_df
