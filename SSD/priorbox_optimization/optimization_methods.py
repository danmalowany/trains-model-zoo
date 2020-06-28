import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from logging import warning


from .bbox_clustering import get_min_max_object_scales_from_feature_size_array, create_match_scales_logic, \
    threshold_clusters


def global_optimization(gt_bbox_df, pvars, fmap_sizes, options, target_image_size):
    def area_cost(centers, fmap_sizes):
        cost = []
        max_val = fmap_sizes['width'].max() * fmap_sizes['height'].max()
        for width, height in zip(centers['width'], centers['height']):
            cost.append(max_val / (width * height))
        return np.array(cost) / max(cost)

    if options.plot_results:
        from matplotlib import pyplot as plt
        plt.style.use(style='seaborn-talk')
        plt.xlabel(pvars[0])
        plt.ylabel(pvars[1])
        plt.title('Collected ROIs @ {im_w}x{im_h} [{var1} vs {var2}]'.format(
            im_w=target_image_size.w,
            im_h=target_image_size.h,
            var1=pvars[0],
            var2=pvars[1]))

    min_object_scale, max_object_scale = get_min_max_object_scales_from_feature_size_array(fmap_sizes)
    clusters = []
    all_groups = gt_bbox_df.copy()
    all_points = all_groups[pvars].values
    all_points = np.clip(all_points, a_min=0, a_max=max_object_scale)

    n_clusters = min(options.max_n_clusters * len(fmap_sizes) , len(all_groups))
    kmeans = KMeans(n_clusters=n_clusters, init='random', verbose=False)
    kmeans.fit(all_points)
    y_km = kmeans.predict(all_points)
    all_groups['major_cluster'] = y_km
    all_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=['width', 'height'])
    # all_centers['weight'] = all_groups['major_cluster'].value_counts(normalize=True).sort_index()
    all_centers_weights = all_groups.groupby('major_cluster').sum()['overlap_score']
    all_centers['weight'] = all_centers_weights / all_centers_weights.max()
    all_centers['cost'] = area_cost(all_centers[['width', 'height']], fmap_sizes)
    all_centers['score'] = all_centers['weight'] * ( 1 /all_centers['cost'])

    from SSD.box_coder import box_iou
    import torch
    def iou_matching(centers, pixel_fov):
        def wh_to_boxes(df):
            out = []
            for width, height in zip(df['width'], df['height']):
                out.append([0, 0, width, height])
            return np.array(out)

        iou_results = box_iou(torch.tensor(wh_to_boxes(centers)), torch.tensor(wh_to_boxes(pixel_fov)))
        mg = iou_results.argmax(dim=1)
        return mg

    mg = iou_matching(all_centers, fmap_sizes)
    all_centers['match_group'] = mg

    if options.plot_results:
        # plot data according to match group
        color = 'C1'
        plt.scatter(all_groups[pvars[0]].values,
                    all_groups[pvars[1]].values, s=25, c=color)

        # plot data according to cluster fit
        for i in reversed(range(n_clusters)):
            plt.scatter(all_groups[pvars[0]][y_km == i],
                        all_groups[pvars[1]][y_km == i], s=35, marker='s', c=None, alpha=0.2)
        k_clusters = np.around(kmeans.cluster_centers_).astype(np.float)
        # plot also the original clusters
        ptr = plt.scatter(*k_clusters.T, s=250, c='yellow', marker='X', edgecolors='black', alpha=0.4)

    new_clusters = threshold_clusters(all_centers[['width', 'height']].values, all_centers['weight'].values, thresh=0.95)
    new_clusters_df = pd.DataFrame(np.vstack(new_clusters), columns=['width', 'height'])
    new_mg = iou_matching(new_clusters_df, fmap_sizes)
    new_clusters_df['match_group'] = new_mg

    if options.plot_results:
        # plot also the truncated clusters
        show_clusters = np.vstack(new_clusters)
        ptr = plt.scatter(*show_clusters.T, s=350, c='cyan', marker='X', edgecolors='black')

    for mg in range(len(fmap_sizes)):
        out = new_clusters_df.loc[new_clusters_df['match_group'] == mg][['width', 'height']].values
        clusters.append((mg, out))

    if options.plot_results:
        plt.show()
        plt.close()

    return clusters


def kmeans_per_fmap(gt_bbox_df, pvars, fmap_sizes, options, target_image_size):
    if options.plot_results:
        from matplotlib import pyplot as plt
        plt.style.use(style='seaborn-talk')
        plt.xlabel(pvars[0])
        plt.ylabel(pvars[1])
        plt.title('Collected ROIs @ {im_w}x{im_h} [{var1} vs {var2}]'.format(
            im_w=target_image_size.w,
            im_h=target_image_size.h,
            var1=pvars[0],
            var2=pvars[1]))

    min_object_scale, max_object_scale = get_min_max_object_scales_from_feature_size_array(fmap_sizes)
    low_bound, optional_figure = create_match_scales_logic(fmap_sizes, gt_bbox_df, enable_graphics=options.plot_results)
    clusters = []

    for mg, match_group in enumerate(tqdm(low_bound,
                                          desc='using k-means for finding bbox clusters per resolution')):
        # this_group = gt_bbox_df[match_group].drop_duplicates()
        this_group = gt_bbox_df[match_group].copy()
        points = this_group[pvars].values
        points = np.clip(points, a_min=0, a_max=max_object_scale)
        # # create kmeans object
        n_clusters = min(options.max_n_clusters, len(this_group))
        if len(this_group) < options.max_n_clusters:
            warning('Not enough samples in match group for n_clusters = {task}, reducing to {this}'.format(
                task=options.max_n_clusters, this=n_clusters))
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, init='random', verbose=False)
            # # fitkmeans object to data
            kmeans.fit(points)
            # # print location of clusters learned by kmeans object
            y_km = kmeans.predict(points)
            this_group['major_cluster'] = y_km

            cluster_weights = this_group.groupby('major_cluster').sum()['overlap_score'].values

            print('direct clustering result:')
            for i, cluster in enumerate(kmeans.cluster_centers_):
                print(' prior_{} : (weight: {:4.2f}) : [{} , {}]'.format(
                    i, cluster_weights[i], *cluster.astype(np.int)))

            if options.plot_results:
                # plot data according to match group
                color = 'C' + str(mg % 5 + 1)
                plt.scatter(this_group[pvars[0]].values,
                            this_group[pvars[1]].values, s=25, c=color)

                # plot data according to cluster fit
                for i in reversed(range(n_clusters)):
                    plt.scatter(this_group[pvars[0]][y_km == i],
                                this_group[pvars[1]][y_km == i], s=35, marker='s', c=None, alpha=0.2)
                k_clusters = np.around(kmeans.cluster_centers_).astype(np.float)
                # plot also the original clusters
                ptr = plt.scatter(*k_clusters.T, s=250, c='yellow', marker='X', edgecolors='black', alpha=0.4)

            # merge centers
            new_clusters = threshold_clusters(kmeans.cluster_centers_, cluster_weights,
                                              thresh=options.cluster_threshold)
            #
            if options.plot_results:
                # plot also the truncated clusters
                show_clusters = np.vstack(new_clusters)
                ptr = plt.scatter(*show_clusters.T, s=350, c='cyan', marker='X', edgecolors='black')

            clusters.append((mg, new_clusters))

        else:
            warning("no cluster for match group!")
            clusters.append((mg, []))

    if options.plot_results:
        plt.show()
        plt.close()

    return clusters
