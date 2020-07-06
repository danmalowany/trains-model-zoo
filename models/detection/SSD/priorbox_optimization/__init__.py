import enum
import attr
import pandas as pd
import numpy as np
from collections import namedtuple

ImageSizeTuple = namedtuple('ImageSizeTuple', 'w h')


def range_validator(min_value, max_value):
    """
    A parameter validator that checks range constraint on a parameter.

    :param min_value: The minimum limit of the range, inclusive. None for no minimum limit.
    :param max_value: The maximum limit of the range, inclusive. None for no maximum limit.
    :return: A new range validator
    """
    def _range_validator(instance, attribute, value):
        if ((min_value is not None) and (value < min_value)) or \
          ((max_value is not None) and (value > max_value)):
            raise ValueError("{} must be in range [{}, {}]".format(attribute.name, min_value, max_value))

    return _range_validator


@attr.s
class PriorOptimizationInput():
    target_image_size = attr.ib()
    gt_bbox = attr.ib(factory=pd.DataFrame)
    in_priors = attr.ib(factory=pd.DataFrame)
    fmap_sizes = attr.ib(factory=pd.DataFrame)
    gt_artifact_name = attr.ib(default='gt_collection')
    fmap_sizes_artifact_name = attr.ib(default='feature_map_sizes')

    @target_image_size.default
    def _default_input_size_512(self):
        return ImageSizeTuple(w=512, h=512)


@attr.s
class PriorOptimizationOutput():
    target_image_size = attr.ib()
    out_priors = attr.ib(factory=pd.DataFrame)

    @target_image_size.default
    def _default_input_size_512(self):
        return ImageSizeTuple(w=512, h=512)


@attr.s
class PriorOptimizationOptions():
    # Max number of clusters to create from drawn samples, per match group
    max_n_clusters = attr.ib(type=int, default=6, validator=range_validator(1, 99))
    # Used to threshold cluster importance
    cluster_threshold = attr.ib(type=float, default=0.5, validator=range_validator(0, 1))
    # 'Limit the width/height of images to max this value in pixels'
    target_size_w = attr.ib(type=int, default=512, validator=range_validator(100, None))
    target_size_h = attr.ib(type=int, default=512, validator=range_validator(100, None))
    #
    plot_results = attr.ib(type=bool, default=False)
    gen_match_report = attr.ib(type=bool, default=True)
    match_report_overlap = attr.ib(type=float, default=0.4, validator=range_validator(0.01, 0.99))
    optimization_method = attr.ib(type=str, default='Kmeans_per_feature_map')


class AlignTypeEnum(enum.Enum):
    BottomLeft = 'bottom_left'
    Nop = 'no_op'
    Random = 'random'
    MultiRandom = 'multi_random'


class AlignTypeFunctions:
    N_MULTI_RANDOM = 8

    @staticmethod
    def nop(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        # return np.tile(np.zeros_like(box1), 10).reshape(-1, 2)
        return np.zeros_like(box1)

    @staticmethod
    def bottom_left(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        return np.atleast_2d(0.5 * (box2 - box1))

    @staticmethod
    def random(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        max_shift_w = np.floor_divide(box1[0] + box2[0], 2)
        max_shift_h = np.floor_divide(box1[1] + box2[1], 2)
        return np.array([[np.random.randint(low=-max_shift_w, high=max_shift_w),
                          np.random.randint(low=-max_shift_h, high=max_shift_h)]])

    @staticmethod
    def multirandom(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        if box1[0] > box2[0] * 2 and box1[1] > box2[1] * 2:
            return np.array([0, 0])
        max_shift_w = np.floor_divide(box1[0] + box2[0], 2)
        max_shift_h = np.floor_divide(box1[1] + box2[1], 2)
        return np.hstack((np.random.randint(low=-max_shift_w,
                                            high=max_shift_w,
                                            size=(AlignTypeFunctions.N_MULTI_RANDOM, 1)),
                          np.random.randint(low=-max_shift_h,
                                            high=max_shift_h,
                                            size=(AlignTypeFunctions.N_MULTI_RANDOM, 1))))


class BoxAlignCalculator(object):
    def __init__(self, align_strategy: AlignTypeEnum = AlignTypeEnum.Nop, vectorize: bool = False):
        self._align_func = self._get_align_function(align_strategy, vectorize)
        if not self._align_func:
            raise TypeError('align_strategy must be of type AlignTypeEnum')

    @staticmethod
    def __get_align_func(align_type):
        return {
            AlignTypeEnum.BottomLeft: AlignTypeFunctions.bottom_left,
            AlignTypeEnum.Nop: AlignTypeFunctions.nop,
            AlignTypeEnum.Random: AlignTypeFunctions.random,
            AlignTypeEnum.MultiRandom: AlignTypeFunctions.multirandom,
        }.get(align_type)

    @staticmethod
    def _get_align_function(align_type, vectorize=False):
        align_func = BoxAlignCalculator.__get_align_func(align_type)
        if align_func is None:
            return None
        if vectorize:
            Warning('Caveat Emptor: Support for vectorized align types is minimal and most of this is untested')
            return np.vectorize(align_func)
        else:
            return align_func

    def __call__(self, box1: np.ndarray, box2: np.ndarray):
        return self._align_func(box1, box2)
