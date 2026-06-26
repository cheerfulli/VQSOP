#import open3d as o3d
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os
from projects.configs.label_config.opv2v_label_mapping import opv2v_label_mapping_dict


@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic

    
    def __call__(self, results):
        occ_size = results["occ_size"]
        voxels = np.load(results['occ_path'])
        for key in opv2v_label_mapping_dict.keys():
            voxels[voxels == key] = opv2v_label_mapping_dict[key]
        results['gt_occ'] = voxels
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

