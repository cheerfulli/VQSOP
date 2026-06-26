import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import tempfile
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
import yaml
from projects.mmdet3d_plugin.models.utils.transformation_utils import cal_dist, x1_to_x2
from collections import OrderedDict

@DATASETS.register_module()
class OPV2V(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, 
                 max_connect_car=0, connect_range= 50, occ_root=None, pose_noise=None, *args, **kwargs):
        self.occ_size = occ_size
        self.occ_root = occ_root if occ_root is not None else self.data_root
        # Initialize attributes BEFORE calling super().__init__() 
        # because parent's __init__ will call load_annotations which needs these
        self.len_record = []
        self.overlap_test = overlap_test
        self.max_connect_car = max_connect_car
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self.pose_noise = pose_noise
        self.connect_range = connect_range
        
        super().__init__(*args, **kwargs)
        
        self._set_group_flag()
        
    def load_annotations(self, ann_file):
        data_infos = []
        self.train_data_root = os.path.join(self.data_root, self.ann_file)
        self.scenes = []
        self.vehicle_infos = {}
        data_infos = []
        self.frames = []
        scene_num = 0

        root_dir=self.train_data_root
        scenario_folders = sorted([x for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x)) and x!="2021_09_09_13_20_58"])


        for scene in scenario_folders:
            scene_num +=1
            self.scenes.append(scene)
            self.vehicle_infos[scene] = OrderedDict()
            self.vehicle_infos[scene]["vehicles"] = []
            self.vehicle_infos[scene]["frames"] = []
            scenario_folder = os.path.join(root_dir, scene)
            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])  
            assert len(cav_list) > 0
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]
            for (j, cav_id) in enumerate(cav_list):
                vehicle=cav_id
                cav_path = os.path.join(scenario_folder, cav_id)
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)
              
                self.vehicle_infos[scene]["vehicles"].append(vehicle)

                for frame in timestamps:
                    self.vehicle_infos[scene]["frames"].append(frame)
                    if j==0:
                        data_infos.append((scene, vehicle, frame))
                
                if j==0:
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))

        return data_infos

    def extract_timestamps(self, yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps
        
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        return example
    def add_pose_noise(self, pose, pos_mean=0.3, pos_std=0.02, yaw_std_deg=2.0):
        x, y, z, roll, yaw, pitch = pose[:]
        ## location noise
        dx = np.random.normal(loc=pos_mean, scale=pos_std)
        dy = np.random.normal(loc=pos_mean, scale=pos_std)
        dz = np.random.normal(loc=0.0, scale=0.01)
        ## rotation noise
        dyaw = 0
        if yaw_std_deg is not None:
            dyaw_deg = np.random.normal(loc=0.0, scale=yaw_std_deg)
            dyaw = np.deg2rad(dyaw_deg)
        return (x + dx, y + dy, z + dz, roll, yaw + dyaw,  pitch)
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
                
        """


        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if index < ele:
                scenario_index = i
                break
        timestamp_index = index if scenario_index == 0 else \
            index - self.len_record[scenario_index - 1]

        occ_dir = os.path.join(self.occ_root, self.ann_file)
        occ_dir=os.path.join(occ_dir,str(scenario_index))
        occ_dir=os.path.join(occ_dir,str(timestamp_index))
        occ_dir = os.path.join(occ_dir,"co_processed_label.npy")

        

        # print(index)
        scene, vehicle, frame_num = self.data_infos[index]

        neighbors = [x for x in self.vehicle_infos[scene]["vehicles"] if x != vehicle]
        frame = os.path.join(self.train_data_root, scene, vehicle, frame_num)
        data = {}
        data["occ_path"] = occ_dir
        data["occ_size"] = self.occ_size
        data["pc_range"] = self.pc_range
        data["img_filename"] = []
        data["lidar2img"] = []
        data["lidar2cams"] = []
        data["cam_intrinsic"] = []
        data["pose"] = []
        data["trans2ego"] = []
        data["vehicle_id"] = []

        ego_info = self.get_vehicle_data(scene, vehicle, frame_num)
        data["img_filename"].extend(ego_info["imgs"])
        data["cam_intrinsic"].extend(ego_info["intrins"])
        data["lidar2img"].extend(ego_info["lidar2img"])
        data["lidar2cams"].extend(ego_info["lidar2cams"])
        data["vehicle_id"].append(ego_info["vehicle_id"])
        data["pose"].append(ego_info["pose"])
        data["trans2ego"].append(np.asarray(x1_to_x2(ego_info["pose"], ego_info["pose"])))
        neighbor_num = 0
        near_neighbor = []
        for neighbor in neighbors:
            neighbor_info = self.get_vehicle_data(scene, neighbor, frame_num)

            if cal_dist(neighbor_info["pose"], ego_info["pose"]) > self.connect_range:
                continue

            near_neighbor.append(neighbor_info)

        near_neighbor = sorted(near_neighbor, key=lambda neighbor: cal_dist(neighbor["pose"], ego_info["pose"]))
        for neighbor_info in near_neighbor:
            if neighbor_num >= self.max_connect_car:
                break
            data["img_filename"].extend(neighbor_info["imgs"])
            data["cam_intrinsic"].extend(neighbor_info["intrins"])
            data["lidar2cams"].extend(neighbor_info["lidar2cams"])
            data["lidar2img"].extend(neighbor_info["lidar2img"])
            data["vehicle_id"].append(neighbor_info["vehicle_id"])
            if self.pose_noise is not None:
                neighbor_info["pose"] = self.add_pose_noise(neighbor_info["pose"],
                                                            pos_mean=self.pose_noise["pos_mean"],
                                                            pos_std=self.pose_noise["pos_std"],
                                                            yaw_std_deg=self.pose_noise["yaw_std_deg"])
            data["pose"].append(neighbor_info["pose"])
            data["trans2ego"].append(np.asarray(x1_to_x2(neighbor_info["pose"], ego_info["pose"])))
            neighbor_num = neighbor_num +1


        data["connect_car_num"] = neighbor_num
        
        return data

    def get_vehicle_data(self, scene, vehicle, frame_num):
        frame = os.path.join(self.train_data_root, scene, vehicle, frame_num)
        
        meta_data = {}
        with open(frame+'.yaml','r') as f:
            meta_data = yaml.load(f, yaml.UnsafeLoader)
        pose = np.array(meta_data["lidar_pose"])
        imgs = []
        intrins = []
        lidar2cams = []
        lidar2imgs = []
        for i in range(4):
            imgs.append(frame + f"_camera{i}.png")

            intrin = np.array(meta_data[f"camera{i}"]["intrinsic"])
            viewpad = np.eye(4)
            viewpad[:intrin.shape[0], :intrin.shape[1]] = intrin
            ## carla coordinate to opencv coordinate
            axis_trans = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
            lidar2img = np.array(viewpad @ axis_trans @ np.array(meta_data[f"camera{i}"]["extrinsic"]))

            intrins.append(viewpad)
            lidar2cams.append(np.array(meta_data[f"camera{i}"]["extrinsic"]))
            lidar2imgs.append(lidar2img)

        data = {
            "imgs": imgs,
            "vehicle_id": vehicle,
            "intrins": intrins,
            "lidar2img": lidar2imgs,
            "lidar2cams": lidar2cams,
            "pose": pose,
        }
        return data

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        
        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {}
            class_num = len(self.class_names)
            for i, name in enumerate(self.class_names):
                class_names[i] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp+0.00001
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])
        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict={'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict
