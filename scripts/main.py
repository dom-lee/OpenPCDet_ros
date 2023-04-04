"""
Created on Apr 2, 2023

@author: Dongmyeong Lee (domlee[at]umich.edu)

"""

import sys
import os
import glob
import time

from easydict import EasyDict
from pathlib import Path

import numpy as np

# ROS
import rospy
import ros_numpy

# ROS Message
from geometry_msgs.msg import Vector3, Point, Quaternion
from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from tf.transformations import quaternion_from_euler

# OpenPCDet
sys.path.append(os.path.join(os.path.dirname(__file__), 'OpenPCDet'))
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
        return result

    return wrapper


def load_params():
    params = {
        "pointcloud_topic": "/velodyne_points",
        "bbox_topic": "/detect_3dbox",
        "bbox_frame_id": "map",
        "threshold": 0.5,
        "cfg_file": None,
        "ckpt_file": None
    }

    for param in params:
        try:
            value = rospy.get_param(param)
            params[param] = value
        except KeyError:
            rospy.logwarn(f"Parameter '{param}' not found. Using a default value.")

    return EasyDict(params)


# Copied from OpenPCDet/tools/demo.py
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class OpenPCDet:
    def __init__(self, cfg_file, ckpt_file, params):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg_file = os.path.join(self.script_dir, cfg_file)
        self.ckpt_file = os.path.join(self.script_dir, ckpt_file)

        self.model = None
        self.logger = None
        self.threshold = params.threshold

        self.dataset = None

        self.pointcloud = np.empty((0, 4)) # 'x', 'y', 'z', 'i'
        self.header = None
        
        self.publisher = None

        self.setup()
    
    def setup(self):
        cfg_from_yaml_file(self.cfg_file, cfg)
        self.logger = common_utils.create_logger()
        
        self.dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(self.script_dir), logger=self.logger
        )

        self.model = build_network(model_cfg=cfg.MODEL,
                                   num_class=len(cfg.CLASS_NAMES),
                                   dataset=self.dataset)

        self.model.load_params_from_file(filename=self.ckpt_file,
                                         logger=self.logger)
        self.model.cuda()
        self.model.eval()
    
    @timer_decorator
    def run(self):
        input_dict = {
            'points': self.pointcloud,
            'frame_id': 0,
        }

        data_dict = self.dataset.prepare_data(data_dict=input_dict)
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = self.model.forward(data_dict)

        pred_boxes  = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()

        return pred_boxes, pred_scores, pred_labels


    def pointcloud_callback(self, msg):
        msg_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        self.pointcloud = msg_array.view(dtype=np.float32).reshape(msg_array.shape + (-1,))
        self.header = msg.header

        pred_boxes, pred_scores, pred_labels = self.run()

        self.publish_bbox_array(pred_boxes, pred_scores, pred_labels)


    def publish_bbox_array(self, pred_boxes, pred_scores, pred_labels):
        bbox_array = BoundingBoxArray()
        bbox_array.header = self.header

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score < self.threshold:
                continue

            bbox = BoundingBox()
            bbox.header = bbox_array.header

            bbox.pose.position = Point(x=box[0], y=box[1], z=box[2])
            bbox.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, box[6]))
            bbox.dimensions = Vector3(x=box[3], y=box[4], z=box[5])

            bbox.value = score
            bbox.label = label

            bbox_array.boxes.append(bbox)

        self.publisher.publish(bbox_array)


def main():
    rospy.init_node('multi-object-tracker_node', anonymous=True)

    # Load Parameters
    params = load_params();

    # Set Publisher
    bbox_pub = rospy.Publisher(params.bbox_topic, BoundingBoxArray,
                               queue_size=10)

    # Construct OpenPCDet
    open_pc_det = OpenPCDet(params.cfg_file, params.ckpt_file, params)
    open_pc_det.publisher = bbox_pub

    # Set Subscriber
    pointcloud_sub = rospy.Subscriber(params.pointcloud_topic, PointCloud2,
                                      open_pc_det.pointcloud_callback,
                                      queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    main()
