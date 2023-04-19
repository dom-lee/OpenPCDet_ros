# ROS Wrapper for OpenPCDet: Real-time 3D Object Detection with Point Cloud

This repository contains a ROS (Robot Operating System) wrapper for [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), enabling real-time 3D object detection using point cloud data. `OpenPCDet` is an open-source toolbox for 3D object detection from point cloud data.

With this ROS wrapper, you can seamlessly integrate OpenPCDet into your robotics applications to perform real-time 3D object detection using point cloud data from LiDAR or depth cameras.

## Features

- Real-time 3D object detection with point cloud data
- Easy integration with ROS-compatible devices
- Compatible with common ROS point cloud message types (e.g., `sensor_msgs/PointCloud2`)
- Configurable detection parameters for adapting to different environments and use cases

## Dependencies

Please provide the following information to help us list the dependencies:

- ROS Noetic
- CUDA version (11.7)
- PyTorch version (1.13.1)

## Installation

1. Clone this repository into your ROS workspace's `src` directory:

```bash
cd ~/your_ros_workspace/src
git clone https://github.com/dom-lee/OpenPCDet_ros
```

2. Install the required dependencies:

```bash
sudo apt-get install ros-<your_ros_distro>-jsk-rviz-plugins
sudo apt-get install ros-<your_ros_distro>-vision-msgs
```

3. Build the package:

```bash
cd ~/your_ros_workspace
catkin_make
```

4. Source the workspace:

```bash
source devel/setup.bash
```

## Usage

1. Launch the ROS wrapper for OpenPCDet:

```bash
roslaunch multiple-object-tracking kitti-raw.launch
```

2. Publish your point cloud data to the `/point_raw` topic using the `sensor_msgs/PointCloud2` message type.

3. Subscribe to the `/detect_3dbox` topic to receive detected 3D objects.

## Configuration

You can configure the ROS wrapper for OpenPCDet by modifying the parameters in the `config/kitti-raw.yaml` file. These parameters include:

- `pointcloud_topic`: Input point cloud topic (default: `/points_raw`)
- `bbox_topic`: Output detected objects topic (default: `/detect_3dbox`)
- `threshold`: Minimum score for an object to be considered detected (default: 0.1)
- `cfg_file`: Path to the OpenPCDet model configuration file
- `ckpt_file`: Path to the OpenPCDet model weights file

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

## License

This ROS wrapper for OpenPCDet is released under the [MIT License](LICENSE). Please refer to the original [OpenPCDet repository](https://github.com/open-mmlab/OpenPCDet) for information about its licensing.
