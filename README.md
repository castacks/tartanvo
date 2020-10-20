# TartanVO: A Generalizable Learning-based VO

TartanVO is a learning-based visual odometry trained on [TartanAir](https://theairlab.org/tartanair-dataset) dataset. It generalizes to multiple datasets and real-world scenarios, and outperforms geometry-based methods in challenging scenes. 

TODO: add a paper link and author info

## Example
Introduction video: [Youtube](https://www.youtube.com/watch?v=NQ1UEh3thbU)

TODO: Add another video source

Our model is trained purely on simulation data, but it generalizes well to real-world data. For example, this is the testing result on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 10 trajectory:

![KITTI10](results/kitti_10_tartanvo_1914.png)

This is the testing result on [EuRoC V102](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):

![EUROC_V102](results/euroc_v102_tartanvo_1914.png)

## Requirments
* Python 2 / 3
* numpy 
* matplotlib
* scipy
* pytorch >= 1.3
* torchvision
* opencv-python
* cupy == 6.7.0
* visdom 
* [WorkFlow](https://github.com/huyaoyu/WorkFlow): branch amigo 

You can install the above dependencies manually, or follow the following steps:
```
$ pip install numpy matplotlib scipy torch==1.4.0 torchvision opencv-python cupy==6.7.0 visdom

```

## Test with pretrained model
### Download the pretrained model

```
$ mkdir models

$ wget https://cmu.box.com/shared/static/t1a5u4x6dxohl89104dyrsiz42mvq2sz.pkl -O models/tartanvo_1914.pkl

```

### Download the testing data
  
* Download KITTI-10 testing trajectory
```
$ mkdir data

$ wget https://cmu.box.com/shared/static/nw3bi7x5vng2xy296ndxt19uozpk64jq.zip -O data/KITTI_10.zip

$ unzip -q data/KITTI_10.zip -d data/KITTI_10/
```

* Download EuRoC-V102 testing trajectory
```
$ mkdir data

$ wget https://cmu.box.com/shared/static/1ctocidptdv1xev6pjxdj0b782mrstps.zip -O data/EuRoC_V102.zip

$ unzip -q data/EuRoC_V102.zip -d data/EuRoC_V102/
```

<!--    * Download TartanAir-MH006 testing trajectory
   ```

   ```
 -->

### Run the testing script
The `vo_trajectory_from_folder.py` script shows an example of running TartanVO on a sequence of images reading out from a folder. Because TartanVO outputs up-to-scale translation, it also reads a pose-file for adjusting the translation scale. If the pose-file is not provided, the default scale of 1.0 will be used. The results will be stored in the `results` folder. 

- Testing on KITTI
```
$ python vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl --kitti --test-dir data/KITTI_10/image_left --pose-file data/KITTI_10/pose_left.txt 
```
- Testing on EuRoC
```

$ python vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl --euroc --test-dir data/EuRoC_V102/image_left --pose-file data/EuRoC_V102/pose_left.txt
```

<!-- - Testing on TartanAir -->

Running the above commands with `--save-flow` tag, allows you to save intermediate optical flow outputs into the `results` folder. 

## Run the ROS node 

We provide a python ROS node in `tartanvo_node.py` for easy integration of the TartanVO to robotic systems. 

### How does TartanVONode work?
1. Subscribed topics
   - rgb_image (sensor_msgs/Image): RGB image.
   - cam_info (sensor_msgs/CameraInfo): camera parameters which are used to calculate intrinsics layer. 
   - vo_scale (std_msgs/Float32): scale of the translation (should be published at the same frequncy with the image). If this is not provided, default value of 1.0 will be used. 

2. Published topics
   - tartanvo_pose (geometry_msgs/PoseStamped): position and orientation of the camera
   - tartanvo_odom (nav_msgs/Odometry): position and orientation of the camera (same with the `tartanvo_pose`).

3. Parameters: 
   We use the following parameters to calculate the initial intrinsics layer. If the `cam_info` topic is received, the intrinsics value will be over-written. 
   - image_width : image width
   - image_height : image height
   - focal_x : camera focal lengh
   - focal_y : camera focal lengh
   - center_x : camera optical center
   - center_y : camera optical center

### Run the ROS node
1. Open a ROS core:
```
$ roscore
```

2. Run the TartanVONode
```
$ python tartanvo_node.py
```

3. Publish the images and scales, e.g. run the following example
```
$ python publish_image_from_folder.py
```

If you open the `rviz`, you can see the visualization as follows

![RVIZ](results/rviz.png)


## Paper

More technical details are available in the TartanVO paper. Please cite this as:

```
@article{tartanvo2020corl,
  title =   {TartanVO: A Generalizable Learning-based VO},
  author =  {Wang, Wenshan and Hu, Yaoyu and Scherer, Sebastian},
  booktitle = {Conference on Robot Learning (CoRL)},
  year =    {2020}
}
```

```
@article{tartanair2020iros,
  title =   {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author =  {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year =    {2020}
}
```

## License
This software is BSD licensed.

Copyright (c) 2020, Carnegie Mellon University All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.