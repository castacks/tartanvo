"""
Nuscenes dataloader to fetch images [key-frames only], rotation and pose for 6 cameras
We fectch only the key-frames because : 
A keyframe is a frame where the time-stamps of data from all the sensors should be very close to the time-stamp of the sample it points to.
"""
from collections import defaultdict
import os.path as osp

import cv2
from nuscenes.nuscenes import NuScenes as NuScenesDevKit
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T


class NuScenes:
    def __init__(
        self,
        version='v1.0-trainval',
        dataroot='/home/nboloor/slam/data/nuscenes',
        image_width=640,
        image_height=448,
    ):
        self.dataroot = dataroot
        self.nusc = NuScenesDevKit(version=version, dataroot=dataroot)
        self.sensors = ['CAM_FRONT','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK','CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.scenes = self.nusc.scene

        self.index = []
        for scene_id, scene in enumerate(self.scenes):
            frame_ids = [
                (scene_id, frame_id)
                for frame_id in range(scene["nbr_samples"])
            ]
            self.index.extend(frame_ids)

        self.transforms = T.Compose([
            T.ToTensor(),
            T.CenterCrop((image_height, image_width)),
        ])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """
        Returns the 6 images [key-frames only] and corresponding camera poses [translation, rotation]
        """
        scene_id, frame_id = self.index[index]

        scene = self.scenes[scene_id]
        start_sample = self.nusc.get('sample', scene['first_sample_token'])

        output_dict = defaultdict(dict, {k: {} for k in self.sensors})
        for sensor in self.sensors:
            curr_cam = self.nusc.get('sample_data', start_sample['data'][sensor])

            # get the keyframe corresponding to frame_id
            curr_frame_id = -1
            if curr_cam['is_key_frame']:
                curr_frame_id += 1
            while not curr_cam['next'] == "":
                if curr_frame_id == frame_id:
                    break
                next_cam = self.nusc.get('sample_data', curr_cam['next'])
                while not next_cam['is_key_frame']:
                    next_cam = self.nusc.get('sample_data', next_cam['next'])
                curr_frame_id += 1
                curr_cam = next_cam

            # load the image
            sample_data = self.nusc.get('sample_data', curr_cam['token'])
            filename = osp.join(self.dataroot, sample_data['filename'])
            output_dict[sensor]['image'] = self.transforms(cv2.imread(filename))

            # load the pose for current camera
            ego_pose_token = curr_cam['ego_pose_token']
            ego_pose = self.nusc.get('ego_pose', ego_pose_token)
            output_dict[sensor]['translation'] =  torch.FloatTensor(ego_pose['translation'])
            output_dict[sensor]['rotation'] = torch.FloatTensor(ego_pose['rotation'])
        return output_dict


if __name__ == "__main__":
    nuscenes_dataset = NuScenes()
    nuscenes_loader = DataLoader(
        nuscenes_dataset,
        batch_size=40,
        shuffle=False,
        num_workers=0,
    )

    for idx, batch in enumerate(nuscenes_loader):
        cam_front = batch["CAM_FRONT"]
        img_cam_front = cam_front["image"]
        rot_cam_front = cam_front["rotation"]
        trans_cam_front = cam_front["translation"]
        import pdb; pdb.set_trace()
