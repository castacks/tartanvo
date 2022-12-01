"""
Script to generate pose files per camera in the nuscenes dataset
"""
import cv2
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/home/nboloor/slam/data/nuscenes_mini', verbose=True)

'''
Fetch a scene, the first and last sample in that scene
'''
scene_1 = nusc.scene[0]
sensors = ['CAM_FRONT','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK','CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
print('first sample token in the scene: ', scene_1['first_sample_token'])
start_sample = nusc.get('sample',scene_1['first_sample_token'])
last_sample = nusc.get('sample',scene_1['last_sample_token'])

'''
For each sensor [6 cameras in our case] get ego_key_token of key frames only
'''
for sensor in sensors:
    start_cam = nusc.get('sample_data', start_sample['data'][sensor])
    last_cam = nusc.get('sample_data', last_sample['data'][sensor])
    frame_count = 0
    key_frame_sample_tokens = []
    ego_key_tokens = []
    if start_cam['is_key_frame']:
        frame_count += 1
        key_frame_sample_tokens.append(start_cam['sample_token'])
        ego_key_tokens.append(start_cam['ego_pose_token'])
    while not start_cam['next'] == "":
        next_cam = nusc.get('sample_data', start_cam['next'])
        while not next_cam['is_key_frame']:
            next_cam = nusc.get('sample_data', next_cam['next'])
        frame_count += 1
        start_cam = next_cam
        key_frame_sample_tokens.append(start_cam['sample_token'])
        ego_key_tokens.append(start_cam['ego_pose_token'])

    '''
    Store the  poses per camera in a txt file
    '''
    with open(sensor+'_pose.txt','w') as log:
        for ego in ego_key_tokens:
            ego_ele = nusc.get('ego_pose',ego)
            string = ''
            for ele in ego_ele['translation']:
                string+=str(ele)+' '
            for ele in ego_ele['rotation']:
                string+=str(ele)+' '
            string+='\n'
