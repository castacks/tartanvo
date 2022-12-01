"""
Script to generate 2D plot of predicted Vs GT trajectory for all 6 cameras
"""
import numpy as np
import matplotlib.pyplot as plt

ATE = [3.6499, 6.1079, 7.0024,1.1086, 6.7542, 7.1317]
gtposes_front = np.load('/home/nboloor/slam/tartanvo/cam_front_scene_0_gt.npy')
estposes_front = np.load('/home/nboloor/slam/tartanvo/cam_front_scene_0_est.npy')
gtposes_back = np.load('/home/nboloor/slam/tartanvo/cam_back_scene_0_gt.npy')
estposes_back = np.load('/home/nboloor/slam/tartanvo/cam_back_scene_0_est.npy')
gtposes_back_left = np.load('/home/nboloor/slam/tartanvo/cam_back_left_scene_0_gt.npy')
estposes_back_left = np.load('/home/nboloor/slam/tartanvo/cam_back_left_scene_0_est.npy')
gtposes_front_left = np.load('/home/nboloor/slam/tartanvo/cam_front_left_scene_0_gt.npy')
estposes_front_left = np.load('/home/nboloor/slam/tartanvo/cam_front_left_scene_0_est.npy')
gtposes_back_right = np.load('/home/nboloor/slam/tartanvo/cam_back_right_scene_0_gt.npy')
estposes_back_right = np.load('/home/nboloor/slam/tartanvo/cam_back_right_scene_0_est.npy')
gtposes_front_right = np.load('/home/nboloor/slam/tartanvo/cam_front_right_scene_0_gt.npy')
estposes_front_right = np.load('/home/nboloor/slam/tartanvo/cam_front_right_scene_0_est.npy')
savefigname = 'test_new.png'
fig = plt.figure(figsize=(4,4))
cm = plt.cm.get_cmap('Spectral')
title = 'Prediction from 6 cameras'
plt.subplot(111)
plt.plot(gtposes_front[:,0],gtposes_front[:,1], linestyle='dashed',c='k') # ATE: 3.6507
plt.plot(estposes_front[:, 0], estposes_front[:, 1],c='pink')

plt.plot(gtposes_front_left[:,0],gtposes_front_left[:,1], linestyle='dashed',c='k') # ATE: 6.0984
plt.plot(estposes_front_left[:, 0], estposes_front_left[:, 1],c='yellow')

plt.plot(gtposes_front_right[:,0],gtposes_front_right[:,1], linestyle='dashed',c='k') # ATE: 3.6450
plt.plot(estposes_front_right[:, 0], estposes_front_right[:, 1],c='orange')



plt.plot(gtposes_back[:,0],gtposes_back[:,1], linestyle='dotted',c='k') # ATE: 3.6325
plt.plot(estposes_back[:, 0], estposes_back[:, 1],c='#33FF3C')

plt.plot(gtposes_back_left[:,0],gtposes_back_left[:,1], linestyle='dashed',c='k') # ATE: 3.6248
plt.plot(estposes_back_left[:, 0], estposes_back_left[:, 1],c='red')

plt.plot(gtposes_back_right[:,0],gtposes_back_right[:,1], linestyle='dashed',c='k') # ATE: 3.6396
plt.plot(estposes_back_right[:, 0], estposes_back_right[:, 1],c='blue')

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend(['GT Front','TartanVO Front ATE: 3.6499', 'GT Front Left','TartanVO Front Left ATE:  6.1079', 'GT Front Right','TartanVO Front Right ATE: 7.0024', 'GT Back','TartanVO Back ATE: 1.1086', 'GT Back Left','TartanVO Back Left ATE: 6.7542', 'GT Back Right','TartanVO Back Right ATE: 7.1317'], loc=5, prop={'size': 5.5})
plt.title(title)
if savefigname is not None:
    plt.savefig(savefigname)
plt.close(fig)