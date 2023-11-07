import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from collections import defaultdict
import cv2

plt.rcParams['figure.dpi'] = 100
# plt.rcParams['axes.facecolor'] = 'black'

bag1_camera_path = "/data2/itri/DCV/single_sensor_data/bag1/camera_0"
bag2_camera_path = "/data2/itri/DCV/single_sensor_data/bag2/camera_0"

# anno_path = "/data2/itri/RAV4/2021-11-24-11-53-49/radar_baraja_overlay_0_p100_timelapsed_no_comp"
# anno_path = "/data2/itri/RAV4/2021-11-24-11-53-49/radar_lidar_overlay_0_p100_timelapsed_no_comp"
anno_path = "/data2/itri/DCV/single_sensor_data/201007_livox_anno_radar_det_0_100"

# output_path = "/data2/itri/RAV4/2021-11-24-11-53-49/radar_baraja_camera"
output_path = "/data2/itri/DCV/single_sensor_data/201007_camera_livox_anno_radar_det_0_100"

anno_file = os.listdir(anno_path)
anno_file.sort()
# anno_file = anno_file
# anno_file = [int(x[:-4]) for x in anno_file]


bag1_camera_file = os.listdir(bag1_camera_path)
bag1_camera_file.sort()
bag1_time = [int(x[:-4]) for x in bag1_camera_file]

bag2_camera_file = os.listdir(bag2_camera_path)
bag2_camera_file.sort()
bag2_time = [int(x[:-4]) for x in bag2_camera_file]


for i, rf in enumerate(tqdm(anno_file[:])):
  fname = rf.split('_')[-1]
  time = int(fname[:-4])
  if 'bag1' in rf:
    closet_idx = min(range(len(bag1_time)), key=lambda i: abs(bag1_time[i]-time))
    camera_root = bag1_camera_path
    camera_fname = bag1_camera_file[closet_idx]
  if 'bag2' in rf:
    closet_idx = min(range(len(bag2_time)), key=lambda i: abs(bag2_time[i]-time))
    camera_root = bag2_camera_path
    camera_fname = bag2_camera_file[closet_idx]
  
  # print(time, camera_fname)
  
  anno_img = cv2.imread(os.path.join(anno_path, rf))
  camera_img = cv2.imread(os.path.join(camera_root, camera_fname))

  combined = np.zeros((anno_img.shape[0], camera_img.shape[1], 3))
  combined[:camera_img.shape[0], :camera_img.shape[1], :] = camera_img
  
  combined = np.hstack((combined, anno_img))
  cv2.putText(combined, str(time), (20, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

  cv2.imwrite(os.path.join(output_path, rf), combined)
  
