import numpy as np
import os
from tqdm import tqdm
import cv2

radar_path = "/data2/itri/DCV/single_sensor_data/bag1_/radar_cartesian_timelapsed_noflip"

output_path = "/data2/itri/DCV/single_sensor_data/bag1_/radar_cart_0_200_timelapsed"

radar_file = os.listdir(radar_path)
radar_file.sort()

img_h, img_w = 5712, 5712
devision = 5
sub_crop = np.array_split(np.arange(img_h), devision)

for rf in tqdm(radar_file[:]):
	rim = cv2.imread(
		os.path.join(
			radar_path,
			rf
	))

	rim = rim[::-1, ::-1]

	row, col = 1, 2
	top_bottom = rim[np.ix_(sub_crop[row], sub_crop[col])]
	top_bottom = top_bottom.astype(np.uint8)
	top_bottom = top_bottom[top_bottom.shape[0]//2+1:]

	row, col = 2, 2
	bottom_top = rim[np.ix_(sub_crop[row], sub_crop[col])]
	bottom_top = bottom_top.astype(np.uint8)
	bottom_top = bottom_top[:bottom_top.shape[0]//2]

	out_img = np.vstack((top_bottom, bottom_top))


	cv2.imwrite(
		os.path.join(
			output_path,
			rf
		),
		out_img
	)