import cv2
import os
from tqdm import tqdm


# input_path = '/data2/itri/RAV4/2021-11-24-11-53-49/radar_baraja_overlay_0_p200_timelapsed_no_comp'
# output_path = '/data2/itri/RAV4/2021-11-24-11-53-49/radar_baraja_overlay_0_p100_timelapsed_no_comp'
input_path = '/home/ee904/NCTU/research/itri_related/20201007/code/draw_anno_detection/tmp_bbox'
output_path = '/data2/itri/DCV/single_sensor_data/201007_livox_anno_radar_det_0_100'

input_f = sorted(os.listdir(input_path))

for f in tqdm(input_f):
	img = cv2.imread(os.path.join(input_path, f))
	h, w, _ = img.shape
	crop = img[h//2:, w//4:3*w//4, :]

	cv2.imwrite(os.path.join(output_path, f), crop)