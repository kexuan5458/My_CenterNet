# this file convert radar.pkl to radar.png, and save 0m ~ -200m cropped img
import numpy as np
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

### ros import 
import ros_numpy
from cv_bridge import CvBridge
###

bridge = CvBridge()

radar_path = "/data2/itri/DCV/single_sensor_data/bag2/radar"
radar_file = os.listdir(radar_path)
radar_file.sort()
radar_timestamp = [int(x[:-4]) for x in radar_file]

all_cart_out = "/data2/itri/DCV/single_sensor_data/bag2/radar_img"
cart_0_n200 = "/data2/itri/DCV/single_sensor_data/bag2/radar_cart_0_n200"

def radar_polar_to_cartesian(azimuths, fft_data, radar_resolution=0.175,
                             cart_resolution=0.175, cart_pixel_width=5712, interpolate_crossover=True):
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    polar_to_cart_warp = polar_to_cart_warp.astype(np.float32)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img

def combine_two_half(img_list):
	im1, im2 = img_list
	h, w = im1.shape
	return np.vstack((im1[h//2:, :], im2[:h//2, :]))


img_h, img_w = 5712, 5712
devision = 5
sub_crop = np.array_split(np.arange(img_h), devision)

for idx, rt in enumerate(tqdm(radar_timestamp[:])):
	id_list = [rt, radar_timestamp[radar_timestamp.index(rt) + 1]]
	img_list = []

	for x in id_list:
		with open("{}/{}.pkl".format(radar_path, x), "rb") as f:
			radar_msg = pickle.load(f)
		cv_image = bridge.imgmsg_to_cv2(radar_msg, "mono8")
		cv_image = cv_image.T
		cv_image = cv_image[:, :]
		img_list.append(cv_image)

	azimuths = np.linspace(0, 2*math.pi, cv_image.shape[0]+1).reshape(-1, 1)
	valid = np.empty_like(azimuths, dtype=bool)
	valid.fill(True)
	fft_data = cv_image / 255.

	cart_img = radar_polar_to_cartesian(azimuths, fft_data)
	radar_img = (cart_img * 255.0).astype(np.uint8)
	combined_img = np.dstack((radar_img,)*3)

	# cv2.imwrite(
	# 	os.path.join(
	# 		all_cart_out,
	# 		"{}.png".format(rt)
	# 	),
	# 	combined_img
	# )

	rim = combined_img

	row, col = 2, 2
	top_bottom = rim[np.ix_(sub_crop[row], sub_crop[col])]
	top_bottom = top_bottom.astype(np.uint8)
	top_bottom = top_bottom[top_bottom.shape[0]//2:]

	row, col = 3, 2
	bottom_top = rim[np.ix_(sub_crop[row], sub_crop[col])]
	bottom_top = bottom_top.astype(np.uint8)
	bottom_top = bottom_top[:bottom_top.shape[0]//2]

	out_img = np.vstack((top_bottom, bottom_top))
  
	cv2.imwrite(
		os.path.join(
			cart_0_n200,
			"{}.png".format(rt)
		),
		out_img
	)