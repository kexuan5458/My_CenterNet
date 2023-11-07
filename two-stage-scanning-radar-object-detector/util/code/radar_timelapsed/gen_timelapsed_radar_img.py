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

radar_path = "/data2/itri/DCV/single_sensor_data/bag1_/radar"
radar_file = os.listdir(radar_path)
radar_file.sort()
radar_timestamp = [int(x[:-4]) for x in radar_file]

output_path = "/data2/itri/DCV/single_sensor_data/bag1_/radar_cartesian_timelapsed_noflip"
output_polar_path = "/data2/itri/DCV/single_sensor_data/bag1_/radar_polar_timelapsed_noflip"

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

for idx, rt in enumerate(tqdm(radar_timestamp[:-1])):
	id_list = [rt, radar_timestamp[radar_timestamp.index(rt) + 1]]
	img_list = []

	for x in id_list:
		with open("{}/{}.pkl".format(radar_path, x), "rb") as f:
			radar_msg = pickle.load(f)
		cv_image = bridge.imgmsg_to_cv2(radar_msg, "mono8")
		cv_image = cv_image.T
		cv_image = cv_image[:, :]
		img_list.append(cv_image)

	combined_time = sum(id_list) // 2
	combined_img = combine_two_half(img_list)
	cv2.imwrite(
		os.path.join(
			output_polar_path,
			"{}.png".format(combined_time)
		),
		combined_img
	)

	azimuths = np.linspace(0, 2*math.pi, combined_img.shape[0]+1).reshape(-1, 1)
	valid = np.empty_like(azimuths, dtype=bool)
	valid.fill(True)
	fft_data = combined_img / 255.

	cart_img = radar_polar_to_cartesian(azimuths, fft_data)
	radar_img = (cart_img * 255.0).astype(np.uint8)
	# combined_img = np.dstack((radar_img, )*3)[::-1, ::-1]
	combined_img = np.dstack((radar_img, )*3)

	cv2.imwrite(
		os.path.join(
			output_path,
			"{}.png".format(combined_time)
		),
		combined_img
	)

  