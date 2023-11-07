import sys
sys.path.insert(0, '.')
# import ipdb; ipdb.set_trace(context=7)



import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import ros_numpy
from tqdm import tqdm
from collections import defaultdict
# from mpl_toolkits.mplot3d import Axes3D
import cv2
from cv_bridge import CvBridge
import math
# import napari
from io import BytesIO
from PIL import Image


from save_single_sensor.read_tf import tfs


# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['axes.facecolor'] = 'black'

radar_path = "/data2/itri/DCV/single_sensor_data/bag2/radar_cart_0_200_timelapsed"
lidar_path = "/data2/itri/DCV/single_sensor_data/bag2/baraja_1"
lidar2_path = "/data2/itri/DCV/single_sensor_data/bag2/baraja_2"
livox_path = "/data2/itri/DCV/single_sensor_data/bag2/velodyne"

bridge = CvBridge()

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

# %%
radar_file = os.listdir(radar_path)
radar_file.sort()
radar_file = radar_file
radar_file = [int(x[:-4]) for x in radar_file]

lidar_file = os.listdir(lidar_path)
lidar_file.sort()
lidar_file = lidar_file
lidar_file = [int(x[:-4]) for x in lidar_file]

lidar2_file = os.listdir(lidar2_path)
lidar2_file.sort()
lidar2_file = lidar2_file
lidar2_file = [int(x[:-4]) for x in lidar2_file]

livox_file = os.listdir(livox_path)
livox_file.sort()
livox_file = livox_file
livox_file = [int(x[:-4]) for x in livox_file]

cur_lidar_id = 0
radar_lidar_group = defaultdict(list)

cur_lidar2_id = 0
radar_lidar2_group = defaultdict(list)

cur_livox_id = 0
radar_livox_group = defaultdict(list)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

for i, rf in enumerate(radar_file[:-1]):
	if lidar2_file[cur_lidar2_id] > rf:
		continue

	while lidar2_file[cur_lidar2_id] < rf:
		radar_lidar2_group[rf].append(lidar2_file[cur_lidar2_id])
		cur_lidar2_id += 1
	
for i, rf in enumerate(radar_file[:-1]):
	if lidar_file[cur_lidar_id] > rf:
		continue

	while lidar_file[cur_lidar_id] < rf:
		radar_lidar_group[rf].append(lidar_file[cur_lidar_id])
		cur_lidar_id += 1
	
for i, rf in enumerate(radar_file[:-1]):
	if livox_file[cur_livox_id] > rf:
		continue

	while livox_file[cur_livox_id] < rf:
		radar_livox_group[rf].append(livox_file[cur_livox_id])
		cur_livox_id += 1

# for i, rf in enumerate(radar_file[:-1]):
# 	while lidar_file[cur_lidar_id] < rf:
# 		cur_lidar_id += 1

# 	while lidar_file[cur_lidar_id] < radar_file[i+1]:
# 		radar_lidar_group[rf].append(lidar_file[cur_lidar_id])
# 		cur_lidar_id += 1



def polar2z(r,theta):
    return r * np.exp( 1j * theta )

def z2polar(z):
    return (np.abs(z), np.angle(z))

def shift_point(tf_key, point):
	homo_point = np.hstack(
		(
			point,
			np.ones((point.shape[0], 1))
		)
	).T

	tf = tfs[tf_key]

	shifted_point = np.dot(tf, homo_point)
	shifted_point = (shifted_point.T)[:, :3]
	
	return shifted_point



def combine_radar_lidar(radar_stamp, lidar_stamp, lidar2_stamp, livox_stamp, overlap):
	plt.clf()
	lidar_scans = []
	lidar_id = -1
	lidar2_id = -1
	livox_id = -1
	lidar_stamp = [lidar_stamp[-1]]
	lidar2_stamp = [lidar2_stamp[-1]]
	livox_stamp = [livox_stamp[-1]]

	# baraja 1
	for ls in lidar_stamp[:]:
		lidar_id += 1
		with open("{}/{}.pkl".format(lidar_path, ls), "rb") as f:
			lidar = pickle.load(f)
		point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar)
		point = shift_point('base_link->sensorhead_1', point)
		
		# import ipdb; ipdb.set_trace(context=7)
		# plt.hist(point[:,2], 100)
		height_mask = point[:, 2] > 0.2
		point = point[height_mask, :]
		

		x = -point[:, 1]
		y = point[:, 0]
		pts_selection = np.ones_like(x, dtype=bool)

		plt.scatter(x[pts_selection], y[pts_selection], s=0.05, alpha=0.8, c='lime')
		plt.scatter([0], [0], s=1, alpha=0.8, c='red')
		

	# baraja 2
	for ls in lidar2_stamp[:]:
		lidar2_id += 1
		with open("{}/{}.pkl".format(lidar2_path, ls), "rb") as f:
			lidar2 = pickle.load(f)
		point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar2)
		
		point = shift_point('base_link->sensorhead_2', point)
		
		height_mask = point[:, 2] > 0.2
		point = point[height_mask, :]
		

		x = -point[:, 1]
		y = point[:, 0]

		pts_selection = np.ones_like(x, dtype=bool)

		plt.scatter(x[pts_selection], y[pts_selection], s=0.05, alpha=0.8, c='lime')
		plt.scatter([0], [0], s=1, alpha=0.8, c='red')

	
	# velodyne
	for ls in livox_stamp[:]:
		livox_id += 1
		with open("{}/{}.pkl".format(livox_path, ls), "rb") as f:
			livox = pickle.load(f)
		point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(livox)
		point = shift_point('base_link->velodyne', point)
		
		height_mask = point[:, 2] > 1.2
		point = point[height_mask, :]

		x = -point[:, 1]
		y = point[:, 0]

		pts_selection = np.ones_like(x, dtype=bool)

		cc = ['sienna', 'orange']
		plt.scatter(x[pts_selection], y[pts_selection], s=0.05, alpha=0.8, c='orange')
	

	# import ipdb; ipdb.set_trace(context=7)
	

	output_img = cv2.imread(os.path.join(radar_path, str(radar_stamp)+'.png'))
	output_img = Image.fromarray(output_img)

	boundary = 0.175*1142/2
	plt.xlim(-boundary,boundary)
	plt.ylim(0,2*boundary)
	plt.gca().set_aspect('equal', adjustable='box')

	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
							hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	buffer_ = BytesIO()

	fig = plt.gcf()
	dpi = fig.get_dpi()
	
	fig.set_size_inches(1142.0/float(dpi), 1142.0/float(dpi))
	fig.set_dpi(dpi)

	plt.savefig(buffer_, format = "PNG", bbox_inches = 'tight', pad_inches = 0, transparent=True)
	buffer_.seek(0)	

	lidar_img = Image.open(buffer_)

	final = output_img.convert('RGBA')
	final = Image.alpha_composite(final, lidar_img)

	# final.save("/data2/itri/DCV/single_sensor_data/bag2/radar_lidar_overlay/{}.png".format(k))
	final.save("/root/NCTU/research/itri_related/20201007/code/overlay/bag2_overlay_test/{}.png".format(k))
	


for k, v in tqdm(list(radar_lidar_group.items())[:]):
	# img = combine_radar_lidar(k, v, 1)
	img = combine_radar_lidar(k, v, radar_lidar2_group[k], radar_livox_group[k], 5)
