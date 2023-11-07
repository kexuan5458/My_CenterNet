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

radar_path = "/data2/itri/DCV/single_sensor_data/bag2/radar"
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
	lidar_stamp = [lidar_stamp[0], lidar_stamp[-1]]
	lidar2_stamp = [lidar2_stamp[0], lidar2_stamp[-1]]
	livox_stamp = [livox_stamp[0], livox_stamp[-1]]

	# baraja 1
	for ls in lidar_stamp[:]:
		lidar_id += 1
		with open("{}/{}.pkl".format(lidar_path, ls), "rb") as f:
			lidar = pickle.load(f)
		point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar)
		point = shift_point('base_link->sensorhead_1', point)
		
		# import ipdb; ipdb.set_trace(context=7)
		# plt.hist(point[:,2], 100)
		height_mask = point[:, 2] > -20
		point = point[height_mask, :]
		

		x = -point[:, 1]-0.03
		# y = -point[:, 0]
		y = point[:, 0]+0.6
		z = x + 1j * y
		r, ang = z2polar(z)
		# ang += (ang < 0).astype(np.float32) * 2. * np.pi
		# import ipdb; ipdb.set_trace(context=7)
		
		ang = np.degrees(ang)
		ang += (ang < 0).astype(np.float32) * 360
		# import ipdb; ipdb.set_trace(context=7)
		
		# pts_selection = np.logical_and(ang > 0, ang < np.pi/2)
		# pts_selection = (ang != 0)

		if lidar_id == 0:
			pts_selection = np.logical_and(ang >= 30, ang < 90)
			continue
	
		# elif lidar_id == 1:
		# 	continue

		# elif lidar_id == 2:
		# 	continue

		# elif lidar_id == 3:
		# 	continue

		elif lidar_id == 1:
			pts_selection = np.logical_and(ang >= 90, ang < 150)
		
		else:
			continue

		pts_selection = (ang != 0)

		# cc = ['red', 'yellow', 'green', 'cyan', 'orange']
		cc = ['mediumslateblue', 'orange']
		plt.scatter(x[pts_selection], y[pts_selection], s=0.06, alpha=0.8, c=cc[lidar_id])
		plt.scatter([0], [0], s=1, alpha=0.8, c='red')

	# baraja 2
	for ls in lidar2_stamp[:]:
		lidar2_id += 1
		with open("{}/{}.pkl".format(lidar2_path, ls), "rb") as f:
			lidar2 = pickle.load(f)
		point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar2)
		
		point = shift_point('base_link->sensorhead_2', point)
		
		# import ipdb; ipdb.set_trace(context=7)
		# plt.hist(point[:,2], 100)
		height_mask = point[:, 2] > -20
		point = point[height_mask, :]
		

		x = -point[:, 1]-0.03
		# y = -point[:, 0]
		y = point[:, 0]+0.6
		z = x + 1j * y
		r, ang = z2polar(z)
		# ang += (ang < 0).astype(np.float32) * 2. * np.pi
		# import ipdb; ipdb.set_trace(context=7)
		
		ang = np.degrees(ang)
		ang += (ang < 0).astype(np.float32) * 360
		# import ipdb; ipdb.set_trace(context=7)
		
		# pts_selection = np.logical_and(ang > 0, ang < np.pi/2)
		# pts_selection = (ang != 0)

		if lidar2_id == 0:
			pts_selection = np.logical_and(ang >= 30, ang < 90)
			continue
	
		# elif lidar_id == 1:
		# 	continue

		# elif lidar_id == 2:
		# 	continue

		# elif lidar_id == 3:
		# 	continue

		elif lidar2_id == 1:
			pts_selection = np.logical_and(ang >= 90, ang < 150)
		
		else:
			continue

		pts_selection = (ang != 0)

		# cc = ['red', 'yellow', 'green', 'cyan', 'orange']
		cc = ['mediumslateblue', 'orange']
		plt.scatter(x[pts_selection], y[pts_selection], s=0.06, alpha=0.8, c=cc[lidar2_id])
		plt.scatter([0], [0], s=1, alpha=0.8, c='red')

		
		# lidar_scans.append(lidar)

		# return angle
		# plt.subplot(122)
		# plt.scatter(np.arange(len(angle[1])), angle[1], s=0.5, c=np.arange(len(angle[1])), cmap="binary")
		
		# import ipdb; ipdb.set_trace(context=7)
	
	# velodyne
	for ls in livox_stamp[:]:
		livox_id += 1
		with open("{}/{}.pkl".format(livox_path, ls), "rb") as f:
			livox = pickle.load(f)
		point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(livox)
		point = shift_point('base_link->velodyne', point)
		
		# import ipdb; ipdb.set_trace(context=7)
		# plt.hist(point[:,2], 100)
		height_mask = point[:, 2] > 1.2
		point = point[height_mask, :]

		x = -point[:, 1]+0.18
		# y = -point[:, 0]
		y = point[:, 0]+0.60
		z = x + 1j * y
		r, ang = z2polar(z)
		# ang += (ang < 0).astype(np.float32) * 2. * np.pi
		# import ipdb; ipdb.set_trace(context=7)
		
		ang = np.degrees(ang)
		ang += (ang < 0).astype(np.float32) * 360
		# import ipdb; ipdb.set_trace(context=7)
		
		# pts_selection = np.logical_and(ang > 0, ang < np.pi/2)
		# pts_selection = (ang != 0)

		if livox_id == 0:
			pts_selection = np.logical_and(ang >= 30, ang < 90)
			continue
	
		# elif livox_id == 1:
		# 	continue

		# elif livox_id == 2:
		# 	continue

		# elif livox_id == 3:
		# 	continue

		elif livox_id == 1:
			pts_selection = np.logical_and(ang >= 90, ang < 150)
		
		else:
			continue


		pts_selection = (ang != 0)

		cc = ['sienna', 'orange']
		plt.scatter(x[pts_selection], y[pts_selection], s=0.06, alpha=0.8, c='forestgreen')

	# xs = np.arange(0, 0.175*1142/2+0.1, 0.175*1142/2/10)
	# ys = np.arange(0, 0.175*1142/2+0.1, 0.175*1142/2/10)
	# xt, xr = np.meshgrid(xs, ys)
	# plt.scatter(xt.ravel(), xr.ravel(), s=10, c="red")

	# import ipdb; ipdb.set_trace(context=7)
	


	# radar image block
	with open("{}/{}.pkl".format(radar_path, radar_stamp), "rb") as f:
		radar_msg = pickle.load(f)

	cv_image = bridge.imgmsg_to_cv2(radar_msg, "mono8")
	cv_image = cv_image.T
	cv_image = cv_image[:,:,np.newaxis]

	azimuths = np.linspace(0, 2*math.pi, cv_image.shape[0]+1).reshape(-1, 1)
	valid = np.empty_like(azimuths, dtype=bool)
	valid.fill(True)
	fft_data = cv_image / 255.

	cart_img = radar_polar_to_cartesian(azimuths, fft_data)
	radar_img = cart_img * 255.0
	# import ipdb; ipdb.set_trace(context=7)

	row, col = 2, 2
	img_h, img_w = 5712, 5712
	devision = 5

	sub_crop = np.array_split(np.arange(img_h), devision)
	output_img = radar_img[np.ix_(sub_crop[row], sub_crop[col])]
	output_img = output_img.astype(np.uint8)
	output_img = np.dstack((output_img,)*3)
	output_img = Image.fromarray(output_img)
	# import ipdb; ipdb.set_trace(context=7)
	

	# cv2.imshow("out", output_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# img = Image.fromarray(output_img)
	# img.open()

	### for lidar overlay
	# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)


	boundary = 0.175*1142/2
	plt.xlim(-boundary,boundary)
	plt.ylim(-boundary,boundary)
	plt.gca().set_aspect('equal', adjustable='box')

	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
							hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	# plt.axis('off')

	# plt.savefig("tttmp.png", bbox_inches = 'tight', pad_inches = 0)


	buffer_ = BytesIO()

	fig = plt.gcf()
	dpi = fig.get_dpi()
	# import ipdb; ipdb.set_trace(context=7)
	
	fig.set_size_inches(1142.0/float(dpi), 1142.0/float(dpi))
	fig.set_dpi(dpi)

	plt.savefig(buffer_, format = "PNG", bbox_inches = 'tight', pad_inches = 0, transparent=True)
	# plt.savefig('test.png', format = "PNG", bbox_inches = 'tight', pad_inches = 0, transparent=False, dpi=dpi)
	buffer_.seek(0)	

	lidar_img = Image.open(buffer_)
	# import ipdb; ipdb.set_trace(context=7)
	
	# lidar_img.save("test.png")
	# output_img.save("test1.png")
	

	final = output_img.convert('RGBA')

	# final = Image.new("RGBA", lidar_img.size)
	
	final = Image.alpha_composite(final, lidar_img)
	# final = Image.alpha_composite(final, layer2)

	# final.show()
	# import ipdb; ipdb.set_trace(context=7)
	# final.save("/data2/itri/RAV4/2021-11-24-11-38-59/processing/radar_baraja_livox/{}.png".format(k))
	# # output_img.save("/data2/itri/RAV4/2021-11-24-11-38-59/processing/radar_raw/{}.png".format(k))
	final.save("/data2/itri/DCV/overlays/final/{}.png".format(k))
	output_img.save("/data2/itri/DCV/overlays/output_img/{}.png".format(k))

	# import ipdb; ipdb.set_trace(context=7)
	


for k, v in tqdm(list(radar_lidar_group.items())[:]):
	# img = combine_radar_lidar(k, v, 1)
	img = combine_radar_lidar(k, v, radar_lidar2_group[k], radar_livox_group[k], 5)
