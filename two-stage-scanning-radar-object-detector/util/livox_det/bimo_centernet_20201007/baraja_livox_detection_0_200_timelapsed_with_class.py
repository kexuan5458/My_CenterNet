# from livox_baraja_merge_config import config as cfg
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))



print(sys.path)
# import ipdb; ipdb.set_trace(context=7)


import pickle5 as pickle
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import copy

# import importlib
# importlib.reload(config)

# import ipdb; ipdb.set_trace(context=7)

from networks.model import *
import lib_cpp

import time

# import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Quaternion
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from collections import defaultdict
import ros_numpy
from io import BytesIO
from PIL import Image
import cv2
# from livox_baraja_merge_config import config as cfg
from livox_baraja_merge_config import bimo_config as cfg



import pickle5 as pickle
with open('/home/ee904/NCTU/research/itri_related/20201007/code/tf_data/sensor_tf.pickle', 'rb') as f:
    tfs = pickle.load(f)




# radar_path = "/data2/itri/RAV4/2021-11-24-11-38-59/radar"
radar_path = "/data2/itri/DCV/single_sensor_data/bag2/radar_cart_0_200_timelapsed"
lidar_path = "/data2/itri/DCV/single_sensor_data/bag2/baraja_1"
livox_path = "/data2/itri/DCV/single_sensor_data/bag2/velodyne"

bbox_npy_path = "/data2/itri/DCV/single_sensor_data/bag2/baraja_velodyne_detection_box_npy"
img_out_path = '/data2/itri/DCV/single_sensor_data/bag2/radar_lidar_bbox'

row, col = 2, 2
img_h, img_w = 5712, 5712
devision = 5
sub_crop = np.array_split(np.arange(img_h), devision)

img_id = 0
mnum = 0
marker_array = MarkerArray()
marker_array_text = MarkerArray()

DX = cfg.VOXEL_SIZE[0]
DY = cfg.VOXEL_SIZE[1]
DZ = cfg.VOXEL_SIZE[2]

X_MIN = cfg.RANGE['X_MIN']
X_MAX = cfg.RANGE['X_MAX']

Y_MIN = cfg.RANGE['Y_MIN']
Y_MAX = cfg.RANGE['Y_MAX']

Z_MIN = cfg.RANGE['Z_MIN']
Z_MAX = cfg.RANGE['Z_MAX']

overlap = cfg.OVERLAP
HEIGHT = round((X_MAX - X_MIN+2*overlap) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)


# print(HEIGHT, WIDTH, CHANNELS)
print("cfg =>>>>>: ", cfg)

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
              )
# lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
#          [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

obj_list = []


class Detector(object):
    def __init__(self, *, nms_threshold=0.1, weight_file=None):
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        self.obj_list = None
        self.obj_cls_list = None
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
                input_bev_img_pl = \
                    self.net.placeholder_inputs(cfg.BATCH_SIZE)
                end_points = self.net.get_model(input_bev_img_pl)

                saver = tf.train.Saver()
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                self.sess = tf.Session(config=config)
                saver.restore(self.sess, cfg.MODEL_PATH)
                
                self.ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                            'end_points': end_points,  # output
                            }
        # rospy.init_node('livox_test', anonymous=True)

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    def get_3d_box(self, box_size, heading_angle, center):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (l,w,h)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.roty(heading_angle)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def data2voxel(self, pclist):

        data = [i*0 for i in range(HEIGHT*WIDTH*CHANNELS)]

        for line in pclist:
            X = float(line[0])
            Y = float(line[1])
            Z = float(line[2])
            if(Y > Y_MIN and Y < Y_MAX and
                X > X_MIN and X < X_MAX and
                    Z > Z_MIN and Z < Z_MAX):
                channel = int((-Z + Z_MAX)/DZ)
                if abs(X) < 3 and abs(Y) < 3:
                    continue
                if (X > -overlap):
                    pixel_x = int((X - X_MIN + 2*overlap)/DX)
                    pixel_y = int((-Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
                if (X < overlap):
                    pixel_x = int((-X + overlap)/DX)
                    pixel_y = int((Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
        voxel = np.reshape(data, (HEIGHT, WIDTH, CHANNELS))
        return voxel

    def detect(self, batch_bev_img):
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)
        result = lib_cpp.cal_result(feature_out[0, :, :, :],
                                    cfg.BOX_THRESHOLD, overlap, X_MIN, HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        is_obj_list = result[:, 0].tolist()

        reg_m_x_list = result[:, 5].tolist()
        reg_w_list = result[:, 4].tolist()
        reg_l_list = result[:, 3].tolist()
        obj_cls_list = result[:, 1].tolist()
        reg_m_y_list = result[:, 6].tolist()
        reg_theta_list = result[:, 2].tolist()
        reg_m_z_list = result[:, 8].tolist()
        reg_h_list = result[:, 7].tolist()

        results = []
        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)
            box3d_pts_3d[:, 0:3] = self.get_3d_box(
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]),
                reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T  # n*4
            if int(obj_cls_list[i]) == 0:
                cls_name = "car"
                cls_name = 0
            elif int(obj_cls_list[i]) == 1:
                cls_name = "bus"
                cls_name = 1
            elif int(obj_cls_list[i]) == 2:
                cls_name = "truck"
                cls_name = 2
            elif int(obj_cls_list[i]) == 3:
                cls_name = "pedestrian"
                cls_name = 3
            else:
                cls_name = "bimo"
                cls_name = 4
            results.append([cls_name,
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i]])
        return results

    def draw_obj_list(self, radar_timestamp, lidar_timestamp):
        np.savez(os.path.join(
                bbox_npy_path,
                str(radar_timestamp)
            ),
            cls=np.array(self.obj_cls_list),
            bbox=np.array(self.obj_list))

        # import ipdb; ipdb.set_trace(context=7)
        
        global img_id
        boundary = 0.175*1142/2

        plt.clf()

        for i in range(len(self.obj_list)):
            obj = self.obj_list[i]
            x = -obj[:, 1]-0.03
            y = obj[:, 0]+0.6
            plt.plot(x, y, color='white')

        fig = plt.gcf()
        DPI = fig.get_dpi()
        fig.set_size_inches(1142.0/float(DPI), 1142.0/float(DPI))
        plt.xlim(-boundary,boundary)
        plt.ylim(0,boundary*2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)

        # plt.savefig("{}/{}".format("overlay_output", timestamp), format = "PNG", pad_inches = 0, transparent=True)
        img_id += 1

        buffer_ = BytesIO()
        plt.savefig(buffer_, format="PNG", bbox_inches='tight',
                    pad_inches=0, transparent=True)
        buffer_.seek(0)

        lidar_anno = Image.open(buffer_)

        radar_img = cv2.imread(
            "{}/{}.png".format(radar_path, radar_timestamp))
        # print("error: ", radar_timestamp)

        # radar_img = cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB)

        # radar_img = radar_img[np.ix_(sub_crop[row], sub_crop[col])]

        radar_img = Image.fromarray(radar_img[:, :, ::-1])
        final = radar_img.convert('RGBA')
        final = Image.alpha_composite(final, lidar_anno)
        final.save(
            '{}/{}.png'.format(img_out_path, radar_timestamp))

    def LivoxCallback(self, msg, radar_timestamp, lidar_timestamp):
        obj_list = []
        obj_cls_list = []

        # header = std_msgs.msg.Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = 'livox_frame'
        # points_list = []

        # import ipdb; ipdb.set_trace(context=7)

        # msg[:, [0, 1, 2]] = msg[:, [1, 0, 2]]
        # msg[:, 1] = -msg[:, 1] + 0.75

        # for point in pcl2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
        #     if point[0] == 0 and point[1] == 0 and point[2] == 0:
        #         continue
        #     if np.abs(point[0]) < 2.0 and np.abs(point[1]) < 1.5:
        #         continue
        #     points_list.append(point)
        # points_list = np.asarray(points_list)
        # import ipdb; ipdb.set_trace(context=7)

        # pointcloud_msg = pcl2.create_cloud_xyz32(header, points_list[:, 0:3])
        # pointcloud_msg = pcl2.create_cloud_xyz32(header, msg)

        vox = self.data2voxel(msg)
        vox = np.expand_dims(vox, axis=0)
        t0 = time.time()
        result = self.detect(vox)
        # import ipdb; ipdb.set_trace(context=7)

        # t1 = time.time()

        # print('det_time(ms)', 1000*(t1-t0))
        # print('det_numbers', len(result))

        for ii in range(len(result)):
            result[ii][1:9] = list(np.array(result[ii][1:9]))
            result[ii][9:17] = list(np.array(result[ii][9:17]))
            result[ii][17:25] = list(np.array(result[ii][17:25]))

        boxes = result

        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append([ob[i+1], ob[i+9], ob[i+17]])

            single_obj = []

            for line in lines:
                # print("line: ", line)
                single_obj.append(detect_points_set[line[0]])
                single_obj.append(detect_points_set[line[1]])

            obj_list.append(single_obj)
            obj_cls_list.append(ob[0])

        self.obj_list = np.array(obj_list)
        self.obj_cls_list = np.array(obj_cls_list)
        self.draw_obj_list(radar_timestamp, lidar_timestamp)
        # import ipdb; ipdb.set_trace(context=7)


def polar2z(r, theta):
    return r * np.exp(1j * theta)


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


def combine_radar_lidar(radar_stamp, lidar_stamp, livox_stamp, overlap):
    plt.clf()
    lidar_scans = []
    lidar_id = -1
    livox_id = -1
    # lidar_stamp = [lidar_stamp[0], lidar_stamp[-1]]
    # livox_stamp = [livox_stamp[0], livox_stamp[-1]]
    lidar_stamp = [lidar_stamp[-1]]
    livox_stamp = [livox_stamp[-1]]
    for ls in lidar_stamp[:]:
        lidar_id += 1

        # plt.figure()
        with open("{}/{}.pkl".format(lidar_path, ls), "rb") as f:
            lidar = pickle.load(f)
        point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar)
        point = shift_point('base_link->sensorhead_1', point)

        x = -point[:, 1]
        y = point[:, 0]

        selection = np.logical_and((y>0), (y<201))
        x = x[selection]
        y = y[selection]

        lidar_scans.append(point)

    for ls in livox_stamp[:]:
        livox_id += 1
        with open("{}/{}.pkl".format(livox_path, ls), "rb") as f:
            livox = pickle.load(f)
        point = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(livox)
        point = shift_point('base_link->velodyne', point)

        x = -point[:, 1]
        y = point[:, 0]

        selection = np.logical_and((y>0), (y<201))
        x = x[selection]
        y = y[selection]

        lidar_scans.append(point)

    # import ipdb; ipdb.set_trace(context=7)

    # plt.scatter(x[pts_selection], y[pts_selection])
    return np.vstack(lidar_scans)


if __name__ == '__main__':
    livox = Detector()

    radar_file = os.listdir(radar_path)
    radar_file.sort()
    radar_file = radar_file
    radar_file = [int(x[:-4]) for x in radar_file]

    lidar_file = os.listdir(lidar_path)
    lidar_file.sort()
    lidar_file = lidar_file
    lidar_file = [int(x[:-4]) for x in lidar_file]

    livox_file = os.listdir(livox_path)
    livox_file.sort()
    livox_file = livox_file
    livox_file = [int(x[:-4]) for x in livox_file]

    cur_lidar_id = 0
    radar_lidar_group = defaultdict(list)

    cur_livox_id = 0
    radar_livox_group = defaultdict(list)

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

    # for k, v in tqdm(list(radar_lidar_group.items())[4613:]):
    for k, v in tqdm(list(radar_lidar_group.items())[:]):
        # import ipdb; ipdb.set_trace(context=7)
        
        lidar_scans = combine_radar_lidar(k, v, radar_livox_group[k], 5)
        livox.LivoxCallback(lidar_scans, k, v)

    # pkl_path = "/data2/itri/rosbag_api_sync/dji/2021-06-01-16-25-03"
    # time_stamp = "/data2/itri/rosbag_api_sync/dji/2021-06-01-16-25-03/radar_img/16225365492498/2_2_lidar_overlay"

    # time_stamp = os.listdir(time_stamp)
    # time_stamp.sort()
    # time_stamp = [x[:-4] for x in time_stamp]

    # pkl_file = glob.glob("{}/*".format(pkl_path))
    # pkl_file.sort()
    # # print(pkl_file)

    # start = pkl_file.index(
    #     list(filter(lambda x: "16225365492498" in x, pkl_file))[0])

    # for t in tqdm(time_stamp[:]):
    #     print("time: ", t)
    #     with open("{}/{}.pkl".format(pkl_path, t), 'rb') as f:
    #         data = pickle.load(f)

    #     lidar = data['ouster']
    #     livox.LivoxCallback(lidar, t)
