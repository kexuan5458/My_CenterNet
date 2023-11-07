# for saving radar image and msg(radar, lidar, camera) pkl file
# from genmsg.base import IODELIM
import rosbag
import cv2
from cv_bridge import CvBridge
import itertools
import numpy as np
import math
# import napari
import pickle
# import cPickle as pickle
from tqdm import tqdm

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

import collections

import argparse
import genpy
import importlib
import logging

def obj_from_str(module_name, class_name, *args, **kwargs):
    # Get an instance of module_name.class_name
    mod = importlib.import_module(module_name)
    obj = getattr(mod, class_name)(*args, **kwargs)
    return obj


def rewrite(x):
    # if isinstance(x, (bool, int, long, float, complex, str, genpy.Time, genpy.Duration, rospy.Time, rospy.Duration)):
    if isinstance(x, (bytes, bool, int, float, complex, str, genpy.Time, genpy.Duration, rospy.Time, rospy.Duration)):
        # A primitive type (see http://wiki.ros.org/msg)
        return x
    elif isinstance(x, list):
        return [rewrite(item) for item in x]
    elif isinstance(x, tuple):
        return tuple(rewrite(item) for item in x)
    elif hasattr(x, '_type') and hasattr(x, '__slots__'):
        # A ROS message type
        module_name, class_name = x._type.split('/')
        y = obj_from_str(module_name + '.msg', class_name)

        assert x.__slots__ == y.__slots__

        # Recursively rewrite fields
        for slot in x.__slots__:
            setattr(y, slot, rewrite(getattr(x, slot)))

        return y
    else:
        raise NotImplementedError("Type '{}' not handled".format(type(x)))




output_root = '/data2/itri/DCV/single_sensor_data'
bag_file = '/data2/itri/DCV/201007_original_data/lidars_1.bag'
bag = rosbag.Bag(bag_file, "r")
# bag_data = bag.read_messages()
# bag_data = bag.read_messages(topics=['/Navtech/Polar', '/usb_cam/image_raw/compressed', 
#                                      '/livox/lidar', '/baraja_lidar/sensorhead_1'])
bag_data = bag.read_messages(topics=['/tf_static'])

for topic, msg, t in tqdm(bag_data):
    import ipdb; ipdb.set_trace(context=7)
    
    if rospy.is_shutdown():
        break

    timestamp = str(t.secs) + "{:09d}".format(t.nsecs)[:4]

    if topic == "/tf_static":
        msg = rewrite(msg)
        with open('{}/{}.pkl'.format("{}/tf".format(output_root), timestamp), 'wb') as f:
            pickle.dump(msg, f, pickle.HIGHEST_PROTOCOL)