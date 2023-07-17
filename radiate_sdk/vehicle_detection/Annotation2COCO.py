# $python3 Annotation2COCO.py --dataset_name city_1_3

# import some common libraries
import numpy as np
import cv2
import random
import os
import sys
import time

# radiate sdk
import argparse
import json
sys.path.insert(0, '..')
import radiate
from detectron2.config import get_cfg
from detectron2.structures import BoxMode

# init params
parser = argparse.ArgumentParser()

parser.add_argument("--root_folder", help="root folder with radiate dataset",
                    default='/data/RADIATE',
                    type=str)

parser.add_argument("--resume", help="Whether to resume training or not",
                    default=False,
                    type=bool)

parser.add_argument("--dataset_mode", help="dataset mode ('good_weather', 'good_and_bad_weather')",
                    default='city',
                    type=str)
parser.add_argument("--dataset_name", help="using which radiate dataset",
                    default='city_1_1',
                    type=str)

# parse arguments
args = parser.parse_args()
root_dir = args.root_folder
resume = args.resume
dataset_mode = args.dataset_mode
dataset_name = args.dataset_name


def __timestamp_format(raw_timestamp):
    """
    function to fix the timestamp
    """
    raw_decimal_place_len = len(raw_timestamp.split('.')[-1])
    if(raw_decimal_place_len < 9):
        place_diff = 9 - raw_decimal_place_len
        zero_str = ''
        for _ in range(place_diff):
            zero_str = zero_str + '0'
        formatted_timestamp = raw_timestamp.split(
            '.')[0] + '.' + zero_str + raw_timestamp.split('.')[1]
        return float(formatted_timestamp)
    else:
        return float(raw_timestamp)

def load_timestamp(timestamp_path):
    """load all timestamps from a sensor

    :param timestamp_path: path to text file with all timestamps
    :type timestamp_path: string
    :return: list of all timestamps
    :rtype: dict
    """
    with open(timestamp_path, "r") as file:
        lines = file.readlines()
        timestamps = {'frame': [], 'time': []}
        for line in lines:
            words = line.split()
            timestamps['frame'].append(int(words[1]))
            timestamps['time'].append(__timestamp_format(words[3]))
    return timestamps
cartesian_timestamp = load_timestamp('/data/RADIATE/'+str(dataset_name)+'/Navtech_Cartesian.txt')
print(type(cartesian_timestamp)) # dict
time_list = cartesian_timestamp['time']

def gen_boundingbox(bbox, angle):
        theta = np.deg2rad(-angle) #????????????????
        # R is rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]], # x + width
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]], # x + width, y + height
                           [bbox[0], bbox[1] + bbox[3]]]).T
        # find center(x, y)
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T # move to origin point
        # matrix multiplication
        points = np.matmul(R, points) + T # -> rotation and translation
        points = points.astype(int)
        # each point in points is (x, y)
        min_x = int(np.min(points[0, :])) # 0 means x coordinate
        min_y = int(np.min(points[1, :])) # 1 means y coordinate
        max_x = int(np.max(points[0, :]))
        max_y = int(np.max(points[1, :]))

        return min_x, min_y, max_x, max_y

def get_radar_dicts(folders):
        # dataset_dicts = []
        dataset_dicts = {}
        class_dict = {'car':1,'van':2,'truck':3, 'bus':4, 'motorbike':5, 
                    'bicycle':6, 'pedestrian':7, 'group of pedestrian':8}
        class_list = ['car', 'van', 'truck', 'bus', 'motorbike', 'bicycle', 'pedestrian', 'group of pedestrian']
        idd = 0
        folder_size = len(folders)
        for folder in folders:
            radar_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')  # get radar image
            annotation_path = os.path.join(root_dir,
                                           folder, 'annotations', 'annotations.json') # find annotations.json
            with open(annotation_path, 'r') as f_annotation:
                annotation = json.load(f_annotation) # very very long words

            radar_files = os.listdir(radar_folder)
            radar_files.sort()
###################################################################################################################
            cats = []
            cat_created = False
            for class_id in range(len(class_dict)):
                if(class_id != 0 or class_id != 7 or class_id != 8):
                    supercat = "car"
                else: supercat = "none"
                cat = { 
                        "id": class_id,
                        "name": class_list[class_id],
                        "supercategory": supercat
                }
                cats.append(cat)
            cat_created = True
            if cat_created:
                dataset_dicts["categories"] = cats
###################################################################################################################
            imgs = []
            print(len(radar_files))
            for img_number in range(len(radar_files)):
                time_local = time.localtime(time_list[img_number])
                dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
                img = { 
                        "id": img_number,
                        "license": 1,
                        "file_name": os.path.splitext( radar_files[img_number] )[0]+ ".jpg",
                        "height": 1152,
                        "width": 1152,
                        "date_captured": dt
                }
                imgs.append(img)

            dataset_dicts["images"] = imgs
###################################################################################################################
            image_count = 0
            objs = []
            bb_created = False
            for frame_number in range(len(radar_files)):                
                
                idd += 1
                filename = os.path.join(
                    radar_folder, radar_files[frame_number])

                if (not os.path.isfile(filename)):
                    print(filename)
                    continue
                
                idd += 1
                for object in annotation:
                    if (object['bboxes'][frame_number]):
                        class_obj = object['class_name']
                        id_obj = int(object['id'])
                        if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                            bbox = object['bboxes'][frame_number]['position']
                            angle = object['bboxes'][frame_number]['rotation']
                            bb_created = True
                            wid = bbox[2]
                            hei = bbox[3]
                            area = wid * hei
                            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "RRPN":
                                cx = bbox[0] + bbox[2] / 2
                                cy = bbox[1] + bbox[3] / 2
                                
                                obj = { 
                                    "id": id_obj,
                                    # "bbox_mode": BoxMode.XYWHA_ABS,  # In detectron2, bbox_mode (int, required)
                                    "image_id": frame_number,
                                    "category_id": int(class_dict[class_obj] - 1),
                                    "bbox": [cx, cy, wid, hei, angle],
                                    "area": area,
                                    "segmentation": [],
                                    "iscrowd": 0
                                }
                            else:
                                xmin, ymin, xmax, ymax = gen_boundingbox(
                                    bbox, angle)
                                obj = {
                                    "id": id_obj,
                                    # "bbox_mode": BoxMode.XYXY_ABS,
                                    "image_id": frame_number,
                                    "category_id": int(class_dict[class_obj] - 1),
                                    "bbox": [xmin, ymin, xmax, ymax],
                                    "area": area,
                                    "segmentation": [],
                                    "iscrowd": 0
                                }
                            
                            objs.append(obj)  # belongs to "annotations"
                
            if bb_created:
                dataset_dicts["annotations"] = objs
        return dataset_dicts
def get_category_dicts(folders):
    category_dicts = []
cfg = get_cfg()
cfg.OUTPUT_DIR = root_dir # change later
# data_folder = ['city_1_0', 'city_1_3']
data_folder = [str(dataset_name)]
Rdicts = get_radar_dicts(data_folder) # this is a list
print(type(Rdicts))
print(len(Rdicts)) # 424 images or (city_1_0 714 images)?  (tiny_foggy 18 images)

with open('COCO.json', 'w') as f:
    json.dump(Rdicts, f)