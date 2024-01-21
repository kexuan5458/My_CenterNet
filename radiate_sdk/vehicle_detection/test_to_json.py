import os, sys, distutils
import glob
sys.path.insert(0, os.path.abspath('/home/ee904/Repo/detectron2'))

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import json


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate
cfg = get_cfg()


# path to the sequence
root_path = '/data/data/RADIATE'
sequence_name = 'junction_1_12' # just for example

network = 'faster_rcnn_R_50_FPN_3x' # just for example
# network = 'faster_rcnn_R_101_FPN_3x'
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.WEIGHTS = '/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection/weights/yenting_model_final_v2.pth'


setting = 'good_and_bad_weather_radar' # just for example

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='/home/ee904/Repo/My_CenterNet/radiate_sdk/config/config.yaml')
print(f"Initial timestamp : {seq.init_timestamp}, End timestamp : {seq.end_timestamp}")



# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

cfg.MODEL.DEVICE = 'cuda'

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

# Get Radar timestamp
radar_timestamp_file = os.path.join(root_path, sequence_name, "Navtech_Cartesian.txt")

timestamp = []
frame = []

with open(radar_timestamp_file, 'r') as file:

    lines = file.readlines()

    for line in lines:
        if "Frame" in line:
            frame_value = line.split(" ")[1]
            frame.append(frame_value)

        if "Time:" in line:
            time_value = float(line.split("Time: ")[1])
            timestamp.append(time_value)

for idx in frame:
    print(idx)


pred_name = "prd.json"
output_file_path = os.path.join('/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection', pred_name)

if os.path.exists(output_file_path):
    with open(output_file_path, 'w') as json_file:
        json_file.write('[')

num_entries = len(timestamp)
not_first = False

for idx, t  in enumerate(timestamp):

    output = seq.get_from_timestamp(float(t))
    sample_token = frame[idx]

    if output != {}:

        
        radar = output['sensors']['radar_cartesian']
        if (radar is None):
            continue

        camera = output['sensors']['camera_right_rect']

        predictions = predictor(radar)

        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes
        scores = predictions.scores 

        objects = []

        for box, score in zip(boxes, scores):
            
            x1, y1, x2, y2 = box
            bbox_position = np.array([min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)])
            bbox_rotation = 0

            x_min = min(x1, x2).item()
            x_max = max(x1, x2).item()
            y_min = min(y1, y2).item()
            y_max = max(y1, y2).item()
   
            if(score > 0):
                #print(f"score{score}")
                objects.append({'bbox': {'position': bbox_position, 'rotation': bbox_rotation}, 'class_name': 'vehicle'})

                points = [[x_min, y_max],
                          [x_min, y_min],
                          [x_max, y_min],
                          [x_max, y_max]]
                
                data_entry = {"sample_token": sample_token, "points": points, "name": "car", "score": score.item() }

                with open(output_file_path, 'a') as json_file:
                    if not_first == True:
                        json_file.write(',')
                    json.dump(data_entry, json_file, indent=2)
                    json_file.write('\n')
                    not_first = True
                           
    with open(output_file_path, 'a') as json_file:
        if idx == num_entries - 1:
            json_file.write(']')









