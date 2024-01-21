import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import math
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

# path to the sequence
# root_path = '/data/data/RADIATE/ray_testdata'
root_path = '/data/data/RADIATE'
sequence_name = 'city_7_0'
#sequence_name = ['city_1_3', 'city_6'] # just for example

network = 'faster_rcnn_R_50_FPN_3x' # just for example
setting = 'good_and_bad_weather_radar' # just for example

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='/home/ee904/Repo/My_CenterNet/radiate_sdk/config/config.yaml')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.DEVICE = 'cuda'
# cfg.MODEL.WEIGHTS = '/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection/train_results/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/model_final.pth'
cfg.MODEL.WEIGHTS = '/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection/train_results/faster_rcnn_R_50_FPN_3x_good_weather/model_final.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

sample_token_counter = 0

predict_result_array = []

def is_similar_box(box1, box2):
    # box1 and box2 are both 4x2 array
    # box1 and box2 are similar if the distance between their center is less than 1m

    [x1, y1] = box1[0]
    [x2, y2] = box2[0]
    [x3, y3] = box1[2]
    [x4, y4] = box2[2]

    if abs(x1 - x2) < 40 and abs(y1 - y2) < 40 and abs(x3 - x4) < 40 and abs(y3 - y4) < 40:
        return True
    else:
        return False

# def filter_valid_box_array(valid_box_array):
#     # Remove similar boxes
#     print("len(valid_box_array) = ", len(valid_box_array))

#     for i in range(len(valid_box_array)):
#         for j in range(i + 1, len(valid_box_array)):
#             if i == j:
#                 continue
#             array_i = valid_box_array[i]
#             print(j)
#             array_j = valid_box_array[j]
#             if is_similar_box(array_i, array_j):
#                 valid_box_array.pop(j)
    
#     return valid_box_array

for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    if output != {}:
        radar = output['sensors']['radar_cartesian']
        camera = output['sensors']['camera_right_rect']

        predictions = predictor(radar)
        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes 
        scores = predictions.scores

        sample_token = str(output['sensors']['id'])
        # Extend the length of sample_token to 6, padding 0
        while len(sample_token) < 6:
            sample_token = '0' + sample_token

        print("sample_token = ", sample_token)

        objects = []

        valid_box_array = []
        score_array = []

        for box, score in zip(boxes, scores):
            # if score < 0.85:
            #     continue

            x1, y1, x2, y2 = box
            #bbox_position = {'left': x1, 'top': y1, 'right': x2, 'bottom': y2}
            #bbox_position = np.array([x1, y1,x2, y2])
            bbox_position = np.array([min(x1, x2), min(y1, y2), abs(x1-x2), abs(y1-y2) ])
            bbox_rotation = 0  # 在2D情况下通常没有旋转

            # print("box = ", [x1.item(), y1.item(), x2.item(), y2.item()], "score = ", score.item())

            if score.item() > 0.8:
                score_array.append(score.item())
                box = [[x1.item(), y1.item()], [x1.item(), y2.item()], [x2.item(), y2.item()], [x2.item(), y1.item()]]
                print("box = ", box, "score = ", score.item())
                has_similar_box = False
                for valid_box in valid_box_array:
                    if is_similar_box(box, valid_box):
                        has_similar_box = True
                        break
                if not has_similar_box:
                    valid_box_array.append(box)

            ### Student implement ###
            # TODO
            
            
            
            objects.append({'bbox': {'position': bbox_position, 'rotation': 0}, 'class_name': 'vehicle'})

        if sample_token != "000001":
            # valid_box_array = filter_valid_box_array(valid_box_array)
            for valid_box, score in zip(valid_box_array, score_array):
                predict_result_array.append({
                    "sample_token": sample_token,
                    "points": valid_box,
                    "name": "car",
                    "scores": score
                })

        print("")
        radar = seq.vis(radar, objects, color=(255,0,0))

        bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                seq.calib.right_cam_mat,
                                                seq.calib.RadarToRight)
        # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
        camera = seq.vis_bbox_cam(camera, bboxes_cam)

        cv2.imshow('radar', radar)
        cv2.imshow('camera_right_rect', camera)
        # You can also add other sensors to visualize
        cv2.waitKey(1)

with open("/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection/output0433.json", 'w') as json_file:
    json.dump(predict_result_array, json_file)

