import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import sys

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate__prd

# path to the sequence
root_path = '/data/data/RADIATE/'
sequence_name = 'junction_1_12'
# sequence_name = 'city_3_7'
# sequence_name = 'city_7_0'

file_path = os.path.join(root_path, sequence_name, 'Navtech_Cartesian.txt')

# timestamp 4th column 的 list
fourth_column_list = []
with open(file_path, 'r') as file:
    for line in file:
        columns = line.split()
        if len(columns) >= 4:
            fourth_column_list.append(columns[3])

network = 'faster_rcnn_R_50_FPN_3x'
# network = 'faster_rcnn_R_101_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate__prd.Sequence(os.path.join(root_path, sequence_name), config_file='../config/config.yaml')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.DEVICE = 'cuda'
# cfg.MODEL.WEIGHTS = os.path.join('weights',  network +'_' + setting + '.pth') 

cfg.MODEL.WEIGHTS = '/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection/weights/faster_rcnn_R_50_FPN_3x_good_weather_ITRIown.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

frame_count = 0
# for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
for t in (fourth_column_list):
    t = float(t)
    frame_count += 1
    print("frame_count = ", frame_count)
    
    output = seq.get_from_timestamp(t)
    if output != {}:

        radar = output['sensors']['radar_cartesian']
        # camera = output['sensors']['camera_right_rect']
        predictions = predictor(radar)
        

        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes 
        scores = predictions.scores
        numpy_scores = scores.numpy()
        # print(type(numpy_scores))
        # print(numpy_scores)

        objects = []

        for box in boxes:
            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RRPN':
                bb, angle = box.numpy()[:4], box.numpy()[4]        
            else:
                bb, angle = box.numpy(), 0   
                bb[2] = bb[2] - bb[0]
                bb[3] = bb[3] - bb[1]
            objects.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'car'})
        # self.vis(output['sensors']['radar_cartesian'], output['annotations']['radar_cartesian'], color=None, mode='rot', frame_count=fc)
        radar = seq.vis(radar, objects, numpy_scores, frame_count, color=(255,0,0), mode='rot')


        # bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                # seq.calib.right_cam_mat,
                                                # seq.calib.RadarToRight)
        # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
        # camera = seq.vis_bbox_cam(camera, bboxes_cam)

        cv2.imshow('radar', radar)
        # cv2.imshow('camera_right_rect', camera)
        cv2.waitKey(10)
    else:
        print("output is empty")
