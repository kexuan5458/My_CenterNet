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
from pascal_voc_writer import Writer

# coco2voc
from pycocotools.coco import COCO
from pascal_voc_writer import Writer


# create pascal voc writer (image_path, width, height)
writer = Writer('path/to/img.jpg', 1152, 1152)

# add objects (class, xmin, ymin, xmax, ymax)
writer.addObject('truck', 1, 719, 630, 468)
writer.addObject('person', 40, 90, 100, 150)

# write to file
writer.save('path/to/img.xml')

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