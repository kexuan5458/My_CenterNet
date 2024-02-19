'''
Author: KeXuan Wang
Process RADIATE annotation format to YOLO format.
Then use ProcessJson.sh to batch process all the json files in the folder.
'''

import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import sys

folder_root = "/data/RADIATE/"
json_path = os.path.join(folder_root, sys.argv[1])
output_path = (os.path.join(folder_root, sys.argv[1]) + '/labels/')
print(output_path)

f = open(json_path + '/' + str(sys.argv[1])+'_coco_annotations.json')

data = json.load(f)
f.close()
file_names = []

def load_images_from_folder(folder):
  count = 0
  for filename in os.listdir(folder):
        source = os.path.join(folder, filename)
        # destination = f"{output_path}images/img{count}.jpg"

        # try:
        #     shutil.copy(source, destination)
        #     print("File copied successfully.")
        # # If source and destination are same
        # except shutil.SameFileError:
        #     print("Source and destination represents the same file.")

        file_names.append(filename)
        count += 1

def get_img_ann(image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None

def get_img(filename):
  for img in data['images']:
    if img['file_name'] == filename:
      return img 

load_images_from_folder(os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian'))
count = 0
for filename in file_names:
  # Extracting image 
  img = get_img(filename)
  img_id = img['id']
  img_w = img['width']
  img_h = img['height']

  # Get Annotations for this image
  img_ann = get_img_ann(img_id)

  if img_ann:
    # Opening file for current image
    filename = os.path.splitext(filename)[0]
    file_object = open(f"{output_path}{filename}.txt", "a")

    for ann in img_ann:
      current_category = ann['category_id'] # As yolo format labels start from 0 
      current_bbox = ann['bbox']
      x = current_bbox[0]
      y = current_bbox[1]
      w = current_bbox[2]
      h = current_bbox[3]
      
      # Finding midpoints
      x_centre = (x + (x+w))/2
      y_centre = (y + (y+h))/2
      
      # Normalization
      x_centre = x_centre / img_w
      y_centre = y_centre / img_h
      w = w / img_w
      h = h / img_h
      
      # Limiting upto fix number of decimal places
      x_centre = format(x_centre, '.6f')
      y_centre = format(y_centre, '.6f')
      w = format(w, '.6f')
      h = format(h, '.6f')
          
      # Writing current object 
      file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

    file_object.close()
    count += 1  # This should be outside the if img_ann block.