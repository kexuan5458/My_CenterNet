'''
The annotation is saved as a .json file, where each entry of a list contains id, class_name, bboxes. 
id is the object identification. 
class_name is a string with the class name. 
bboxes contains position: (x, y, width, height) where (x, y) is the upper-left pixel locations of the bounding box of the given width and height. 
And rotation is the angle in degrees using counter-clockwise.
'''

import json
import os
import sys
import numpy as np
custom_annotations = '/data/RADIATE/fog_6_0/annotations/annotations.json'
myDict = {}

array = []  # list
array_processed = []  # list
images = []
categories = []
ann = []

with open('/data/RADIATE/fog_6_0/Navtech_Cartesian.txt', 'r') as f:
    for line in f.readlines():
        array.append(line.replace('\n', ' ').split(' '))


for i in range(len(array)):
    timestamp = '{:.4f}'.format(float(array[i][3])).replace('.', '')
    array_processed.append([(array[i][1]), timestamp])
array2D = np.asarray(array_processed)  # <class 'numpy.ndarray'>
arr_timestamp = array2D[:,1] # column 1 -> timestamp (str)
for img_idx in range(len(arr_timestamp)):
  img_name = arr_timestamp[img_idx] + '.png'
  images.append(
      { 
        "id": img_idx,
        "file_name": img_name,
        "height": 1152,
        "width": 1152
      })
# Adding list as dictionary value
myDict["images"] = images


# Define categories
category = ['car', 'van', 'bus', 'truck', 'motorbike', 'bicycle', 'pedestrian']
for cat_idx in range(len(category)):
    categories.append(
        {
        "id": cat_idx,
        "name": category[cat_idx]
        })
# Adding list as dictionary value
myDict["categories"] = categories
catDic ={
    "car": 0,
    "van": 1,
    "bus": 2,
    "truck": 3,
    "motorbike": 4,
    "bicycle": 5,
    "pedestrian": 6}


# Open and read the JSON file
with open(custom_annotations, 'r') as file:
    data = json.load(file)   # data is a list
# print((data[0])) # data[0] is a dictionary

for data_idx in range(len(data)):

    # Object
    # Here, object means sth like a car, a van, ...pedestrian.
    objID = data[data_idx]["id"]
    objClsName = data[data_idx]["class_name"]
    objBbox = data[data_idx]["bboxes"]
    print(objID)    # 1   int
    print(objClsName)   # bus  str
    # print(type(objBbox))    # list

    '''
    for key in data[0].keys():
        print(key)
    id
    class_name
    bboxes
    '''
    idx = 0 # image ID of one object. 
    for item in objBbox:
        # 檢查是否存在 'position' key
        if 'position' in item:
        # # 檢查是否存在 'rotation' key
        # if 'rotation' in item:
            position = item['position']     # position is a list [x, y, width, height]
            # print(f"Position: {position}") # (x, y) is the upper-left pixel
            rotation = item['rotation']    # rotation is a float
            # print(f"Rotation: {rotation}")
            img_name = arr_timestamp[idx] + '.png'
            # print(f"Image name: {img_name}")
        
            ann.append(
            { 
                "id": data_idx, # object id
                "image_id": idx,
                "category_id": catDic[objClsName],
                "bbox": position, 
                "angle": rotation,
                "area": (position[2] * position[3])
            })

        # 輸出空行，以分隔不同的字典元素
        print()
        idx += 1

myDict["annotations"] = ann

'''
# Initialize the COCO JSON skeleton
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Process your custom annotations
for image_data in custom_annotations:
    image_info = {
        "id": image_data["id"],
        "file_name": "path/to/your/image.jpg",  # Replace with the actual image path
        "width": 640,  # Replace with actual image width
        "height": 480,  # Replace with actual image height
        "annotations": [annotation_id for annotation_id in image_data["bboxes"]]
    }
    coco_data["images"].append(image_info)

    for bbox_data in image_data["bboxes"]:
        annotation = {
            "id": bbox_data["id"],
            "image_id": image_data["id"],
            "category_id": 1,  # Replace with the appropriate category ID
            "bbox": bbox_data["position"],
            "area": bbox_data["position"][2] * bbox_data["position"][3],
            "iscrowd": 0  # 0 for non-crowd objects
        }
        coco_data["annotations"].append(annotation)

# Define categories
category = {
    "id": 1,
    "name": "van"  # Replace with the actual class name
}


# Save the COCO JSON to a file
with open("coco_annotations.json", "w") as f:
    json.dump(coco_data, f)

'''
# Save the COCO JSON to a file
with open("coco_annotations.json", "w") as outfile:
    json.dump(myDict, outfile)