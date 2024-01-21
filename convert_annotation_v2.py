'''
The annotation is saved as a .json file, where each entry of a list contains id, class_name, bboxes. 
id is the object identification. 
class_name is a string with the class name. 
bboxes contains position: (x, y, width, height) where (x, y) is the upper-left pixel locations of the bounding box of the given width and height. 
And rotation is the angle in degrees using counter-clockwise.
# 改為以原radiate照片檔名製作成coco.json
'''

import json
import os
import sys
import numpy as np

# 資料夾的路徑
folder_root = '/data/data/RADIATE'
txt_path = os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian.txt')
folder_path = os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian/')
print(sys.argv[1])
custom_annotations = os.path.join(folder_root, sys.argv[1], 'annotations/annotations.json')

myDict = {}

array = []  # list
array_processed = []  # list
images = []
categories = []
ann = []

with open(txt_path, 'r') as f:
    for line in f.readlines():
        array.append(line.replace('\n', ' ').split(' '))


# Define categories
category = ['car', 'van', 'bus', 'truck', 'motorbike', 'bicycle', 'pedestrian', 'group_of_pedestrians']
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
    "pedestrian": 6,
    "group_of_pedestrians": 7}


for i in range(len(array)):
    timestamp = '{:.4f}'.format(float(array[i][3])).replace('.', '')
    array_processed.append([(array[i][1]), timestamp])
array2D = np.asarray(array_processed)  # <class 'numpy.ndarray'>
arr_timestamp = array2D[:,0] # column 1 -> timestamp (str)   # column 0 -> 000001, 000002 (str)
# 改為以原radiate照片檔名製作成coco.json
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
            crowd0_1 = 1 if (objClsName == "group_of_pedestrians") else 0
            ann.append(
            { 
                "id": data_idx, # object id
                "image_id": idx,
                "category_id": catDic[objClsName],
                "bbox": position, 
                "angle": rotation,
                "area": (position[2] * position[3]),
                "iscrowd": crowd0_1  # 0 for non-crowd objects
            })
        # 輸出空行，以分隔不同的字典元素  print()
        idx += 1
myDict["annotations"] = ann

# Save the COCO JSON to a file
outputfile = os.path.join(folder_root, sys.argv[1], (sys.argv[1]+'_coco_annotations.json'))
with open(outputfile, "w") as outfile:
    json.dump(myDict, outfile)