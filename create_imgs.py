import os
import sys
import numpy as np
import json

# 資料夾的路徑
folder_root = '/data/RADIATE'
txt_path = os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian.txt')
folder_path = os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian/')
print(sys.argv[1])

myDict = {}

array = []  # list
array_processed = []  # list

with open(txt_path, 'r') as f:
    for line in f.readlines():
        array.append(line.replace('\n', ' ').split(' '))


for i in range(len(array)):   
    timestamp = '{:.4f}'.format(float(array[i][3])).replace('.', '')
    array_processed.append([(array[i][1]), timestamp])
array2D = np.asarray(array_processed)    
# print(array2D) # <class 'numpy.ndarray'>
arr_timestamp = array2D[:,1] # column 1 -> timestamp (str)


images = []
for idx in range(len(arr_timestamp)):
  img_name = arr_timestamp[idx] + '.png'
  images.append(
      {
        "id": idx,
        "file_name": img_name,
        "height": 1152,
        "width": 1152
      })
# print(images)
 
# Adding list as value
myDict["images"] = images
with open("sample.json", "w") as outfile:
    json.dump(myDict, outfile)