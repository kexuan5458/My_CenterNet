'''
Done
'''
import os
import sys
import numpy as np

# 資料夾的路徑
folder_root = '/data/RADIATE'
txt_path = os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian.txt')
folder_path = os.path.join(folder_root, sys.argv[1], 'Navtech_Cartesian/')
print(sys.argv[1])

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
arr_name = array2D[:,0]     # column 0 -> image name(number) (str)
# arr_name = arr_name.astype('int')

# Using index of arr_timestamp as image ID
print(arr_timestamp)
# dictionary = np.stack((arr_timestamp, arr_name), axis=1)    # both column are str




