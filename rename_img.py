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
    array_processed.append([array[i][1], timestamp])
    # timestamp = str(round(float(array[i][3]), 4))
    # timestamp = timestamp.split('.')    # split integer and decimal
    # array_processed.append([array[i][1], (timestamp[0]+timestamp[1])])
array2D = np.array(array_processed)    
# print(array2D) # <class 'numpy.ndarray'>
# print(array2D[:,1]) # column 1 -> timestamp
# print(array2D[:,0]) # column 0 -> image name(number)


# 遍歷資料夾中的檔案
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # 檢查每個檔案名稱是否在array2D中的第一列中
        print(filename)
        for row in array2D:
            # print(row)
            if filename == row[0]+'.png':
                # 構建新的檔案路徑，包括新的檔案名稱
                new_filename = os.path.join(root, row[1]+'.png')
                old_filename = os.path.join(root, filename)
                # 重新命名檔案
                os.rename(old_filename, new_filename)
