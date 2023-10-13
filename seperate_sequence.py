'''
Written by Chen-Yi Peng

Iterate through all folders under root_dir, then, find out 'meta.json' file.
According to the "set" label, classify the folder by its set type.
There are 3 types: train_good_weather, train_good_and_bad_weather, and test.
Not only record type of each folder's data, but also calculate the number of images of each type.
'''
import os 
import json
import pathlib
from os.path import isdir
# import torch
# from torch import einsum
# torch.einsum('ij,jk->ik', A, B) # matrix multiplication

def split_sequence(root_dir, dataset_mode):

    good_count = 0
    good_and_bad_count = 0
    test_count = 0

    # get folders depending on dataset_mode
    # folders_train = []
    folders_train_good = []
    folders_train_good_and_bad = []
    folders_test = []
    for curr_dir in os.listdir(root_dir):   # iterate through all folders under root_dir
        
        if (curr_dir[-3:] != "zip"):
            with open(os.path.join(root_dir, curr_dir, 'meta.json')) as f:
                meta = json.load(f)
            if meta["set"] == "train_good_weather":
                if isdir(os.path.join(root_dir, curr_dir,'annotations')):
                    folders_train_good.append(curr_dir)
                for path in pathlib.Path(os.path.join(root_dir, curr_dir, 'Navtech_Cartesian')).iterdir():
                    if path.is_file():
                        good_count += 1

            elif meta["set"] == "train_good_and_bad_weather" and dataset_mode == "good_and_bad_weather":
                if isdir(os.path.join(root_dir, curr_dir,'annotations')):
                    folders_train_good_and_bad.append(curr_dir)
                for path in pathlib.Path(os.path.join(root_dir, curr_dir, 'Navtech_Cartesian')).iterdir():
                    if path.is_file():
                        good_and_bad_count += 1
            
            elif meta["set"] == "test":
                if isdir(os.path.join(root_dir, curr_dir,'annotations')):
                    folders_test.append(curr_dir)
                for path in pathlib.Path(os.path.join(root_dir, curr_dir, 'Navtech_Cartesian')).iterdir():
                    if path.is_file():
                        test_count += 1
            
    file = open('items.txt', 'w')
    file.write('train_good list:' + '\n')
    for item in folders_train_good:
        file.write(item + '\n')
    file.write('\n' + 'train_good_and_bad list:' + '\n')
    for item in folders_train_good_and_bad:
        file.write(item + '\n')
    file.write('\n' + 'test list:' + '\n')
    for item in folders_test:
        file.write(item + '\n')
    
    
    print(folders_train_good)
    print(folders_train_good_and_bad)
    print(folders_test)

if __name__ == "__main__":
    split_sequence("/data/RADIATE", "good_and_bad_weather")   
