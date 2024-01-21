import radiate__gt
import numpy as np
import os

# path to the sequence
root_path = '/data/data/RADIATE/'
# sequence_name = 'rain_4_0'
sequence_name = 'city_7_0'
# sequence_name = 'junction_1_12'

# time (s) to retrieve next frame 
dt = 0.25

file_path = os.path.join(root_path, sequence_name, 'Navtech_Cartesian.txt')
# timestamp 4th column 的 list
fourth_column_list = []
with open(file_path, 'r') as file:
    for line in file:
        columns = line.split()
        if len(columns) >= 4:
            fourth_column_list.append(columns[3])

# load sequence
seq = radiate__gt.Sequence(os.path.join(root_path, sequence_name))

# play sequence
frame_count = 0
# for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
for t in (fourth_column_list):
    t = float(t)
    frame_count += 1
    print("frame_count = ", frame_count)
    output = seq.get_from_timestamp(t)
    seq.vis_all(output, 10, frame_count)
