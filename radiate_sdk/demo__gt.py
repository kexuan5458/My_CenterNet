import radiate__gt
import numpy as np
import os

# path to the sequence
root_path = '/data/data/RADIATE/'
sequence_name = 'rain_4_0'
# sequence_name = 'city_7_0'
# sequence_name = 'junction_1_12'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate__gt.Sequence(os.path.join(root_path, sequence_name))

# play sequence
frame_count = 0
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    frame_count += 1
    print("frame_count = ", frame_count)
    output = seq.get_from_timestamp(t)
    seq.vis_all(output, 10, frame_count)
