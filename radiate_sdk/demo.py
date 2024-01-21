import radiate
import numpy as np
import os

# path to the sequence
root_path = '/data/data/RADIATE/'
sequence_name = 'city_7_0'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name))

# play sequence
frame_count = 0
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    print("frame: ", frame_count)
    output = seq.get_from_timestamp(t)
    seq.vis_all(output, 0)
    frame_count += 1