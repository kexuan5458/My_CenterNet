import radiate
import numpy as np
import os

# path to the sequence
root_path = '/data/RADIATE'
sequence_name = 'city_1_0'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
print(os.path.join(root_path, sequence_name))
seq = radiate.Sequence(os.path.join(root_path, sequence_name))

# play sequence
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    seq.vis_all(output, 0)