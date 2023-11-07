import os
import pickle
from .tf_util import msg_to_se3
# from tf_util import msg_to_se3

path = '/data2/itri/DCV/single_sensor_data/tf'


all_tf = sorted(os.listdir(path))
tfs = {}

for _tf in all_tf:
	with open(os.path.join(path, _tf), 'rb') as f:
		data = pickle.load(f)

		for x in data.transforms:
			tfs.update(
				{"{}->{}".format(x.header.frame_id, x.child_frame_id): msg_to_se3(x.transform)}
			)

# import ipdb; ipdb.set_trace(context=7)
