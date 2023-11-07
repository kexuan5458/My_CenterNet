import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

bbb = torch.randint(100, 200, (5, 4), dtype=torch.float32)
mask = (bbb[:, 0] >= bbb[:, 2])
bbb[mask] = bbb[mask][:, [2, 1, 0, 3]]
mask = (bbb[:, 1] >= bbb[:, 3])
bbb[mask] = bbb[mask][:, [0, 3, 2, 1]]


images = [torch.rand((3, 100, 100), dtype=torch.float32)]
boxes = torch.cat([bbb, torch.tensor([[0, 1, 2, 3]])], dim=0)
target = [{
	# "boxes": boxes,
	"boxes": boxes,
	# "labels": torch.zeros(0, dtype=torch.int64),
	"labels": torch.randint(0, 2, (boxes.shape[0],), dtype=torch.int64),
	"image_id": 4,
	"area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
	"iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
}]


proposals = [torch.cat([bbb, torch.tensor([[0, 1, 2, 3]])], dim=0)]
# gt_boxes = [torch.cat([bbb, torch.tensor([[4, 5, 6, 7]])], dim=0)]
# gt_labels = [torch.ones((6,), dtype=torch.int64)]
gt_boxes = [bbb]
gt_labels = [torch.ones((5,), dtype=torch.int64)]

box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

resolution = box_roi_pool.output_size[0]
representation_size = 1024
box_head = TwoMLPHead(4 * resolution ** 2, representation_size)

representation_size = 1024
box_predictor = FastRCNNPredictor(representation_size, 2)

roi_heads = RoIHeads(
		# Box
		box_roi_pool,
		box_head,
		box_predictor,
		0.5,
		0.2,
		128,
		0.25,
		None,
		0.05,
		0.5,
		150,
)

# labels: show roi hits or not
# matched_idx: show roi matches which gt idx
# sampled_inds: return positive, negative sample given fg&bg ratio
matched_idxs, labels = roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
# roi_heads.select_training_samples(proposals, target)
sampled_inds = roi_heads.subsample(labels)

import ipdb; ipdb.set_trace(context=7)
