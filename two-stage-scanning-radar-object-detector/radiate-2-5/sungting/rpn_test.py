import torch
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import cv2
import numpy as np

bbb = torch.randint(100, 200, (5, 4), dtype=torch.float32)

mask = (bbb[:, 0] >= bbb[:, 2])
bbb[mask] = bbb[mask][:, [2, 1, 0, 3]]
mask = (bbb[:, 1] >= bbb[:, 3])
bbb[mask] = bbb[mask][:, [0, 3, 2, 1]]

def make_empty_sample(add_masks=False, add_keypoints=False):
	images = [torch.rand((3, 100, 100), dtype=torch.float32)]
	boxes = torch.zeros((0, 4), dtype=torch.float32)
	boxes = torch.randint(100, 200, (10, 4), dtype=torch.float32)
	negative_target = {
            # "boxes": boxes,
            "boxes": torch.cat([bbb, torch.tensor([[0, 1, 2, 3]])], dim=0),
						# "labels": torch.zeros(0, dtype=torch.int64),
						"labels": torch.randint(0, 2, (boxes.shape[0], ), dtype=torch.int64),
						"image_id": 4,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
						"iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

	if add_masks:
			negative_target["masks"] = torch.zeros(0, 100, 100, dtype=torch.uint8)

	if add_keypoints:
		negative_target["keypoints"] = torch.zeros(17, 0, 3, dtype=torch.float32)

	targets = [negative_target]
	return images, targets


_, targets = make_empty_sample()

anchors = [torch.cat([bbb, torch.tensor([[4, 5, 6, 7]])], dim=0)]

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
rpn_head = RPNHead(4, rpn_anchor_generator.num_anchors_per_location()[0])

head = RegionProposalNetwork(rpn_anchor_generator, rpn_head, 0.5, 0.3, 256, 0.5, 2000, 2000, 0.7)
head.proposal_matcher.allow_low_quality_matches = False


# labels: 1=matched, 0=under_matching_thres, -1=in_between
labels, matched_gt_boxes = head.assign_targets_to_anchors(anchors, targets)

from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage

img = np.zeros((512, 512, 3))
bb_list = BoundingBoxesOnImage([BoundingBox(*b) for b in np.array(anchors[0])], shape=img.shape)
img = bb_list.draw_on_image(image=img, size=3)
gt_list = BoundingBoxesOnImage([BoundingBox(*b) for b in np.array(targets[0]['boxes'])], shape=img.shape)
img = gt_list.draw_on_image(image=img, color=(255, 0, 0), size=3)
cv2.imwrite('anchor_bbox.png', img)
import ipdb; ipdb.set_trace(context=7)