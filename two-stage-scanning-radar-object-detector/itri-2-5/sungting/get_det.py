import numpy as np

down_ratio = 4
rotated_boxes = True

def get_detection(batch):
    mask = (batch["reg_mask"] == 1).squeeze()
    dets_gt = batch["gt_dets"]
    areas_gt = batch["gt_areas"]
    dets_gt[:, :4] *= down_ratio

    gt_boxes = []
    gt_clss = []
    gt_ids = []
    gt_areas = []
    gt_kps = []

    box_idx = 4
    cls_idx = 5

    if rotated_boxes:
        box_idx = 5
        cls_idx = 6

    for i in range(1):
        det_gt = dets_gt[mask]

        gt_boxes.append(det_gt[:, :box_idx])
        gt_clss.append(det_gt[:, cls_idx].astype(np.int32))
        gt_areas.append(areas_gt[mask])


    out = {
        'gt_boxes': gt_boxes,
        'gt_classes': gt_clss,
        'gt_areas': gt_areas
    }

    return out
