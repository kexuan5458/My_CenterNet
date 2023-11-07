import os
import numpy as np
from tqdm import tqdm
import itertools

import sys
sys.path.insert(
    0, "/home/950154_customer/sungting/da_centernet_uda/centernet-uda-ang-two-stage-bimo")
from evaluation.coco_easy import Evaluator

npz_path = '/home/950154_customer/sungting/da_centernet_uda/centernet-uda-ang-two-stage-bimo/outputs/hw2itri_em_dla_itri_test/2022-06-25-17-24-51-11241153-livox_anno/npz_output'
npz_file = sorted(os.listdir(npz_path))


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# all exps <
all_exp_root_dir = {
    0: npz_path
}

all_exp = {
    0: npz_file
}

all_exp_text = {
    0: 'livox_anno vs radar_detection'
}
# all exps >


defaults = {
    'score_threshold': 0.0,
    "per_class": True}

num_classes = 3
cats = {1: {'id': 1, 'name': '"car"'},
        2: {'id': 2, 'name': '"truck/bus"'},
        3: {'id': 3, 'name': '"ped/bimo"'}}

cat_mapping = {v: i for i,
               v in enumerate(range(1, num_classes + 1))}
classes = {y: cats[x] if x in cats else ''
           for x, y in cat_mapping.items()}

# evaluator = Evaluator(**defaults)
# evaluator.classes = classes
# evaluator.num_workers = 6
# evaluator.use_rotated_boxes = True

all_result = {}
img_shape = (1142, 1142)
pixel2m = 0.175

nms_threshold = 0.1
conf_threshold = 0.25

for k, v in all_exp.items():
    # for ii, dist_logic in enumerate(logic_list):
    evaluator = Evaluator(**defaults)
    evaluator.classes = classes
    evaluator.num_workers = 20
    evaluator.use_rotated_boxes = True

    # for i, exp in enumerate(tqdm(v)):
    for i, exp in enumerate(tqdm(
        v[-200:]
    )):
        exp_arr = np.load(os.path.join(
            all_exp_root_dir[k],
            exp
        ))

        ## nms <
        # conf_selection = (exp_arr['pred_scores'] > conf_threshold)

        # pred_boxes = exp_arr['pred_boxes'][conf_selection]
        # pred_boxes_score = exp_arr['pred_scores'][conf_selection]
        # pred_boxes_class = exp_arr['pred_classes'][conf_selection]

        # selection_prep = np.hstack((
        #     (pred_boxes[:, 0]).reshape(-1, 1),
        #     (pred_boxes[:, 1]).reshape(-1, 1),
        #     (pred_boxes[:, 0]+pred_boxes[:, 2]).reshape(-1, 1),
        #     (pred_boxes[:, 1]+pred_boxes[:, 3]).reshape(-1, 1),
        #     pred_boxes_class.reshape(-1, 1)
        # ))

        # selection = nms(selection_prep, nms_threshold)

        # inp = {"pred_boxes": pred_boxes[selection][None, ...],
        #        "pred_classes": pred_boxes_class[selection][None, ...],
        #        "pred_scores": pred_boxes_score[selection][None, ...],
        #        "gt_boxes": exp_arr['gt_boxes'][None, ...],
        #        "gt_classes": exp_arr['gt_classes'][None, ...],
        #        "gt_ids": [i],
        #        "gt_areas": None,
        #        "image_shape": (3, 1152, 1152)}
        ## nms >

        inp = {"pred_boxes": exp_arr['pred_boxes'][None, ...],
               "pred_classes": exp_arr['pred_classes'][None, ...],
               "pred_scores": exp_arr['pred_scores'][None, ...],
               "gt_boxes": exp_arr['gt_boxes'][None, ...],
               "gt_classes": exp_arr['gt_classes'][None, ...],
               "gt_ids": [i],
               "gt_areas": None,
               "image_shape": (3, 1152, 1152)}

        evaluator.add_batch(**inp)

    result = evaluator.evaluate()
    all_result["[{}]]".format(all_exp_text[k])] = result

print('ipdb')
import ipdb; ipdb.set_trace(context=7)
