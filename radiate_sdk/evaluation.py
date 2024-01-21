import json
import numpy as np
import argparse
import mAP_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # arg("-p", "--pred_file", type=str, help="Path to the predictions file.", required=True)
    # arg("-g", "--gt_file", type=str, help="Path to the ground truth file.", required=True)
    arg("-t", "--iou_threshold", type=float, help="iou threshold", default=0.5)

    args = parser.parse_args()

    # gt_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/gt_junction_1_12.json'
    # pred_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/competition_pred_faster_rcnn_24999.json'
    gt_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/gt_itri.json'
    pred_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/eval_itri.json'
    gt = []
    predictions = []
    
    with open(pred_file) as f:
        predictions = json.load(f)
    with open(gt_file) as f:
        gt = json.load(f)
    

    class_names = mAP_evaluation.get_class_names(gt)
    print("Class_names = ", class_names)
    

    average_precisions, gt_valid = mAP_evaluation.get_average_precisions(gt, predictions, class_names, args.iou_threshold)
    print("ap length = " + str(len(average_precisions)))
    mAP = np.mean(average_precisions)
    print("Average per class mean average precision = ", mAP)

    if len(average_precisions) > 0:
        for class_id in sorted(list(zip(class_names, average_precisions.flatten().tolist()))):
            print(class_id)
