import cv2
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon


class Box:
    def __init__(self, **kwargs):
        try:
            self.bbox_coords = Polygon(kwargs["points"])
        except:
            print(kwargs)
        assert len(kwargs["points"]) == 4 or kwargs["points"].shape == (4, 2)
        self.area = self.bbox_coords.area
        self.name = kwargs["name"]
        self.sample_token = kwargs["sample_token"]

    def get_iou(self, other):
        intersection = self.get_intersection(other)
        union = self.area + other.area - intersection

        iou = np.clip(intersection / union, 0, 1)

        return iou

    def get_intersection(self, other) -> float:
        area_intersection = self.bbox_coords.intersection(other.bbox_coords).area

        return area_intersection

    def __repr__(self):
        return str(self.serialize())

    def serialize(self) -> dict:
        """Returns: Serialized instance as dict."""

        return {
            "points": self.bbox_coords,
            "area": self.area,
            "name": self.name,
        }



def group_by_key(detections, key):
    groups = defaultdict(list)
    for detection in detections:
        groups[detection[key]].append(detection)
    return groups


def get_envelope(precisions):
    """Compute the precision envelope.

    Args:
      precisions:

    Returns:

    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls, precisions):
    """Calculate average precision.

    Args:
      recalls:
      precisions: Returns (float): average precision.

    Returns:

    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def get_ious(gt_boxes, predicted_box):
    return [predicted_box.get_iou(x) for x in gt_boxes]


def wrap_in_box(input):
    result = {}
    for key, value in input.items():
        result[key] = [Box(**x) for x in value]

    return result
def recall_precision(gt, predictions, iou_threshold):
    # , range, image_center, metric):
    # gt  = filter_boxes(gt, image_center, range)
    num_gts = len(gt)
    image_gts = group_by_key(gt, "sample_token")
    image_gts = wrap_in_box(image_gts)
    
    # image_gts = wrap_in_box(image_gts)

    sample_gt_checked = {
        sample_token: np.zeros(len(boxes)) for sample_token, boxes in image_gts.items()
    }
    
    
    # predictions = sorted(predictions, key=lambda x: x["scores"], reverse=True)
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)
    avg_max_overlap = 0
    avg_max_overlap_cnt = 0
    for prediction_index, prediction in enumerate(predictions):
        
        predicted_box = Box(**prediction)


        sample_token = prediction["sample_token"]


        max_overlap = -np.inf
        min_overlap = np.inf
        jmax = -1

        try:
            gt_boxes = image_gts[sample_token]  # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]  # gt flags per sample
            
        except KeyError:
            gt_boxes = []
            gt_checked = None
        # if predicted_box.center_distance < range[1]:
        #     print(predicted_box.center_distance)
        # if predicted_box.center_distance < range[0] or predicted_box.center_distance > range[1]:
            # print(predicted_box.center_distance)
            # continue
        # print(predicted_box.center_distance)
        
        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box)
            max_overlap = np.max(overlaps)

            avg_max_overlap += max_overlap
            avg_max_overlap_cnt += 1

            jmax = np.argmax(overlaps)

        if max_overlap > iou_threshold:
            if gt_checked[jmax] == 0:
                tp[prediction_index] = 1.0
                gt_checked[jmax] = 1
            else:
                fp[prediction_index] = 1.0
        else:
            fp[prediction_index] = 1.0
        

    print(avg_max_overlap / (avg_max_overlap_cnt))
    tp_sum = np.sum(tp)
    print("tp_sum = ", tp_sum)
    print("gt_num = ", num_gts)
    print("fp_sum = ", np.sum(fp))
    print("fn_sum = ", num_gts - tp_sum )
    print("=========")
    recall = tp_sum / float(num_gts)
    precision = tp_sum / float(num_predictions)

    # return recall, precision

    # for detection model usage since we require the confidence score to calculate the average precision

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)
    
    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    assert np.all(0 <= precisions) & np.all(precisions <= 1)

    ap = get_ap(recalls, precisions)

    return recalls, precisions, ap


# def get_average_precisions(
#     gt: list, predictions: list, class_names: list, iou_threshold: float, range: int, image_center: np.array, metric: str
# ) -> np.array:
def get_average_precisions(
    gt: list, predictions: list, class_names: list, iou_threshold: float
) -> np.array:
    """Returns an array with an average precision per class.


    Args:
        gt: list of dictionaries in the format described below.
        predictions: list of dictionaries in the format described below.
        class_names: list of the class names.
        iou_threshold: IOU threshold used to calculate TP / FN

    Returns an array with an average precision per class.


    Ground truth and predictions should have schema:

    gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
    }]

    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]

    """
    # assert 0 <= iou_threshold <= 1

    gt_by_class_name = group_by_key(gt, "name")
    pred_by_class_name = group_by_key(predictions, "name")

    average_precisions = np.zeros(len(class_names))
    gt_valid = np.zeros((2))
    for class_id, class_name in enumerate(class_names):
        if class_name in gt_by_class_name:
            gt_valid[class_id] = 1


    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            # recalls, precisions, ap = recall_precision(
            #     gt_by_class_name[class_name],
            #     pred_by_class_name[class_name],
            #     iou_threshold,
            #     range,
            #     image_center,metric
            # )
            recalls, precisions, ap = recall_precision(
                gt_by_class_name[class_name],
                pred_by_class_name[class_name],
                iou_threshold,
            )
            average_precisions[class_id] = ap
            # average_recall[class_id] = recalls

    return average_precisions, gt_valid #average_recall,  gt_valid, ap


def get_class_names(gt: dict) -> list:
    """Get sorted list of class names.

    Args:
        gt:

    Returns: Sorted list of class names.

    """
    return sorted(list(set([x["name"] for x in gt])))

def eval(gt, predictions):
    gt_by_token = group_by_key(gt, "sample_token")
   
    class_names = get_class_names(gt)
    mAP = []    
    for iou in [ 0.3, 0.5, 0.65]:
        average_precisions, gt_valid= get_average_precisions(gt, predictions, class_names, iou)
        # , range, image_center, metric="iou")
        mAP.append(average_precisions)
