import cv2
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
import json

def group_by_key(detections, key):
    groups = defaultdict(list)
    for detection in detections:
        groups[detection[key]].append(detection)
    return groups

# gt_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/gt_junction_1_12.json'
# pred_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/prd_ITRI_junction_1_12.json'
gt_file = '/home/ee904/ITRI/ITRI_final/gt_itri.json'
# pred_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/prd_ITRI_junction_1_12.json'
# gt_file = '/home/ee904/Repo/My_CenterNet/radiate_sdk/gt_city_7_0.json'
pred_file = '/home/ee904/ITRI/ITRI_final/gt_itri.json'
gt = []
predictions = []

with open(pred_file) as f:
    predictions = json.load(f)
with open(gt_file) as f:
    gt = json.load(f)
grouped_gt = group_by_key(gt, "sample_token")
grouped_predictions = group_by_key(predictions, "sample_token")

from copy import copy
cv2.namedWindow('I', cv2.WINDOW_NORMAL)
colors = {'car': (255, 0, 255), 'pedestrian': (0, 0, 255), 'scooter': (50, 255, 0)}
# Iterate through each sample_token
sample_token_count = 0
for sample_token in grouped_gt:
    print("sample_token_count = ", sample_token_count)
    # Create a blank image (or load the actual image corresponding to the sample_token)
    # image = cv2.imread(f"/data/data/RADIATE/city_7_0/Navtech_Cartesian/" + sample_token + ".png", 1)
    image = cv2.imread(f"/home/ee904/ITRI/ITRI_final/radar/" + sample_token + ".png", 1)
    canvas = np.zeros((1600, 1600, 3), dtype=np.uint8)  # Change dimensions as needed
    image = cv2.addWeighted(image, 0.8, image, 0, 0)
    gt_image = copy(image)
    predict_image = copy(image)
    sample_token_count += 1
    
    # Draw ground truth boxes in white
    # Draw prediction boxes in magenta
    for box in grouped_predictions.get(sample_token, []):
        points = np.array(box["points"], dtype=np.int32)
        if box['name'] == 'car':
            cv2.polylines(canvas, [points], isClosed=True, color=colors[box['name']], thickness=2)  # pred bbox
            # cv2.polylines(image, [points], isClosed=True, color=colors[box['name']], thickness=2)
            cv2.polylines(predict_image, [points], isClosed=True, color=colors[box['name']], thickness=2)   # pred bbox+image
    for box in grouped_gt.get(sample_token, []):
        points = np.array(box["points"], dtype=np.int32)
        if box['name'] == 'car':
            cv2.polylines(image, [points], isClosed=True, color=(255, 255, 255), thickness=2)   # gt bbox+image
            # cv2.polylines(gt_image, [points], isClosed=True, color=(255, 255, 255), thickness=2)    # gt bbox+image

    image = cv2.addWeighted(canvas, 0.8, image, 0.6, 0)
    # Display the image
    cv2.imshow(f'Image', image)
    cv2.imshow(f'Image', image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break