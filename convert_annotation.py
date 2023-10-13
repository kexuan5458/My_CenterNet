import json
import os
import sys
import numpy as np

folder_root = '/data/RADIATE'
out_json_path = os.path.join(folder_root, sys.argv[1], 'annotations')
json_anno_path = os.path.join(folder_root, sys.argv[1], 'annotations')
# Initialize the COCO JSON skeleton
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Process your custom annotations
for image_data in (os.path.join(json_anno_path, 'annotations.json')):
    image_info = {
        "id": image_data["id"],
        "file_name": "path/to/your/image.jpg",  # Replace with the actual image path
        "width": 1152,  # Replace with actual image width (1152 pixel?)
        "height": 1152,  # Replace with actual image height (1152 pixel?)
        "annotations": [annotation_id for annotation_id in image_data["bboxes"]]
    }
    coco_data["images"].append(image_info)
    for bbox_data in image_data["bboxes"]:
        annotation = {
            "id": bbox_data["id"],
            "image_id": image_data["id"],
            "category_id": 1,  # Replace with the appropriate category ID
            "bbox": bbox_data["position"],
            "area": bbox_data["position"][2] * bbox_data["position"][3],
            "iscrowd": 0  # 0 for non-crowd objects
        }
        coco_data["annotations"].append(annotation)
# Define categories
categories = [  # Replace with the actual class name
		{'id': 1, 'name': 'car'},
    {'id': 2, 'name': 'van'},
		{'id': 3, 'name': 'bus'},
		{'id': 4, 'name': 'truck'},
    {'id': 5, 'name': 'motorbike'},
		{'id': 6, 'name': 'bicycle'},
    {'id': 7, 'name': 'pedestrian'}
]  # 8 categories of road actors (a group of pedestrians ?)
coco_data["category"].append(categories)

# Save the COCO JSON to a file
with open("coco_annotations.json", "w") as f:
    json.dump(coco_data, f)





###############################################################
###############################################################
###############################################################
img_idx = 1
anno_idx = 1

annotations = []
images = []

for i, f in enumerate(list(map(lambda x: int(x[:-4]), s))):
  try:
    with open(os.path.join(json_anno_path, '{}.json'.format(f))) as fl:
      data = json.load(fl)
  except:
    continue

  
  images.append(
    {
      "file_name": "{}/{}.png".format(split_str[si], f),
      "id": img_idx
    })

  
  # print(data['shapes'])
  single_img_anno = [x['points'] for x in data['shapes']]
  # print(single_img_anno)

  single_img_cls = [x['label'] for x in data['shapes']]
  # print(single_img_cls)

  # if f == 16377251434601:
  # 	print(len(data['shapes']))
  # 	print(len(single_img_cls))
  # 	print(len(single_img_anno))
  # 	print(img_idx)

  for anno, cls in zip(single_img_anno, single_img_cls):
    x_lt = anno[0]
    y_lt = anno[1]
    w = anno[2]
    h = anno[3]
    ang = anno[4]


    annotations.append(
      {
        "id": anno_idx,
        "category_id": cat_cvt[cls] + 1,
        "image_id": img_idx,
        "bbox": [
          x_lt-1,
          y_lt-1,
          w+2,
          h+2,
          ang
        ],
        "conf": 1,
        "iscrowd": 0,
        "area": (w+2) * (h+2)
      })
    anno_idx += 1

  # if f == 16377251434601:
  # 	print(list(filter(lambda x: x['image_id'] == 8, annotations)))
  # 	print(len(list(filter(lambda x: x['image_id'] == 8, annotations))))

  img_idx += 1

out = dict(
  images=images,
  annotations=annotations,
  categories=categories
)

with open('coco.json', "w") as f:
	json.dump(out, f, indent = 2)
###############################################################
###############################################################
###############################################################