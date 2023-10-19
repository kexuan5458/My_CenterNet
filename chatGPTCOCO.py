'''
# Initialize the COCO JSON skeleton
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Process your custom annotations
for image_data in custom_annotations:
    image_info = {
        "id": image_data["id"],
        "file_name": "path/to/your/image.jpg",  # Replace with the actual image path
        "width": 640,  # Replace with actual image width
        "height": 480,  # Replace with actual image height
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
category = {
    "id": 1,
    "name": "van"  # Replace with the actual class name
}


# Save the COCO JSON to a file
with open("coco_annotations.json", "w") as f:
    json.dump(coco_data, f)

'''
