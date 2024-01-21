import json
import shutil
import os

# Source and destination folders

destination_folder = "/home/ee904/Repo/yolov8/datasets/"

# Create destination folders if not exist
image_destination_folder = os.path.join(destination_folder, "images")
annotation_destination_folder = os.path.join(destination_folder, "labels")
os.makedirs(image_destination_folder, exist_ok=True)
os.makedirs(annotation_destination_folder, exist_ok=True)

# Read training.json
json_file_path = "/home/ee904/Repo/My_CenterNet/radiate_sdk/vehicle_detection/training.json"  # Replace with the actual path
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

counter = 0

# Copy images and generate YOLO annotations
for item in data:
    # Copy image
    image_path = item["file_name"]
    image_name = str(counter) + ".png"
    image_destination_path = os.path.join(image_destination_folder, image_name)
    # shutil.copy(image_path, image_destination_path)

    # Generate YOLO annotation
    annotation_name = os.path.splitext(image_name)[0] + ".txt"
    annotation_path = os.path.join(annotation_destination_folder, annotation_name)

    counter += 1
    
    with open(annotation_path, "w") as annotation_file:
        for annotation in item["annotations"]:
            bbox = annotation["bbox"]
            # Calculate center coordinates and normalized width/height
            x_center = (bbox[0] + bbox[2]) / 2 / item["width"]
            y_center = (bbox[1] + bbox[3]) / 2 / item["height"]
            width = abs(bbox[2] - bbox[0]) / item["width"]
            height = abs(bbox[3] - bbox[1]) / item["height"]
            
            # Write to annotation file
            annotation_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Image and annotation files generated successfully.")
