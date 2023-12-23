import numpy as np
from shapely import wkt
import os
import json
# from PIL import Image, ImageDraw
import glob
import cv2
from matplotlib import pyplot as plt


def parse_json(json_file, data):

    class_label_to_int = {
        'building-no-damage': 0,
        'building-unknown': 1,
        'building-un-classified': 2,
        'building-minor-damage': 3,
        'building-major-damage': 4,
        'building-destroyed': 5
        # Add more class labels and corresponding integer values as needed
    }

    annotations_dict = {}
    for entry in data['features']['xy']:
        properties = entry['properties']
        feature_type = properties['feature_type']
        # Provide a default value if 'subtype' is missing
        subtype = properties.get('subtype', 'unknown')
        uid = properties['uid']
        wkt_data = entry['wkt']

        class_label = f"{feature_type}-{subtype}"

        shape = wkt.loads(wkt_data)
        if shape.geom_type != "Polygon" or class_label not in class_label_to_int:
            continue

        # Extract the bounding box coordinates
        x_min = shape.bounds[0]
        y_min = shape.bounds[1]
        x_max = shape.bounds[2]
        y_max = shape.bounds[3]

        # Check if the bounding box is valid (width and height > 0)
        if x_min < x_max and y_min < y_max:
            # Extract the class label (e.g., 'building-no-damage')

            # Create a dictionary to store the annotation for this entry
            image_name = os.path.splitext(os.path.basename(json_file))[0]

            # Create a new entry for this image name if it does not exist
            if image_name not in annotations_dict:
                annotations_dict[image_name] = {
                    "boxes": [],
                    "labels": [],
                    "image_name": image_name
                }

            # Append the box and label to the existing entry for this image name
            annotations_dict[image_name]["boxes"].append(
                [x_min, y_min, x_max, y_max])
            annotations_dict[image_name]["labels"].append(
                class_label_to_int[class_label])

    return annotations_dict, image_name

def read_annotation(json_filename):
    image_name = os.path.splitext(os.path.basename(json_filename))[
        0]  # Get the image name without extension

    # Check if the JSON file should be processed based on the presence of 'wkt' key
    with open(json_filename, 'r') as f:
        data_parse = json.load(f)

        data_array = data_parse.get('features', {}).get('xy', [])

        for entry in data_array:
            if 'wkt' in entry:
                annotations = parse_json(json_filename, data_parse)
                return {image_name: annotations}

    return {}
def get_png_names(directory):
  """
  Returns a list of all PNG filenames in a directory.

  Args:
    directory: The path to the directory.

  Returns:
    A list of strings containing the filenames of all PNGs in the directory.
  """
  png_names = []
  for filename in os.listdir(directory):
    if filename.endswith(".png"):
      png_names.append(filename)
  return png_names

def create_masks(json_files_folder):
    class_colors = [
        # "building-no-damage" Green
        (0, 255, 0),
        (0, 255, 0),
        # "building-unknown":   # Orange
        (255, 128, 0),
        # "building-un-classified":   # Gray
        (128, 128, 128),
        # "building-minor-damage":   # Red
        (255, 0, 0),
        # "building-major-damage":   # Dark Orange
        (255, 165, 0),
        # "building-destroyed":
        (0, 0, 0),  # Black
    ]

    # Get the list of all JSON files in the folder
    json_files = glob.glob(os.path.join(json_files_folder, '*.json'))

    # Initialize an empty dictionary to store annotations for each image
    all_annotations = {}

    for json_file in json_files:
        data = read_annotation(json_file)
        all_annotations.update(data)

    og_masks = get_png_names("dataset/train/masks")

    for empty_mask in og_masks:
        mask_image = cv2.imread("dataset/train/masks/" + empty_mask)
        annotation_name = empty_mask[:-11]
        try:
            image_entry = all_annotations[annotation_name]
        except KeyError:
            print("Skipped image entry: " + empty_mask)
            cv2.imwrite("dataset/train/filled_masks/" + empty_mask, mask_image)
            continue

        boxes = image_entry[0][annotation_name]["boxes"]
        labels = image_entry[0][annotation_name]["labels"]
        for i, box in enumerate(boxes):
            [x_min, y_min, x_max, y_max] = [round(coordinate) for coordinate in box]
            polygon_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            if labels[i] < 4:
                polygon_color = (255, 0, 0)
            else:
                polygon_color = (0, 0, 255)

            cv2.fillPoly(mask_image, [polygon_points], polygon_color)


        cv2.imwrite("dataset/train/filled_masks/" + empty_mask, mask_image)

if __name__ == '__main__':
    create_masks("dataset/train/labels")
    print("created masks")