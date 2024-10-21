import json
import os
import cv2
import numpy as np

# mask_dir = "/Users/terminus/Downloads/total_TE_labelmeMask"
# img_and_json_dir = "/Users/terminus/Downloads/total_TE"

mask_dir = "/Users/terminus/Downloads/inpainting_template_labelmeMask"
img_and_json_dir = "/Users/terminus/Downloads/inpainting_template"

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

img_list_file = os.listdir(img_and_json_dir)

for it in img_list_file:
    if it.find(".json") == -1:
        continue

    with open(os.path.join(img_and_json_dir, it), 'r') as f:
        data = f.read()

    json_data = json.loads(data)

    img_name = os.path.join(img_and_json_dir, json_data['imagePath'])
    mask_name = os.path.join(mask_dir, it[:-5] + '.jpg')

    # get the points
    points = json_data["shapes"][0]["points"]
    points = np.array(points, dtype=np.int32)  # tips: points location must be int32

    # read image to get shape
    if not os.path.exists(img_name):
        print("file not exist:", img_name,"json name:",os.path.join(img_and_json_dir, it))
        continue
    image = cv2.imread(img_name)

    # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)

    # fill the contour with 255
    cv2.fillPoly(mask, [points], (255, 255, 255))

    # save the mask
    cv2.imwrite(mask_name, mask)
