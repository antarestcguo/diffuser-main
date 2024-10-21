from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str)
args = parser.parse_args()


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


postfix_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

img_path = args.img_path
img_file_list = os.listdir(img_path)
for it in img_file_list:
    s_n, s_e = os.path.splitext(it)
    if s_e not in postfix_list:
        print("no need to process:", it)
        continue

    img = Image.open(os.path.join(img_path, it))
    img = exif_transpose(img)
    img.save(os.path.join(img_path, it))
