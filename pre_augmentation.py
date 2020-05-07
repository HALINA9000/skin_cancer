# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:25:52 2019

@author: Tom S. tom dot s at halina9000 dot com

ISIC Archive image dataset pre-augmentation.

Original ISIC image dataset has two issue to solve:
1. It contains images with different sizes. For example: 600x450, 3008x200,
   1788x1713 px.
2. Dataset contains about 20 878 images of benign lesions and 3 008
   of malignant ones.

To deal with (1) and do not loose unknown features by significant resizing,
images are handled in two ways:
- small images are placed on one color background with pre-defined size
- bigger images are resized to width x pre-defined height.

To solve (2):
- for images with benign benign lesions: 4 stripes are cropped
  Result: 4 * 20 878 = 83 512 stripes
- for images with malignant lesions: 7 stripes are cropped and then each one
  of them additionally is flipped 3 times
  Result: 3 008 * 7 * (1 + 3) = 84 224 stripes
"""
import os  # Operating system functions
from PIL import Image  # Image processing
from tqdm import tqdm  # Progress bar
from utils import load_data  # Project utilities

STRIPE_SIZE = (1000, 2000)  # Width & height (px) of stripe as output file
DATA_PATH = 'Data'  # Main data dir
IMG_DIR = os.path.join(DATA_PATH, 'Images')  # Original images dir
STRIPES_DIR = os.path.join(DATA_PATH, 'stripes')  # Pre-augmented images
CSV_DIR = os.path.join(DATA_PATH, 'csv')  # CSV files dir
CSV_FILE = 'main.csv'

if not os.path.exists(STRIPES_DIR):
    os.makedirs(STRIPES_DIR)


def crop_stripe(img_obj, src_size, out_size):
    """
    Crop from input image a stripe with requested size.

    Parameters
    ----------
        img_obj : PIL.Image
            Object containing picture of lesion.
        src_size : tuple(int, int)
            Size of source image as a tuple(width, height).
        out_size : tuple(int, int)
            Size of output stripe as a tuple(width, height).

    Returns
    -------
        PIL.Image
            stripe cropped from input image.
    """
    return img_obj.crop(
        ((src_size[0] - out_size[0]) // 2,
         (src_size[1] - out_size[1]) // 2,
         (src_size[0] + out_size[0]) // 2,
         (src_size[1] + out_size[1]) // 2))


def stripe_save(img_file_name, stripe_obj, angle, flip, lesion_type,
                stripes_dir):
    """
    Save stripe image.

    Parameters
    ----------
        img_file_name : str
            Original name of image file with no file extension.
        stripe_obj : PIL.Image
            Object containing input stripe.
        angle : str
            Angle of stripe in original image file. 3 digits as a string.
        flip : str
            Number of flip.
        lesion_type: str
            Type of lesion.
        stripes_dir : str
            Full path of destination directory where both input and generated
            stripes will be saved.
    """
    stripe_file = img_file_name + lesion_type[0] + angle + flip
    stripe_obj.save(os.path.join(stripes_dir, stripe_file + '.jpg'))


def stripes_gen(src_img_dir, out_img_dir, img_file_name, img_file_ext,
                src_size, out_size, lesion_type):
    """
    Generates rectangular stripes from original image file.

    Parameters
    ----------
        src_img_dir : str
            Full path of source directory with image files.
        out_img_dir : str
            Full path of output directory where both input and generated
            stripes will be saved.
        img_file_name : str
            Original name of image file without extension.
        img_file_ext : str
            Original name of image file extension.
        src_size : tuple(int, int)
            Size of input image as a tuple(width, height).
        out_size : tuple(int, int)
            Size of output image as a tuple(width, height).
        lesion_type: str
            Type of lesion.
    """
    img_path = os.path.join(src_img_dir, img_file_name + img_file_ext)
    img_obj = Image.open(img_path)
    """Flatten if image with transparency."""
    if img_obj.mode != 'RGB':
        img_obj = img_obj.convert('RGB')
    """Portrait to landscape."""
    if src_size[1] > src_size[0]:
        img_obj = img_obj.transpose(Image.ROTATE_90)
        src_size = (src_size[1], src_size[0])
    """Handling small images - adding fixed size square background."""
    if src_size[1] < out_size[1]:
        sqr_size = (out_size[1], out_size[1])
        bg_img_obj = Image.new('RGB', sqr_size, color='black')
        bg_img_obj.paste(img_obj, ((out_size[1] - src_size[0]) // 2,
                                   (out_size[1] - src_size[1]) // 2))
        img_obj = bg_img_obj.copy()
        src_size = sqr_size  # New source image size.
    """Handling large images - resizing."""
    if src_size[1] > out_size[1]:
        resize_ratio = src_size[1] / out_size[1]
        src_size = (int(src_size[0] / resize_ratio), out_size[1])
        img_obj = img_obj.resize(src_size)
    """Generate stripes."""
    if lesion_type == 'benign':
        angles_lst = [i * 45. for i in range(4)]
    else:
        angles_lst = [i * 22.5 for i in range(7)]
    for angle in angles_lst:
        stripe_obj = img_obj
        angle_str = str(int(angle * 10)).zfill(4)
        flip = 'f0'
        if angle != 0:
            stripe_obj = img_obj.rotate(angle)
        stripe_obj = crop_stripe(stripe_obj, src_size, out_size)
        stripe_save(img_file_name,
                    stripe_obj,
                    angle_str,
                    flip,
                    lesion_type,
                    out_img_dir)
        if lesion_type == 'malignant':
            flipping_types_lst = [Image.FLIP_LEFT_RIGHT,
                                  Image.FLIP_TOP_BOTTOM,
                                  Image.FLIP_LEFT_RIGHT]
            for i, flipping_type in enumerate(flipping_types_lst):
                stripe_obj = stripe_obj.transpose(flipping_type)
                flip = 'f' + str(i + 1)
                stripe_save(img_file_name,
                            stripe_obj,
                            angle_str,
                            flip,
                            lesion_type,
                            out_img_dir)

    img_obj.close()


def main():
    main_df = load_data(CSV_DIR, CSV_FILE)
    records_num = len(main_df)
    for i in tqdm(range(records_num)):
        record = main_df[['dsc_file',
                          'img_ext',
                          'width',
                          'height',
                          'lesion_type']].iloc[i]
        stripes_gen(IMG_DIR,
                    STRIPES_DIR,
                    record['dsc_file'],
                    record['img_ext'],
                    (record['width'], record['height']),
                    STRIPE_SIZE,
                    record['lesion_type'])


if __name__ == "__main__":
    main()
