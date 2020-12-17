import os
import numpy as np
from .constants import *
from tqdm import tqdm
from PIL import Image


def predict_patch(p):
    """
    For a given patch, predict road or background
    :param p: Image patch
    :return: 1 if the patch is a road patch, and 0 if it's a background patch
    """
    return 1 if p.mean() > ROAD_THRESHOLD_PATCH else 0


def prediction_to_rgb_image(prediction):
    """
    Transforms a greyscale prediction into an RBG image
    :param prediction: Prediction to be transformed
    :return: An RBG copy of it
    """
    return np.stack((prediction, prediction, prediction), axis=-1)


def apply_masks_on_test(opacity=100):
    """
    Superimposes a transparent red mask with a prediction on test images
    :param opacity: Opacity of the mask
    """
    for i in tqdm(range(N_TEST_IMAGES), desc="Loading"):
        image_id = 'test_{}.png'.format(i + 1)
        test_image_path = TEST_IMAGES_DIR + image_id
        mask_path = PREDICTIONS_SAVE_DIR + image_id
        if os.path.isfile(test_image_path) and os.path.isfile(mask_path):
            img = Image.open(test_image_path)
            mask = Image.open(mask_path)
            masked = mask_image(img, mask, opacity)
            masked_path = PREDICTIONS_SAVE_DIR + 'test_{}_with_mask.png'.format(i + 1)
            masked.save(masked_path)
        else:
            raise ValueError('Files {} or {} '.format(test_image_path, mask_path) + ' do not exist')


def mask_image(img, mask, opacity=100):
    """
    Applies a transparent red mask onto an image
    :param img: Original image
    :param mask: Mask to be applied
    :param opacity: Opacity of the mask (0-255)
    :return: The image with the mask on it
    """
    mask = mask.convert('RGBA')
    pixels = mask.getdata()

    new_pixels = []
    for px in pixels:
        if px[0] == 255 and px[1] == 255 and px[2] == 255:
            new_pixels.append((255, 0, 0, opacity))
        else:
            new_pixels.append((0, 0, 0, 0))

    mask.putdata(new_pixels)
    img.paste(mask, None, mask)
    return img
