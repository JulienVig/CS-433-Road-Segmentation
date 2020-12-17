import os
import numpy as np
from .constants import *
from PIL import Image
from tqdm import tqdm


def load_test_images(images=[], low_memory=False):
    """
    Loads the test images. See load_images
    :param images: If low_memory is true, will fill the argument instead of creating a new array. Instead does nothing
    :param low_memory: If True and an array is given as argument,
    will load images while saving as much RAM as possible
    :return: A numpy array with all loaded images, or nothing if low_memory is true
    """
    return load_images(TEST_IMAGES_DIR, N_TEST_IMAGES, filename_format="test_{}.png", images=images, low_memory=low_memory)


def load_features(num_images, images=[], low_memory=False):
    """
    Loads the train images. See load_images
    :param num_images: Number of train images to load
    :param images: If low_memory is true, will fill the argument instead of creating a new array. Instead does nothing
    :param low_memory: If True and an array is given as argument,
    will load images while saving as much RAM as possible
    :return: A numpy array with all loaded images, or nothing if low_memory is true
    """
    return load_images(TRAIN_IMAGES_DIR, num_images, images=images, low_memory=low_memory)


def load_labels(num_images, images=[], low_memory=False):
    """
    Loads the train images. See labels
    :param num_images: Number of train labels to load
    :param images: If low_memory is true, will fill the argument instead of creating a new array. Instead does nothing
    :param low_memory: If True and an array is given as argument,
    will load images while saving as much RAM as possible
    :return: A numpy array with all loaded images, or nothing if low_memory is true
    """
    if low_memory:
        load_images(TRAIN_LABELS_DIR, num_images, images=images, low_memory=low_memory)
        for i in range(num_images):
            for x in range(TRAINING_IMG_SIZE):
                for y in range(TRAINING_IMG_SIZE):
                    images[i][x][y] = 1 * (images[i][x][y] > ROAD_THRESHOLD_PATCH)
    else:
        gt = load_images(TRAIN_LABELS_DIR, num_images)
        return 1.0 * (gt > ROAD_THRESHOLD_PIXEL)


def load_generated_data(transformations=None, images=[], groundtruth=[], low_memory=False):
    """
    Load all the images we generated during the data augmentation phase
    :param transformations: list of transformation folders to load, if None or empty loads everything,
    current possible values: ['flip', 'hard_mix', 'hard_raw', 'mix', 'mix_big', 'rotation', 'rotation_big', 'shift']
    :param images: If low_memory is true, will fill the argument instead of creating a new array. Instead does nothing
    :param groundtruth: Same logic as images
    :param low_memory: If True and an array is given as argument,
    will load images while saving as much RAM as possible
    :return: A numpy array with all loaded images and groundtruths, or nothing if low_memory is true
    """
    # List all possible folders we can load
    # Condition on isdir to exclude files.
    folders_to_load = [folder for folder in os.listdir(GENERATION_DIR) if os.path.isdir(GENERATION_DIR + folder)]

    # If specific folders are specified
    if transformations is not None and len(transformations) > 0:
        folders_to_load = [folder for folder in transformations if folder in folders_to_load]

    if not low_memory:
        images = []
        groundtruth = []

    for folder in folders_to_load:
        image_path = GENERATION_DIR + folder + '/images/'
        gt_path = GENERATION_DIR + folder + '/groundtruth/'
        load_images(image_path, images=images, low_memory=True)
        load_images(gt_path, images=groundtruth, low_memory=True, grayscale=True)

    if not low_memory:
        return np.asarray(images), np.asarray(groundtruth)


def load_images(path, num_images=None, filename_format="satImage_{:03d}.png", images=[], low_memory=False, grayscale=False):
    """
    Loads all images in some folder. It is assume that said folder contains only those images and nothing else.
    :param path: Path of the folder from which to load the images
    :param num_images: Will load images according to the given format, i = 1 to num_images + 1
    :param filename_format: Format of the names of images to load
    :param images: If low_memory is true, will fill the argument instead of creating a new array. Instead does nothing
    :param low_memory: If True and an array is given as argument,
    will load images while saving as much RAM as possible
    :return: A numpy array with all loaded images, or nothing if low_memory is true
    """
    if not low_memory:
        images = []

    if num_images is None:
        num_images = len(os.listdir(path))

    for i in tqdm(range(num_images), desc="Loading " + path):
        image_id = filename_format.format(i+1)
        image_filename = path + image_id
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            if img.mode == 'RGBA':
                img = img.convert('L' if grayscale else 'RGB')  # For RGBA images
            img = np.array(img)/255  # Scale between 0 and 1
            images.append(img)
        else:
            raise ValueError('File ' + image_filename + ' does not exist')

    if not low_memory:
        return np.asarray(images)
