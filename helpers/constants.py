# Section 1 : Not parameters (don't try and tune these)
ROAD_THRESHOLD_PATCH = .25
# Set image patch size in pixels
# PATCH_SIZE should be a multiple of 4 and divide the images' size
PATCH_SIZE = 16
PIXEL_DEPTH = 255
TRAINING_IMG_SIZE = 400
TEST_IMG_SIZE = 608
N_TEST_IMAGES = 50
N_TRAIN_IMAGES = 100

# Section 2 : Paths
MODELS_SAVE_DIR = 'saved_models/'
SUBMISSIONS_DIR = 'submissions/'
TRAIN_IMAGES_DIR = 'data/train/original/images/'
TRAIN_LABELS_DIR = 'data/train/original/groundtruth/'
GENERATION_DIR = 'data/train/generated/'
TEST_IMAGES_DIR = 'data/test/original/'
PREDICTIONS_SAVE_DIR = 'data/test/predictions/'

# Section 3 : Hyperparameters
SEED = 66478  # Set to None for random seed
VALIDATION_SIZE = .2  # Remaining part is for training and testing
TRAINING_SIZE = .8  # Remaining part is for testing

ROAD_THRESHOLD_PIXEL = 0.5  # The greyscale value above which a pixel from original image is considered black
ROAD_THRESHOLD_PIXEL_PRED = 0.5  # The threshold value above which we predict road instead of background
