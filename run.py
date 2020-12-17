import tensorflow as tf
from helpers.loss_functions import *
from helpers.submission import *

def main():
    path = 'saved_models/final_model'
    print(f'Loading model from {path}...')
    model = tf.keras.models.load_model(path, custom_objects={'focal_tversky_loss': focal_tversky_loss, 'dice_coef': dice_coefficient})
    predict_submissions(model, write_masks=False, submission_file='submission.csv')

if __name__ == "__main__":
    main()