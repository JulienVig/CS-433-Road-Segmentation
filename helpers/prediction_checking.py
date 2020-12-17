import numpy as np
import matplotlib.pyplot as plt
from .image_processing import prediction_to_rgb_image
from .constants import *


def visualize_random_predictions(x, y, predictions, size=4):
    """
    Visualises random predictions. For each, print from left to right :
    Base image
    Prediction with probabilities between 0 and 1
    Prediction with only 0s and 1s
    Labels
    :param x: Original images
    :param y: Labels
    :param predictions: Predictions
    :param size: Number of predictions to display
    """
    fig, axes = plt.subplots(size, 4, figsize=(20, 5*size))
    for i, idx in enumerate(np.random.randint(len(x), size=size)):
        pred = predictions[idx]
        binary_pred = (pred >= ROAD_THRESHOLD_PIXEL_PRED) * 1.0
        axes[i, 0].imshow(x[idx])
        axes[i, 1].imshow(prediction_to_rgb_image(pred))
        axes[i, 2].imshow(prediction_to_rgb_image(binary_pred))
        axes[i, 3].imshow(prediction_to_rgb_image(y[idx]))
