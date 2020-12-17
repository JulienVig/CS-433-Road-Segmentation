import tensorflow.keras.backend as k

# ALPHA = 0.5 and GAMMA = 1 is dice loss
ALPHA = 0.6  # Closer to 1 will penalize False Negatives more (= Saying background when it's road)
BETA = 1 - ALPHA  # Closer to 1 will penalize False Positives more (= Saying road when it's background)
GAMMA = 0.75  # Non-linearity. Above one will focus on harder examples
SMOOTH = 1.  # Just a constant to avoid zero divisions


def create_loss_function(alpha=ALPHA, gamma=GAMMA):
    """
    Return a function that only takes y_pred and y_true as an argument, fixing alpha and gamma
    :param alpha: See above
    :param gamma: See above
    :return: A function computing FTL with fixed parameters
    """
    def loss_function(y_true, y_pred):
        return focal_tversky_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    return loss_function


def focal_tversky_loss(y_true, y_pred, gamma=GAMMA, alpha=ALPHA):
    """
    Non-linear Tversky loss
    :param y_true: labels
    :param y_pred: predictions
    :param gamma: non-linearity parameter
    :param alpha: Weight of false negatives
    :return: The Focal Tversky Loss
    """
    return k.pow((1 - tversky_index(y_true, y_pred, alpha=alpha)), gamma)


def tversky_index(y_true, y_pred, alpha=ALPHA):
    """
    Computes the Tversky index, which is a weighted dice index
    Higher alpha means more penalty for false negatives
    """
    beta = 1 - alpha
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    tp = k.sum(y_true_f * y_pred_f)
    fn = k.sum(y_true_f * (1 - y_pred_f))
    fp = k.sum((1 - y_true_f) * y_pred_f)
    return (tp + SMOOTH) / (tp + alpha*fn + beta*fp + SMOOTH)


# The rest is here for illustration but all of these are special cases of the Focal Tversky Loss
def tversky_loss(y_true, y_pred, alpha=ALPHA):
    """
    Non-Focal Tversky Loss. Equivalent to FTL with gamma=1
    """
    return 1 - tversky_index(y_true, y_pred, alpha=alpha)


def dice_loss(y_true, y_pred):
    """
    F1-score but as a loss function Equivalent to Tversky Loss with alpha=0.5
    """
    return 1 - dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred):
    """
    F1-score Equivalent to Tversky Index with alpha=0.5
    """
    return tversky_index(y_true, y_pred, alpha=0.5)
