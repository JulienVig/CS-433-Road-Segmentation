from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, LeakyReLU, Conv2D, MaxPooling2D

IMAGE_SIZE = 400
KERNEL_SIZE = (3, 3)
FIRST_KERNEL_SIZE = (5, 5)
DROPOUT_RATE = 0.2
LEAKYRATE = 0.05


def cnn_patch_prediction():
    """
    Creates a simple CNN model
    :return: The model
    """
    input_img = Input((IMAGE_SIZE, IMAGE_SIZE, 3), name='img')
    l1 = Conv2D(64, FIRST_KERNEL_SIZE, activation=LeakyReLU(alpha=LEAKYRATE), padding='same')(input_img)
    l1 = convolution_layer(l1, 128, dropout=False)
    l2 = convolution_layer(l1, 128)
    l3 = convolution_layer(l2, 256)
    l4 = convolution_layer(l3, 64)
    outputs = Conv2D(1, KERNEL_SIZE, activation='sigmoid', padding='same')(l4)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def convolution_layer(input, n_filters, dropout=True):
    """
    A convolution layer
    """
    output = Conv2D(n_filters, KERNEL_SIZE, activation=LeakyReLU(alpha=LEAKYRATE), padding='same')(input)
    output = MaxPooling2D(2, 2)(output)
    if dropout:
        output = Dropout(DROPOUT_RATE)(output)
    return output
