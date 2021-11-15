import os
from PIL import Image
import numpy as np
from matplotlib import pyplot
from scipy import optimize

# Constants
VERBOSE = 1
RESCALE_FACTOR = 10


def image_path_to_array(path: str, scale_factor: int) -> np.array:
    # Open an image, resize and split into channels
    img_og = Image.open(path)
    img_res = img_og.resize((img_og.width//scale_factor,
                            img_og.height//scale_factor))
    (img_R_chan, img_G_chan, img_B_chan) = img_res.split()

    if VERBOSE:
        print('[INFO] Loaded image: height=%s, width=%s' % img_og.size)

    # Combine image channels to np array
    (R_chan, G_chan, B_chan) = (np.asarray(img_R_chan).flatten(),
                                np.asarray(img_G_chan).flatten(),
                                np.asarray(img_B_chan).flatten())
    input_data = np.concatenate((R_chan, G_chan, B_chan))

    if VERBOSE:
        print('[INFO] Generated array: size=%s' % input_data.shape)

    return input_data


input_layer = image_path_to_array(
    './data/garbage_classification/metal/metal45.jpg', RESCALE_FACTOR)
print(input_layer)
