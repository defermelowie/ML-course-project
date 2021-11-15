import os
from PIL import Image
import numpy as np
from matplotlib import pyplot
from scipy import optimize

# Constants
RESCALE_FACTOR = 10

# Open an image, resize and split into channels
img_og = Image.open(
    './data/garbage_classification/cardboard/cardboard384.jpg')
img_res = img_og.resize((img_og.width//RESCALE_FACTOR,
                        img_og.height//RESCALE_FACTOR))
(img_R_chan, img_G_chan, img_B_chan) = img_res.split()

print('[INFO] Loaded image: height=%s, width=%s' % img_res.size)

# Combine image channels to np array
(R_chan, G_chan, B_chan) = (np.asarray(img_R_chan).flatten(),
                            np.asarray(img_G_chan).flatten(),
                            np.asarray(img_B_chan).flatten())

input_data = np.concatenate((R_chan, G_chan, B_chan))

print('[INFO] Input data: size=%s' % input_data.shape)
