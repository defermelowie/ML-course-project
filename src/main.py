import os
from PIL import Image
import numpy as np
from matplotlib import pyplot
from scipy import optimize

# Open an image, resize and split into channels
img_og = Image.open(
    './data/Garbage classification/cardboard/cardboard384.jpg')
img_res = img_og.resize((img_og.width//5, img_og.height//5))
(img_R_chan, img_G_chan, img_B_chan) = img_res.split()

print('height: %s, width: %s' % img_res.size)
