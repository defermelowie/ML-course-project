from typing import List
import numpy as np
from csv import DictReader

from imageLoader import ImageLoader

# Constants
VERBOSE = 1
IMG_RESCALE_FACTOR = 15
DATA_PATHS = './data/path_list.csv'

path_list = []
with open(DATA_PATHS, mode='r') as fd:
    dictReader = DictReader(fd)
    for dict in dictReader:
        path_list.append(dict)

imageLoader = ImageLoader(path_list, 'test')
input_layer = imageLoader.load_images()
# print(input_layer)
