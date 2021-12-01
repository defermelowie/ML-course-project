from typing import List
import numpy as np
from csv import DictReader
import logging

from imageLoader import ImageLoader

# Constants
IMG_RESCALE_FACTOR = 15
DATA_PATHS = './data/path_list.csv'

# Set up logging
logging.basicConfig(
    format='[%(asctime)s | %(levelname)s] %(message)s', level=logging.INFO)

# Load path_list from csv file
path_list = []
with open(DATA_PATHS, mode='r') as fd:
    dictReader = DictReader(fd)
    for dict in dictReader:
        path_list.append(dict)

# Load X from path_list
imageLoader = ImageLoader(path_list, 'test')
(input_layer, y) = imageLoader.load_images()
print(input_layer)
print(y)
