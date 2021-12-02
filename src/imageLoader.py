from os import path
from typing import Dict, List, Set, Tuple
from PIL import Image
import numpy as np
import logging


class ImageLoader:
    path_list: List[dict]
    sets: Set[str]
    scale_factor: int

    def __init__(self, path_list: List[dict], sets: Set[str], scale_factor: int = 10) -> None:
        self.path_list = filter(lambda item: item['set'] in sets, path_list)
        self.sets = sets
        self.scale_factor = scale_factor

        logging.info(f'Load set: {self.sets}')

    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all images in path_list of given set as numpy array

        Returns:
            (X, Y)

            X: (n, 3*dim)
            Y: (n,)
        """
        # Create empty arrays
        X = np.array([])
        Y = np.array([])

        # Populate arrays
        for image in self.path_list:
            if(X.size == 0):
                X = self._img_array_from_path(image['path'])
                Y = np.array(image['type'])
            else:
                X = np.c_[X, self._img_array_from_path(image['path'])]
                Y = np.r_[Y, np.array(image['type'])]

        # Transpose X in order to have shape (n, 3*dim)
        X = X.T

        # Logging
        for setname in ['papier', 'glas', 'pmd', 'restafval']:
            logging.debug(
                f'#{setname} in Y: {np.count_nonzero(Y == setname)}')
        logging.info(f'Shape of X: {X.shape}')
        logging.info(f'Shape of Y: {Y.shape}')

        return (X, Y)

    def _img_array_from_path(self, path: str) -> np.array:
        """Open image and convert to numpy array

        Returns:
            x: np.array: 3*dim x 1
        """
        # Open an image, resize and split into channels
        img_og = Image.open(path)
        img_res = img_og.resize((img_og.width//self.scale_factor,
                                 img_og.height//self.scale_factor),
                                resample=Image.ANTIALIAS)
        (img_R_chan, img_G_chan, img_B_chan) = img_res.split()

        logging.debug('Loaded image: path="%s" height=%s, width=%s' %
                      (path, img_og.size[0], img_og.size[1]))
        logging.debug('Resized image: height=%s, width=%s' % img_res.size)

        # Combine image channels to np array
        (R_chan, G_chan, B_chan) = (np.asarray(img_R_chan).flatten(),
                                    np.asarray(img_G_chan).flatten(),
                                    np.asarray(img_B_chan).flatten())
        input_data = np.concatenate((R_chan, G_chan, B_chan))

        #logging.debug(f'Generated image array: size={input_data.shape}')

        return input_data
