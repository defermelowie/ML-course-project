from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

VERBOSE = 1


class ImageLoader:
    path_list: List[dict]
    set: str
    scale_factor: int

    def __init__(self, path_list: List[dict], set: str, scale_factor: int = 10) -> None:
        self.path_list = filter(lambda item: item['set'] == set, path_list)
        self.set = set
        self.scale_factor = scale_factor

        if VERBOSE:
            print(f'[INFO] Set: {self.set}')

    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all images in path_list of given set as numpy array

        Returns:
            (X, Y)

            X: n by 3*dim
            Y: n
        """
        x = []
        for image in self.path_list:
            x = [x, self._img_array_from_path(image['path'])]
        print(x)
        # TODO: Convert x to array
        # TODO: Load y values
        pass

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

        if VERBOSE:
            print('[INFO] Loaded image: height=%s, width=%s' % img_og.size)
            print('[INFO] Resized image: height=%s, width=%s' % img_res.size)

        # Combine image channels to np array
        (R_chan, G_chan, B_chan) = (np.asarray(img_R_chan).flatten(),
                                    np.asarray(img_G_chan).flatten(),
                                    np.asarray(img_B_chan).flatten())
        input_data = np.concatenate((R_chan, G_chan, B_chan))

        if VERBOSE:
            print(f'[INFO] Generated array: size={input_data.shape}')

        return input_data
