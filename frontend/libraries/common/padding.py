"""
This file contains functions for padding images.
"""


from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


NumpyReal = Union[np.floating, np.integer]
NumpyRealType = type(NumpyReal)


def zero_pad_img(img: ArrayLike, padding_width: int) -> NDArray[np.number]:
    """
    Takes a one-channel image and returns a zero-padded image.
    @param img: image to be zero-padded
    @param padding_width: width of the padding, a positive integer
    @return: zero-padded image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(padding_width, int) or padding_width < 1:
        raise ValueError("Padding width must be a positive integer")
    if not issubclass(img.dtype.type, np.number):
        raise ValueError(f"Image values must be a numpy numeric type, got {img.dtype.type} instead")
    if img.ndim != 2:
        raise ValueError("Image must be a 2D array")
    img_padded = \
        np.zeros((img.shape[0] + 2 * padding_width, img.shape[1] + 2 * padding_width),
                 dtype=img.dtype)
    img_padded[padding_width:padding_width + img.shape[0],
               padding_width:padding_width + img.shape[1]] = img
    return img_padded


def extension_pad_img(img: ArrayLike, size: int) -> NDArray[np.number]:
    """
    Extends the image border with its own values. Useful for edge detection.
    @param img: input image
    @param size: size of the extension
    @return: output image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(size, int) or size < 1:
        raise ValueError("Size must be a positive integer")
    if not issubclass(img.dtype.type, np.number):
        raise ValueError(f"Image values must be a numpy numeric type, got {img.dtype.type} instead")
    image_type = img.dtype
    up = np.ones((size, 1)) * img[0,:]
    bottom = np.ones((size, 1)) * img[-1,:]
    left = img[:, 0].reshape((-1, 1)) * np.ones((1, size))
    right = img[:, -1].reshape(-1, 1) * np.ones((1, size))
    up_right = np.ones((size, size), dtype=image_type) * img[0, 0]
    up_left = np.ones((size, size), dtype=image_type) * img[0, -1]
    bottom_right = np.ones((size, size), dtype=image_type) * img[-1, 0]
    bottom_left = np.ones((size, size), dtype=image_type) * img[-1, -1]
    result = np.zeros((img.shape[0] + 2 * size, img.shape[1] + 2 * size), dtype=image_type)
    result[size:-size, size:-size] = img
    result[:size, size:-size] = up
    result[-size:, size:-size] = bottom
    result[size:-size, :size] = left
    result[size:-size, -size:] = right
    result[:size, :size] = up_left
    result[:size, -size:] = up_right
    result[-size:, :size] = bottom_left
    result[-size:, -size:] = bottom_right
    return result
