"""
Module that contains scaling image functions
"""


from numbers import Real
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from libraries.common.padding import zero_pad_img


NumpyReal = Union[np.floating, np.integer]


def equalize_hist(img: ArrayLike) -> NDArray[np.uint8]:
    """
    Computes the image histogram equalization for a single channel, 8-bit image.
    @param img: input image
    @return: output image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim != 2 or not issubclass(img.dtype.type, np.uint8):
        raise ValueError('Image must be a 2D numpy array with 8-bit integer values')

    flattened = img.flatten()
    hist = np.zeros(256)
    for pixel in flattened:
        hist[pixel] += 1
    hist_cum_sum_norm = np.cumsum(hist)
    hist_cum_sum_norm /= hist_cum_sum_norm[-1]
    density_min = hist_cum_sum_norm[img.min()]
    scaling = 255 / (1 - density_min)
    hist_cum_sum_norm = (hist_cum_sum_norm - density_min) * scaling
    result = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = hist_cum_sum_norm[img[i, j]]
    return result


def equalize_local_hist(img: ArrayLike,
                        mask_size: int = 3,
                        mode: str = 'same') -> NDArray[np.uint8]:
    """
    Computes the image local histogram equalization
    @param img: input image
    @param mask_size: size of the mask
    @param mode: 'same' or 'valid'
    @return: output image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(img, np.ndarray) or img.ndim != 2 or not img.dtype.type == np.uint8:
        raise ValueError('Image must be a 2D numpy array with 8-bit integer values')
    if not isinstance(mask_size, int) or mask_size < 3 or mask_size % 2 == 0:
        raise ValueError(f'Mask size must be an odd integer greater or \
                         equal than 3, got {mask_size} instead')
    if not isinstance(mode, str) or mode not in ['same', 'valid']:
        raise ValueError("Mode must be 'same' or 'valid'")

    # zero_padding
    pad = mask_size // 2
    if mode == 'same':
        img = zero_pad_img(img, pad)

    result = np.zeros((img.shape[0] - pad, img.shape[1] - pad), dtype=np.uint8)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            patch = img[i:i + mask_size, j:j + mask_size]
            flattened = patch.flatten()
            hist = np.zeros(256, dtype=np.int_)
            for pixel in flattened:
                hist[int(pixel)] += 1
            hist_cum_sum_norm = np.cumsum(hist) / np.sum(hist)
            density_min = hist_cum_sum_norm[img.min()]
            scaling = 255 / (1 - density_min)
            hist_cum_sum_norm = (hist_cum_sum_norm - density_min) * scaling
            result[i, j] = hist_cum_sum_norm[patch[pad, pad]]
    return result


@np.vectorize
def _intensity_scale(f, f1, f2, f_max):
    """
    Compute the intensity scaling formula
    @param f: input value
    @param f1: lower bound
    @param f2: upper bound
    @param f_max: maximum value
    @return: output value
    """
    return (f1 < f <= f2) * (f - f1) / (f2 - f1) * f_max


def intensity_scale(img: ArrayLike, f1: Real, f2: Real, f_max: NumpyReal) -> np.float64:
    """
    Compute the intensity scaling formula (wrapper function)
    :param img: single-channel image
    :param f1: lower bound
    :param f2: upper bound
    :param f_max: maximum value
    :return: output value according to the intensity scaling formula
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not img.ndim != 2:
        raise ValueError('Image must be a 2D numpy array with real values')
    if not f1 < f2:
        raise ValueError("F1 must be less than F2")
    if not f_max > .0:
        raise ValueError("f_max must be greater than zero")
    return _intensity_scale(img, f1, f2, f_max)


@np.vectorize
def binarize(value: Real, threshold: Real) \
-> int:
    """
    Takes a value and a threshold and returns a binary value.
    @param value: value to be compared
    @param threshold: threshold to be used for comparison
    @return: result value, either 1 or 0
    """
    if value > threshold:
        return 1
    return 0
