"""
This module contains Class Definitions and functions for Tomography reconstructions
"""


from typing import Union


import numpy as np
from numpy.typing import NDArray
from scipy import fft


NumpyReal = Union[np.floating, np.integer]


class FilterBP:
    """Class for Back Projection Filter"""
    RAMP = 0
    # noinspection SpellCheckingInspection
    SHEPP_LOGAN = 1
    COSINE = 2
    HAMMING = 3
    # noinspection SpellCheckingInspection
    HANN = 4
    all = {RAMP, SHEPP_LOGAN, COSINE, HAMMING, HANN}

    def __init__(self, filter_type: int):
        if not filter_type in self.all:
            raise ValueError(f'Non existing filter: {filter_type}')
        else:
            self.filter_type = filter_type

    def __call__(self, n: int) -> NDArray:
        if n <= 0:
            raise ValueError('n must be positive')
        filter_array = self._ramp_filter(n)
        if self.filter_type == self.RAMP:
            pass
        elif self.filter_type == self.SHEPP_LOGAN:
            omega = np.pi * fft.fftfreq(n)[1:]
            filter_array[1:] *= np.sin(omega) / omega
        elif self.filter_type == self.COSINE:
            freq = np.linspace(0.0, np.pi, n, endpoint=False)
            filter_array *= fft.fftshift(np.sin(freq))
        elif self.filter_type == self.HAMMING:
            filter_array *= fft.fftshift(np.hamming(n))
        elif self.filter_type == self.HANN:
            filter_array *= fft.fftshift(np.hanning(n))
        return filter_array

    @staticmethod
    def _ramp_filter(n: int) -> NDArray:
        filter_array = np.zeros(n)
        if n % 2: # odd number of frequencies, even number of non-zero frequencies
            filter_array[0: n // 2 + 1] = np.linspace(0, 1, n // 2 + 1, endpoint=False)
            filter_array[n // 2 + 1:] = np.linspace(0, 1, n // 2 + 1, endpoint=False)[:0:-1]
        else: # even number of frequencies, odd number of non-zero frequencies
            filter_array[0 : n // 2 + 1] = np.linspace(0, 1, n//2 + 1)
            # Though we are placing a 1 two times, it is necessary in order to get the right values
            filter_array[n // 2:] = np.linspace(1, 0, n // 2, endpoint=False)
        return filter_array


def cut_square(img: NDArray) -> NDArray:
    """Cut Square from Image"""
    img_dim = img.shape[0]
    original_dim = img_dim / np.sqrt(2)
    to_cut = int(np.ceil((img_dim - original_dim) / 2))
    return img[to_cut:-to_cut + 1,to_cut:-to_cut + 1]


def cut_circle(img: NDArray) -> None:
    """Cut Circle from Image"""
    img_dim = img.shape[0]
    coords = np.array(np.ogrid[: img_dim, : img_dim], dtype=object)
    img_shape = np.array([img_dim, img_dim])
    radius = img_dim // 2
    dist = ((coords - (img_shape // 2)) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    img[outside_reconstruction_circle] = img.min()
