"""
This file contains tomography reconstruction functions
"""


from typing import Union

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import fft
from skimage.transform import rotate

from libraries.reconstruction.common import (FilterBP,
                                             cut_square)


NumpyReal = Union[np.floating, np.integer]

def inverse_radon(sinogram: ArrayLike, theta: ArrayLike | None = None, circle: bool = False,
                  filter_type: int | None = None) -> NDArray[np.floating]:
    """
    Computes the inverse Radon transform of a sinogram.
    @param sinogram: numeric array containing the sinogram
    @param theta: numeric array containing the angle projections
    @param circle: bool, whether the final image is a circle
    @param filter_type: int, filter to be applied to the sinogram
    @return: numeric array containing the reconstructed image
    """
    if not isinstance(sinogram, np.ndarray):
        sinogram = np.array(sinogram)
    if not issubclass(sinogram.dtype.type, NumpyReal) or not sinogram.ndim == 2:
        raise ValueError("Sinogram must be a real-valued 2D array")
    if theta is None:
        theta = np.linspace(0, 180, sinogram.shape[1], endpoint=False)
    else:
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        if not issubclass(theta.dtype.type, np.floating) or not theta.ndim == 1:
            raise ValueError("Theta must be a real-valued 1D array")
        if not theta.shape[0] == sinogram.shape[1]:
            raise ValueError("Theta must have the same length as the number of projections")
    if filter_type is not None:
        filter_array = FilterBP(filter_type)(sinogram.shape[0])
    else:
        filter_array = np.ones(sinogram.shape[0])
    reconstruction = np.zeros((sinogram.shape[0], sinogram.shape[0]))
    for angle_index, angle in enumerate(theta):
        filtered_projection = \
            np.real(np.array(fft.ifft(np.array(fft.fft(sinogram[:,angle_index])) * filter_array)))
        reconstruction +=  rotate(np.tile(filtered_projection, (sinogram.shape[0], 1)), angle)
    reconstruction *= np.pi / (2 * sinogram.shape[1])
    if not circle:
        reconstruction = cut_square(reconstruction)
    else:
        img_dim = reconstruction.shape[0]
        coords = np.array(np.ogrid[: img_dim, : img_dim], dtype=object)
        img_shape = np.array([img_dim, img_dim])
        radius = img_dim // 2
        dist = ((coords - (img_shape // 2)) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius**2
        reconstruction[outside_reconstruction_circle] = reconstruction.min()
    return reconstruction
