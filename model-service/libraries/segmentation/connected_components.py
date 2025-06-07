"""
Modules that connect components in a binary image.
"""


import numpy as np
from numpy.typing import ArrayLike, NDArray


def _bounded_coords(x, y, width, height):
    x_min = np.max([0, x - 1])
    x_max = np.min([width - 1, x + 1])
    y_min = np.max([0, y - 1])
    y_max = np.min([height - 1, y + 1])
    return x_min, x_max, y_min, y_max


def connected_components(img: ArrayLike) -> tuple[int, NDArray[np.int_]]:
    """
    Return the connected components labeled as integer indexes
    @param img: binary image, a 2D numpy array with integer values
    @return: number of connected components, and a 2D numpy array with integer values
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    if not np.array(((img == 0) | (img == 1))).all():
        raise ValueError("Image must have ones and zeroes as values")
    labels = np.zeros(img.shape, dtype=np.int_)
    actual_label = 1
    height, width = img.shape
    for i in range(width):
        for j in range(height):
            if img[j, i] and not labels[j, i]:
                labels[j, i] = actual_label
                queue = [(i, j)]
                while queue:
                    x, y = queue.pop(0)
                    x_min, x_max, y_min, y_max = _bounded_coords(x, y, width, height)
                    for k in range(x_min, x_max + 1):
                        for l in range(y_min, y_max + 1):
                            if img[l, k] and not labels[l, k]:
                                labels[l, k] = actual_label
                                queue.append((k, l))
                actual_label += 1
    return actual_label - 1, labels
