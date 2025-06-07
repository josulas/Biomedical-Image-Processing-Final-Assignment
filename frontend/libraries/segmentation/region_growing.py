"""
Region Growing Algorithm Implementation
"""


import numpy as np
from numpy.typing import ArrayLike, NDArray


class NeighborhoodTypes:
    """Class to store the neighborhood types for region growing algorithms."""
    N4 = 0
    N8 = 1
    def __iter__(self):
        yield self.N4
        yield self.N8


class RegionGrowingModes:
    """Class to store the modes for region growing algorithms."""
    FIXED = 0
    ADAPTIVE = 1
    def __iter__(self):
        yield self.FIXED
        yield self.ADAPTIVE


class ConnectionCriteria:
    """
    Class to store the connection criteria of a region growing algorithm.
    """
    def __init__(self, connectivity: int = NeighborhoodTypes.N4,
                 mode: int = RegionGrowingModes.FIXED,
                 params: tuple[int, int] | int = (0, 255)):
        if connectivity not in {NeighborhoodTypes.N4, NeighborhoodTypes.N8}:
            raise ValueError(f"Connectivity must be {NeighborhoodTypes.N4} for \
                             'N4' or {NeighborhoodTypes.N8} for 'N8',\
                             got{connectivity} instead.")
        if mode not in list(RegionGrowingModes()):
            raise ValueError(f"Mode must be {RegionGrowingModes.FIXED} for 'fixed' \
                             or {RegionGrowingModes.ADAPTIVE} for\
                             'adaptive', got{mode} instead.")
        if mode == RegionGrowingModes.FIXED and (not isinstance(params, tuple) \
                                                 or not 0 <= params[0] <= params[1]):
            raise ValueError(f"Params must be a tuple of two positive integers, \
                             the second greater than the first, \
                             got {params} instead.")
        if mode == RegionGrowingModes.ADAPTIVE and (not isinstance(params, int) or params < 1):
            raise ValueError(f"Params must be a positive integer for the adaptive mode, \
                             got {params} instead.")
        self.connectivity = connectivity
        self.mode = mode
        self.params = params


def region_growing(img: ArrayLike, seed: tuple[int, int], criterion: ConnectionCriteria) \
-> NDArray[np.integer]:
    """
    Takes a one-channel image, a seed value, a criterion, and returns a binary image with the 
    same size as the input image.
    @param img: grayscale image, a 2D numpy array with integer values
    @param seed: seed value, a tuple of two integers
    @param criterion: connection criteria of the region growing algorithm
    @return: binary image with the same size as the input image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    height, width = img.shape
    if seed[0] < 0 or seed[0] >= height or seed[1] < 0 or seed[1] >= width:
        raise ValueError(f"Seed value must be inside the image, got {seed} instead.")
    params = criterion.params
    if not isinstance(params, tuple): # Avoids linter errors
        params = (params, 0)
    neighbor_type = criterion.connectivity

    def _get_neighbors(coordinates_tuple: tuple[int, int],
                       connectivity: int) -> set[tuple[int, int]]:
        """
        Returns the neighbors of a given pixel in the image based on the connectivity type.
        :param coordinates_tuple: a tuple of two integers representing the pixel coordinates (y, x)
        :param connectivity: the connectivity type, either N4 or N8
        :return: a set of tuples representing the coordinates of the neighbors
        """
        neighbors_set = set()
        y, x = coordinates_tuple
        if y > 0:
            neighbors_set.add((y - 1, x))
        if x > 0:
            neighbors_set.add((y, x - 1))
        if y < height - 1:
            neighbors_set.add((y + 1, x))
        if x < width - 1:
            neighbors_set.add((y, x + 1))
        if connectivity == NeighborhoodTypes.N8:
            if y > 0 and x > 0:
                neighbors_set.add((y - 1, x - 1))
            if y > 0 and x < width - 1:
                neighbors_set.add((y - 1, x + 1))
            if y < height - 1 and x > 0:
                neighbors_set.add((y + 1, x - 1))
            if y < height - 1 and x < width - 1:
                neighbors_set.add((y + 1, x + 1))
        return neighbors_set

    def _is_in_shape(coordinates_tuple: tuple[int, int],
                     mode: int,
                     params_tuple: tuple[int, int]) -> bool:
        """
        Checks if a given pixel is in the shape defined by the connection criteria.
        :param coordinates_tuple: a tuple of two integers representing the pixel coordinates (y, x)
        :param mode: the mode of the region growing algorithm, either FIXED or ADAPTIVE
        :param params_tuple: a tuple of two integers representing 
                             the parameters of the connection criteria
        :return: True if the pixel is in the shape, False otherwise
        """
        y, x = coordinates_tuple
        if mode == RegionGrowingModes.FIXED:
            # params[0] is the min value and params[1] the max value
            return params_tuple[0] <= img[y, x] <= params_tuple[1]
        else:
            # params[0] is the max_diff and params[1] the parent's value
            return np.abs(img[y, x] - params_tuple[1]) <= params_tuple[0]

    result = np.zeros_like(img, dtype=np.uint8)
    result[seed] = 1
    if criterion.mode == RegionGrowingModes.FIXED:
        queue = {seed}
        while queue:
            coordinates = queue.pop()
            neighbors = _get_neighbors(coordinates, neighbor_type)
            for neighbor in neighbors:
                if result[neighbor] == 0 and _is_in_shape(neighbor, criterion.mode, params):
                    result[neighbor] = 1
                    queue.add(neighbor)
    else:
        img = img.astype(np.int32)
        queue = {(seed, img[seed])}
        while queue:
            coordinates, parent_val = queue.pop()
            neighbors = _get_neighbors(coordinates, neighbor_type)
            for neighbor in neighbors:
                if result[neighbor] == 0 and \
                    _is_in_shape(neighbor, criterion.mode, (params[0], parent_val)):
                    result[neighbor] = 1
                    queue.add((neighbor, img[neighbor]))
    return result
