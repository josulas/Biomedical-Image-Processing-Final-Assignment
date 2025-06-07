"""
This module provides functions to compute bounding rectangles 
for connected components in a binary image,
and to draw these rectangles on a color image.
"""


import numpy as np
from numpy.typing import ArrayLike
from libraries.segmentation.connected_components import _bounded_coords


class Rectangle(tuple):
    """Rectamgle class representing a bounding box with upper left and lower right corners."""
    def __new__(cls, tl: tuple[int, int], br: tuple[int, int]):
        if not isinstance(tl, tuple) or not isinstance(br, tuple) or len(tl) != 2 or len(br) != 2:
            raise ValueError("tl and br must be tuples of length 2")
        if not all(isinstance(i, int) for i in tl) or not all(isinstance(i, int) for i in br):
            raise ValueError("tl and br must be tuples of integers")
        if not all(i >= 0 for i in tl) or not all(i >= 0 for i in br):
            raise ValueError("tl and br must be tuples of non-negative integers")
        if tl[0] > br[0] or tl[1] > br[1]:
            raise ValueError("tl's components must be smaller than br's")
        return super().__new__(cls, (tl, br))


def bounding_rects(img: ArrayLike) -> list[Rectangle]:
    """
    Return the bounding boxes (upper left corner and lower right corner) of the connected components
    as a list of tuples
    @param img: binary image, a 2D numpy array with integer values
    @return: list of tuples with the bounding boxes (upper left corner and lower right corner)
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    working_img = img.copy()
    height, width = img.shape
    rects = []
    for i in range(width):
        for j in range(height):
            if working_img[j, i]:
                rect_ul_x = i
                rect_ul_y = j
                rect_br_x = i
                rect_br_y = j
                working_img[j, i] = 0
                queue = [(i, j)]
                while queue:
                    x, y = queue.pop(0)
                    x_min, x_max, y_min, y_max = _bounded_coords(x, y, width, height)
                    for k in range(x_min, x_max + 1):
                        for l in range(y_min, y_max + 1):
                            if working_img[l, k]:
                                working_img[l, k] = 0
                                if k > rect_br_x:
                                    rect_br_x = k
                                if k < rect_ul_x:
                                    rect_ul_x = k
                                if l > rect_br_y:
                                    rect_br_y = l
                                if l < rect_ul_y:
                                    rect_ul_y = l
                                queue.append((k, l))
                rects.append(Rectangle((rect_ul_x, rect_ul_y), (rect_br_x + 1, rect_br_y + 1)))
    return rects


def draw_rects(img: ArrayLike, rects: list[Rectangle],
               color: tuple[int | None, int | None, int | None] | \
              tuple[int | None, int | None, int | None, float | None] | None = None,
               thickness: int | None = None) -> None:
    """
    Draw rectangles
    @param img: image to be drawn on
    @param rects: list of tuples with the bounding boxes (upper left corner and lower right corner)
    @param color: color of the rectangles, a tuple of 3 integers 
                  (plus an additional float) between 0 and 255,
                  or None values, or None
    @param thickness: thickness of the rectangles, a positive integer
    @return: None

    """
    if not isinstance(img, np.ndarray):
        try:
            img = np.array(img).astype(np.uint8)
        except ValueError as e:
            raise ValueError("Image must be convertible to \
                             a numpy array with uint8 values") from e
    if not img.dtype == np.uint8 or not img.ndim == 3 or not img.shape[2] == 3:
        raise ValueError("Image must be an integer 3D array representing a color 24-bit image")
    if thickness is None:
        thickness = 1
    elif thickness < 0:
        raise ValueError("Thickness must be a positive integer")
    if color is not None and (sum([i is not None and (i < 0 or i > 255) for i in color[:3]]) > 0 or
                              (len(color) == 4 and \
                               color[3] is not None and not 0 <= color[3] <= 1)):
        raise ValueError("Color must be a tuple of 3 integers between 0 and 255 \
        (plus a float between 0 and 1 optionally) or None values, or None itself")

    height, width = img.shape[:2]
    if color is not None and not sum([i is not None for i in color[:3]]):
        color = None
    if color is not None and len(color) == 4:
        alpha = color[3]
    else:
        alpha = None

    def draw_rect(coords):
        def fill_rect(x_min, x_max, y_min, y_max):
            if color is not None:
                if color[0] is not None:
                    if alpha is None:
                        img[y_min:y_max + 1, x_min:x_max + 1, 0] = color[0]
                    else:
                        img[y_min:y_max + 1, x_min:x_max + 1, 0] = \
                         (1 - alpha) * img[y_min:y_max + 1, x_min:x_max + 1, 0] + alpha * color[0]
                if color[1] is not None:
                    if alpha is None:
                        img[y_min:y_max + 1, x_min:x_max + 1, 1] = color[1]
                    else:
                        img[y_min:y_max + 1, x_min:x_max + 1, 1] = \
                         (1 - alpha) * img[y_min:y_max + 1, x_min:x_max + 1, 1] + alpha * color[1]
                if color[2] is not None:
                    if alpha is None:
                        img[y_min:y_max + 1, x_min:x_max + 1, 2] = color[2]
                    else:
                        img[y_min:y_max + 1, x_min:x_max + 1, 2] = \
                         (1 - alpha) * img[y_min:y_max + 1, x_min:x_max + 1, 2] + alpha * color[2]
            else:
                img[y_min:y_max + 1, x_min:x_max + 1] = 255 - img[y_min:y_max + 1, x_min:x_max + 1]

        l_x, t_y = coords[0]
        r_x, b_y = coords[1]
        if not 0 <= l_x < r_x <= width or not 0 <= t_y < b_y <= height:
            raise ValueError("Rectangles' Coordinates are out of image bounds or not valid")
        extend_1 = thickness // 2 # Inwards extend
        extend_2 = (thickness - 1) // 2 # Outwards extend
        # Top line
        fill_rect(max(l_x - extend_2, 0), r_x - extend_1 - 1,
                    max(t_y - extend_2, 0), min(t_y + extend_1, b_y))
        # Bottom line
        fill_rect(l_x + extend_1 + 1, min(r_x + extend_2, width - 1),
                    max(b_y - extend_1, t_y), min(b_y + extend_2, height - 1))
        # Left line
        fill_rect(max(l_x - extend_2, 0), min(l_x + extend_1, r_x),
                    t_y +  extend_1 + 1, min(b_y + extend_2, height - 1))
        # Right line
        fill_rect(max(r_x - extend_1, l_x), min(r_x + extend_2, width - 1),
                    max(t_y - extend_2, 0), b_y - extend_1 - 1)
    for rect in rects:
        draw_rect(rect)


def intersection_over_union(rect1: Rectangle, rect2: Rectangle):
    """Calculates IoU for two rectangles."""
    tl1, br1 = rect1
    tl2, br2 = rect2
    if min(br1[0], br2[0]) < max(tl1[0], tl2[0]) or min(br1[1], br2[1]) < max(tl1[1], tl2[1]):
        return 0

    area1 = (br1[0] - tl1[0]) * (br1[1] - tl1[1])
    area2 = (br2[0] - tl2[0]) * (br2[1] - tl2[1])
    intersection = \
        (min(br1[0], br2[0]) - max(tl1[0], tl2[0])) * (min(br1[1], br2[1]) - max(tl1[1], tl2[1]))
    union = area1 + area2 - intersection
    return intersection / union
