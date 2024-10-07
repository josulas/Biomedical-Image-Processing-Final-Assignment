from common.padding import *
from typing import Callable

def _validate_parameters(img: NDArray, structuring_element: NDArray, mode: str) -> None:
    """
    Validate the input parameters for mathematical morphology.
    :param img: a 2D binary array, composed of 1s and 0s
    :param structuring_element: a 2D array, can have NaN values
    :param mode: 'same' or 'valid'
    :return: None
    """
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    if not np.array(((img == 0) | (img == 1))).all():
        raise ValueError("Image must have ones and zeroes as values")
    if not isinstance(structuring_element, np.ndarray):
        structuring_element = np.array(structuring_element)
    if not issubclass(structuring_element.dtype.type, np.integer) or not structuring_element.ndim == 2:
        raise ValueError("Structuring element must be an integer 2D array")
    if not structuring_element.shape[0] == structuring_element.shape[1]:
        raise NotImplementedError("Structuring element must be a square array for now")
    if not ((structuring_element == 0) | (structuring_element == 1) | np.isnan(structuring_element)).all():
        raise ValueError("Structuring element must have ones, zeroes and NaNs as values")
    if mode not in ['same', 'valid']:
        raise ValueError("Mode must be 'same' or 'valid'")


def fit(patch: NDArray, structuring_element: NDArray) -> int:
    """
    Performs the fit operation with a binary patch and a structuring element
    :param patch: binary 2D NDArray
    :param structuring_element: binary 2D NDArray
    :return: 1 if all the non-zero elements of the structuring elements are matched with non-zero elements in the patch
    """
    return 1 if np.sum(patch * structuring_element) == np.sum(structuring_element) else 0


def hit(patch: NDArray, structuring_element: NDArray) -> int:
    """
    Performs the hit operation with a binary patch and a structuring element
    :param patch: binary 2D NDArray
    :param structuring_element: binary 2D NDArray
    :return: 1 if at least one non-zero element of the structuring elements is matched with any non-zero element
    in the patch
    """
    return 1 if np.sum(patch * structuring_element) else 0


def _iterate_over_image(img: ArrayLike, structuring_element: NDArray, mode: str, operation: Callable) -> NDArray:
    """
    Performs a mathematical morphology operation on an image given a structuring element
    :param img: binary 2D NDArray
    :param structuring_element: binary 2D NDArray
    :param operation: either 'fit' or 'hit'
    :return: resultant 2D NDArray
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    _validate_parameters(img, structuring_element, mode)
    if operation not in (fit, hit):
        raise ValueError("Mode must be 'fit' or 'hit'")
    structuring_element = np.where(np.isnan(structuring_element), 0, structuring_element)
    el_height, el_width = structuring_element.shape
    el_dim = el_height # should be equal to el_width too
    if mode == 'same':
        working_img =  zero_pad_img(img, el_dim // 2)
    else:
        working_img =  img
    img_height, img_width = working_img.shape
    result = np.zeros((img_height - (el_height // 2) * 2, img_width - (el_height // 2) * 2), dtype=img.dtype)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = operation(working_img[i:i + el_height, j:j + el_width], structuring_element)
    return result


def erosion(img: ArrayLike, structuring_element: ArrayLike, mode: str = 'same') -> NDArray[np.integer]:
    """
    Performs the erosion operation over an Image
    @param img: ArrayLike containing integer values, only 1s and 0s
    @param structuring_element: ArrayLike containing integer values, only 1s, 0s and NaNs
    @param mode: Mode of operation, 'same' or 'valid'
    """
    return _iterate_over_image(img, structuring_element, mode, fit)


def dilation(img: ArrayLike, structuring_element: ArrayLike, mode: str = 'same') -> NDArray[np.integer]:
    """
    Performs the dilation operation over an Image
    @param img: ArrayLike containing integer values, only 1s and 0s
    @param structuring_element: ArrayLike containing integer values, only 1s, 0s and NaNs
    @param mode: Mode of operation, 'same' or 'valid'
    """
    return _iterate_over_image(img, structuring_element, mode, hit)


def opening(img: ArrayLike, structuring_element: ArrayLike, mode: str = 'same') -> NDArray[np.integer]:
    """
    Performs the opening operation over an Image: erosion followed by dilation
    @param img: ArrayLike containing integer values, only 1s and 0s
    @param structuring_element: ArrayLike containing integer values, only 1s, 0s and NaNs
    @param mode: Mode of operation, 'same' or 'valid'
    """
    return dilation(erosion(img, structuring_element, mode), structuring_element, mode)


def closing(img: ArrayLike, structuring_element: ArrayLike, mode: str = 'same') -> NDArray[np.integer]:
    """
    Performs the closing operation over an Image: dilation followed by erosion
    @param img: ArrayLike containing integer values, only 1s and 0s
    @param structuring_element: ArrayLike containing integer values, only 1s, 0s and NaNs
    @param mode: Mode of operation, 'same' or 'valid'
    """
    return erosion(dilation(img, structuring_element, mode), structuring_element, mode)