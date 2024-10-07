import numpy as np
from numpy.typing import ArrayLike, NDArray
from common.histogram import img2hist


def _validate_img(img: NDArray) -> None:
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")


def otsu(img: ArrayLike) -> np.integer:
    """
    Takes a one-channel image and returns the threshold value that maximizes the variance between classes.
    @param img: grayscale image, a 2D numpy array with integer values
    @return: Otsu's threshold: a positive integer
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    _validate_img(img)

    info = np.iinfo(img.dtype)
    abs_max, abs_min = info.max, info.min

    hist, bins = img2hist(img).get_hist()

    norm_hist = hist / hist.sum()
    norm_hist_cum_sum = np.cumsum(norm_hist)
    values = np.arange(abs_min, abs_max + 1)
    mean_g = np.sum(values * norm_hist)
    mean_1 = np.cumsum(values * norm_hist) / np.where(norm_hist_cum_sum == 0, 1, norm_hist_cum_sum)
    norm_hist_cum_sum[-1] = 0
    inter_class_variance = norm_hist_cum_sum * (mean_g - mean_1) ** 2 / (1 - norm_hist_cum_sum)

    return np.argmax(inter_class_variance).astype(img.dtype)


class ThresholdType(int):
    BINARY = 0
    BINARY_INV = 1
    TRUNC = 2
    TRUNC_INV = 3
    TO_ZERO = 4
    TO_ZERO_INV = 5
    def __iter__(self):
        yield self.BINARY
        yield self.BINARY_INV
        yield self.TRUNC
        yield self.TRUNC_INV
        yield self.TO_ZERO
        yield self.TO_ZERO_INV


def apply_threshold(img: ArrayLike, threshold: int, t_type: ThresholdType,
                    min_v: int | None = None, max_v: int | None = None) -> NDArray:
    """
    Takes a value and a threshold and returns a binary value, according to type.
    @param img: a 2D numpy array with real numeric values
    @param threshold: threshold to be used
    @param t_type: type of thresholding to be used
    @param min_v: minimum value
    @param max_v: maximum value
    @return: binary value
    """
    @np.vectorize
    def _binary_t(value, t):
        return 1 if value > t else 0
    @np.vectorize
    def _binary_inv_t(value, t):
        return 0 if value > t else 1
    @np.vectorize
    def _trunc_t(value, t):
        return t if value > t else value
    @np.vectorize
    def _trunc_inv_t(value, t):
        return t if value < t else value
    @np.vectorize
    def _to_zero_t(value, t):
        return value if value >= t else 0
    @np.vectorize
    def _to_zero_inv_t(value, t):
        return value if value < t else 1

    if not isinstance(img, np.ndarray):
        img = np.array(img)
    _validate_img(img)
    if min_v is not None and min_v < 0:
        raise ValueError("min_v should be greater than or equal to 0")
    if max_v is not None and max_v < 0:
        raise ValueError("max_v should be greater than or equal to 0")

    match t_type:
        case ThresholdType.BINARY:
            if min_v is None and max_v is None:
                return _binary_t(img, threshold)
            elif min_v is None or max_v is None:
                raise ValueError("Both or none max_c and min_v should be None")
            if min_v > max_v:
                raise ValueError("max_v should be greater than min_v")
            return np.where(_binary_t(img, threshold, min_v, max_v), max_v, min_v)
        case ThresholdType.BINARY_INV:
            if min_v is None and max_v is None:
                return _binary_inv_t(img, threshold)
            elif min_v is None or max_v is None:
                raise ValueError("Both or none max_c and min_v should be None")
            if min_v > max_v:
                raise ValueError("min_v should be lower than max_v")
            return np.where(_binary_inv_t(img, threshold, min_v, max_v), max_v, min_v)
        case ThresholdType.TRUNC:
            if min_v is not None or max_v is not None:
                raise ValueError("min_v and max_v should be None when threshold is not binary")
            return _trunc_t(img, threshold)
        case ThresholdType.TRUNC_INV:
            if min_v is not None or max_v is not None:
                raise ValueError("min_v and max_v should be None when threshold is not binary")
            return _trunc_inv_t(img, threshold)
        case ThresholdType.TO_ZERO:
            if min_v is not None or max_v is not None:
                raise ValueError("min_v and max_v should be None when threshold is not binary")
            return _to_zero_t(img, threshold)
        case ThresholdType.TO_ZERO_INV:
            if min_v is not None or max_v is not None:
                raise ValueError("min_v and max_v should be None when threshold is not binary")
            return _to_zero_inv_t(img, threshold)
        case _:
            raise ValueError(f"Invalid threshold type: {t_type}")