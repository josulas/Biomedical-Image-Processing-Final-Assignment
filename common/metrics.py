import numpy as np
from numpy.typing import ArrayLike
from typing import Union


numpy_real = Union[np.floating, np.integer]


class SimilarityIndex:
    DICE = 0
    JACCARD = 1
    def __iter__(self):
        yield self.DICE
        yield self.JACCARD


def similarity(img1: ArrayLike, img2: ArrayLike, mode: int) -> float:
    """
    Return the similarity index between two images
    @param img1: binary image, a 2D numpy array with integer values
    @param img2: binary image, a 2D numpy array with integer values
    @param mode: see SimilarityIndex class
    @return: index, ||A∩B||/||A∪B|| (float value)
    """
    if not isinstance(img1, np.ndarray):
        img1 = np.array(img1)
    if not isinstance(img2, np.ndarray):
        img2 = np.array(img2)
    if not issubclass(img1.dtype.type, np.integer) or not img1.ndim == 2:
        raise ValueError("img1 must be an integer 2D array")
    if not issubclass(img2.dtype.type, np.integer) or not img2.ndim == 2:
        raise ValueError("img2 must be an integer 2D array")
    if not np.array(((img1 == 0) | (img1 == 1))).all():
        raise ValueError("img1 must have ones and zeroes as values")
    if not np.array(((img2 == 0) | (img2 == 1))).all():
        raise ValueError("img2 must have ones and zeroes as values")
    if not img1.shape == img2.shape:
        raise ValueError("img1 and img2 must have the same shape")
    if not mode in list(SimilarityIndex()):
        raise ValueError(f"mode must be either {SimilarityIndex.DICE} or {SimilarityIndex.JACCARD}")

    intersection_count = np.sum(img1 * img2)
    img1_count = np.sum(img1)
    img2_count = np.sum(img2)

    if mode == SimilarityIndex.JACCARD:
        return intersection_count / (img1_count + img2_count - intersection_count)
    elif mode == SimilarityIndex.DICE:
        return 2 * intersection_count / (img1_count + img2_count)


def _are_comparable(tensor1: ArrayLike, tensor2: ArrayLike) -> None:
    """
    Checks for comparability among two tensors and informs the user about the actual issue
    :param tensor1: tensor, a ND Array containing real values
    :param tensor2: tensor, a ND Array containing real values
    :return: None, an exception will be raised if tensor1 and tensor2 are not comparable
    """
    if not isinstance(tensor1, np.ndarray):
        tensor1 = np.array(tensor1)
    if not isinstance(tensor2, np.ndarray):
        tensor2 = np.array(tensor2)
    if not issubclass(tensor1.dtype.type, numpy_real):
        raise ValueError("tensor1 must be a numpy array with real values")
    if not issubclass(tensor2.dtype.type, numpy_real):
        raise ValueError("tensor2 must be a numpy array with real values")
    if not tensor1.shape == tensor2.shape:
        raise ValueError("tensor1 and tensor2 must have the same shape")


def l_norm(tensor1: ArrayLike, tensor2: ArrayLike, l: int) -> np.float64:
    """
    Return the LNorm between two tensors
    @param tensor1: tensor, a ND array containing real values
    @param tensor2: tensor, a ND array containing real values
    @param l: length of the tensor
    @return: LNorm between tensor1 and tensor2
    """
    _are_comparable(tensor1, tensor2)
    if l < 1:
        raise ValueError("L must be a positive integer")
    return np.sum(np.abs(tensor1 - tensor2) ** l) ** (1 / l)


def mse(tensor1: ArrayLike, tensor2: ArrayLike) -> np.float64:
    """
    Return the mean squared error between two tensors
    @param tensor1: tensor, a ND array containing real values
    @param tensor2: tensor, a ND array containing real values
    return MSE(tensor1, tensor2)
    """
    _are_comparable(tensor1, tensor2)
    return np.mean((tensor1 - tensor2) ** 2)


def rms(tensor1: ArrayLike, tensor2: ArrayLike) -> np.float64:
    """
    Return the root mean squared error between two tensors
    @param tensor1: tensor, a ND array containing real values
    @param tensor2: tensor, a ND array containing real values
    return RMS(tensor1, tensor2)
    """
    return np.sqrt(mse(tensor1, tensor2))


# noinspection SpellCheckingInspection
def psnr(tensor1: ArrayLike, tensor2: ArrayLike) -> np.float64:
    """
    Return the peak signal-to-noise ratio between two tensors
    @param tensor1: tensor, a ND array containing real values
    @param tensor2: tensor, a ND array containing real values
    return pSNR(tensor1, tensor2)
    """
    _are_comparable(tensor1, tensor2)
    return 10 * np.log10(1 / mse(tensor1, tensor2))