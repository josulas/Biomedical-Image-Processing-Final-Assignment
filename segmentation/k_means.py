import numpy as np
from numpy.typing import ArrayLike, NDArray


class KmeansTermOpt(int):
    """
    Class to store the termination option of the Kmeans algorithm.
    """
    EPS = 0
    MAX_ITER = 1
    BOTH = 2
    def __iter__(self):
        yield self.EPS
        yield self.MAX_ITER
        yield self.BOTH


class KmeansTermCrit:
    """
    Class to store the termination criteria of the Kmeans algorithm.
    """
    def __init__(self, opt: int, max_iter: int = 10, epsilon: float = 0.2):
        self.opt = opt
        self.max_iter = max_iter
        self.epsilon = epsilon


class KmeansFlags(int):
    """
    Class to store the flags of the Kmeans algorithm.
    """
    RANDOM_CENTERS = 0
    KMEANS_PP_CENTERS = 1
    CUSTOM_CENTERS_RANDOM = 2
    CUSTOM_CENTERS_PP = 3
    def __iter__(self):
        yield self.RANDOM_CENTERS
        yield self.KMEANS_PP_CENTERS
        yield self.CUSTOM_CENTERS_RANDOM
        yield self.CUSTOM_CENTERS_PP


def kmeans(arr: ArrayLike, k: int = 2, best_labels: ArrayLike | None = None,
           criteria: KmeansTermCrit = KmeansTermCrit(KmeansTermOpt.BOTH, 10, 1e-6),
           attempts: int = 10, flags: int = KmeansFlags.RANDOM_CENTERS) -> tuple[float, NDArray, NDArray]:
    """
    Takes a numeric array and returns the centerness score, the labels and the centers obtained by the Kmeans
    algorithm.
    @param arr: array to be clustered
    @param k: number of clusters, a positive integer
    @param best_labels: best initial label given to the algorithm, a tuple numbers
    @param criteria: termination criteria of the Kmeans algorithm, a positive integer
    @param attempts: number of attempts to find the best solution, a positive integer
    @param flags: indicate how to initialize the centers, a positive integer
    @return: compactness score, labels and centers obtained by the Kmeans algorithm
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not issubclass(arr.dtype.type, (np.integer, np.floating)) or len(arr.shape) > 2:
        raise ValueError("array must consist on a list of scalars or vectors")
    arr = arr.astype(np.float64)
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if best_labels is not None:
        if flags == KmeansFlags.RANDOM_CENTERS or flags == KmeansFlags.KMEANS_PP_CENTERS:
            raise ValueError("bestLabels must be None when flags is RANDOM_CENTERS or KMEANS_PP_CENTERS")
        if not isinstance(best_labels, np.ndarray):
            best_labels = np.array(best_labels)
        if not issubclass(best_labels.dtype.type, (np.integer, np.floating)):
            raise ValueError("bestLabels must be a list of scalars or vectors")
        if best_labels.shape[0] != k:
            raise ValueError("bestLabels must have the same length as K")
        if len(best_labels.shape) > 2:
            raise ValueError("bestLabels must be a list of scalars or vectors")
        if len(arr.shape) != len(best_labels.shape) or best_labels.shape[1] != arr.shape[1]:
            raise ValueError("bestLabels elements and arr elements must have the same dimension")
    if attempts <= 0:
        raise ValueError("attempts must be a positive integer")

    def _random_init() -> NDArray:
        return arr[np.random.choice(arr.shape[0], k, False)]

    def _kmeans_pp_init() -> NDArray:
        centers_pp_init = np.zeros((k, arr.shape[1])) # if len(arr.shape) == 2 else np.zeros(K)
        distances_pp_init = np.ones((k, arr.shape[0])) * np.inf
        centers_pp_init[0] = arr[np.random.choice(arr.shape[0])]
        for center_index in range(1, k):
            for created_center_index in range(center_index):
                distances_pp_init[created_center_index] = np.sum((arr - centers_pp_init[created_center_index]) ** 2,
                                                                 axis = 1)
            distribution = np.min(distances_pp_init, axis=0)
            distribution /= np.sum(distribution)
            centers_pp_init[center_index] = arr[np.random.choice(arr.shape[0], p=distribution)]
        return centers_pp_init

    def _k_means_step(centers_array):
        distances_array = np.zeros((k, arr.shape[0]))
        for j in range(k):
            distances_array[j] = np.sum((arr - centers_array[j]) ** 2, axis=1)
        labels_array = np.argmin(distances_array, axis=0)
        for j in range(k):
            centers_array[j] = np.mean(arr[labels_array == j], axis=0) if len(arr[labels_array == j]) \
                else centers_array[j]
        return distances_array, labels_array

    if flags == KmeansFlags.RANDOM_CENTERS or flags == KmeansFlags.CUSTOM_CENTERS_RANDOM:
        init_centers = _random_init
    elif flags == KmeansFlags.KMEANS_PP_CENTERS or flags == KmeansFlags.CUSTOM_CENTERS_PP:
        init_centers = _kmeans_pp_init
    else:
        raise ValueError(f"flags must be RANDOM_CENTERS: {KmeansFlags.RANDOM_CENTERS}, "
                         f"KMEANS_PP_CENTERS: {KmeansFlags.KMEANS_PP_CENTERS}, "
                         f"CUSTOM_CENTERS_RANDOM: {KmeansFlags.CUSTOM_CENTERS_RANDOM}, "
                         f"CUSTOM_CENTERS_PP: {KmeansFlags.CUSTOM_CENTERS_PP}, got {flags} instead.")

    if criteria.opt == KmeansTermOpt.EPS:
        termination = lambda c, pc, ii: np.linalg.norm(c - pc) < criteria.epsilon
    elif criteria.opt == KmeansTermOpt.MAX_ITER:
        termination = lambda c, pc, ii: ii >= criteria.max_iter
    elif criteria.opt == KmeansTermOpt.BOTH:
        termination = lambda c, pc, ii: np.linalg.norm(c - pc) < criteria.epsilon or ii >= criteria.max_iter
    else:
        raise ValueError(f"criteria.opt must be EPS: {KmeansTermOpt.EPS}, \
        MAX_ITER: {KmeansTermOpt.MAX_ITER}, BOTH: {KmeansTermOpt.BOTH}, got {criteria.opt} instead.")

    labels_end = np.zeros(arr.shape[0], dtype=np.int_)
    compactness_end = np.inf
    centers_end = np.zeros((k, arr.shape[1]), dtype=np.float64)

    for i in range(attempts):
        if not i and flags in [KmeansFlags.CUSTOM_CENTERS_PP, KmeansFlags.CUSTOM_CENTERS_RANDOM]:
            centers = np.array(best_labels)
        else:
            centers = init_centers()
        # Until convergence
        iteration_index = 0
        previous_centers = centers.copy()
        while True:
            distances, labels = _k_means_step(centers)
            if termination(centers, previous_centers, iteration_index):
                break
            iteration_index += 1
        compactness = np.sum(np.min(distances, axis=0))
        if compactness < compactness_end:
            compactness_end = compactness
            centers_end = centers
            labels_end = labels
    return compactness_end, labels_end, centers_end