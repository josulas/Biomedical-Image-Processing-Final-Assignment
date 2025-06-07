"""Texture analysis using co-occurrence matrices."""


from typing import Union
import functools

import numpy as np
from numpy.typing import ArrayLike


Real = Union[int, float]
MetricType = Union[Real, None]

class CoOccurrenceMatrixMetrics:
    """
    Class to store the metrics of a co-occurrence matrix.
    It contains the expected value, variance, energy, entropy, correlation,
    inverse difference moment, contrast, and cluster shade.
    It also contains a flag to indicate if any of the metrics is None.
    """
    def __init__(self):
        self.expected_value: MetricType = None
        self.variance: MetricType = None
        self.energy: MetricType = None
        self.entropy: MetricType = None
        self.correlation: MetricType = None
        self.inverse_difference_moment: MetricType = None
        self.contrast: MetricType = None
        self.cluster_shade: MetricType = None
        self.any_is_none: MetricType = True

    def __iter__(self):
        yield self.expected_value
        yield self.variance
        yield self.energy
        yield self.correlation
        yield self.inverse_difference_moment
        yield self.contrast
        yield self.cluster_shade
        yield self.any_is_none


class CoOccurrence:
    """
    Class to calculate the co-occurrence matrix of a single channel image.
    The co-occurrence matrix is calculated for a given direction.
    The direction can be horizontal, vertical, or diagonal (45 or 135 degrees).
    The class also calculates various metrics from the co-occurrence matrix,
    such as expected value, variance, energy, entropy, correlation,
    inverse difference moment, contrast, and cluster shade.
    The metrics can be calculated on demand or at initialization.
    """
    DIRECTION_HORIZONTAL = 0
    DIRECTION_DIAGONAL_45 = 1
    DIRECTION_VERTICAL = 2
    DIRECTION_DIAGONAL_135 = 3
    VALID_DIRECTIONS = [DIRECTION_HORIZONTAL,
                        DIRECTION_DIAGONAL_45,
                        DIRECTION_VERTICAL,
                        DIRECTION_DIAGONAL_135]

    def __init__(self, img: ArrayLike, direction: int, calculate_metrics: bool = True):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
            raise TypeError("'img' must be a single channel image, i.e., \
                            a two dimensional array with integer values")
        if not isinstance(direction, int):
            raise TypeError("'direction' must be an integer")
        if direction not in [0, 1, 2, 3]:
            raise ValueError("'direction' must be one of the following: 0, 1, 2, 3)")

        self.direction = direction
        self.min_val = np.min(img)
        self.max_val = np.max(img)
        self.num_levels = int(self.max_val - self.min_val + 1)
        self.co_occurrence_matrix = self.calculate_co_occurrence_matrix(img)
        self.metrics = CoOccurrenceMatrixMetrics()
        if calculate_metrics:
            self.calculate_metrics()

    def __str__(self):
        directions = {
            0: "Horizontal",
            1: "Diagonal 45",
            2: "Vertical",
            3: "Diagonal 135"
        }
        return f"Co-occurrence matrix for direction \
            {directions[self.direction]}: {str(self.co_occurrence_matrix)}"

    def __repr__(self):
        if self.metrics.any_is_none:
            self.calculate_metrics()
        title = str(self)
        lines = [title,
                 f"Expected value: {self.metrics.expected_value}",
                 f"Variance: {self.metrics.variance}",
                 f"Energy: {self.metrics.energy}",
                 f"Entropy: {self.metrics.entropy}",
                 f"Correlation: {self.metrics.correlation}",
                 f"Inverse difference moment: {self.metrics.inverse_difference_moment}",
                 f"Contrast: {self.metrics.contrast}",
                 f"Cluster shade: {self.metrics.cluster_shade}"]
        return "\n".join(lines)

    def calculate_co_occurrence_matrix(self, img):
        """
        Calculates the co-occurrence matrix for the given image and direction.
        :param img: 2D numpy array representing the image
        :return: 2D numpy array representing the co-occurrence matrix
        """
        working_img = img - self.min_val
        dim_y, dim_x = working_img.shape
        def _get_start_stop(direction):
            if direction == self.DIRECTION_HORIZONTAL:
                return 0, dim_y, 0, dim_x - 1
            elif direction == self.DIRECTION_DIAGONAL_45:
                return 1, dim_y, 0, dim_x - 1
            elif direction == self.DIRECTION_VERTICAL:
                return 1, dim_y, 0, dim_x
            elif direction == self.DIRECTION_DIAGONAL_135:
                return 1, dim_y, 1, dim_x
        def _get_next(k, l):
            if self.direction == self.DIRECTION_HORIZONTAL:
                return working_img[k, l + 1]
            elif self.direction == self.DIRECTION_DIAGONAL_45:
                return working_img[k - 1, l + 1]
            elif self.direction == self.DIRECTION_VERTICAL:
                return working_img[k - 1, l]
            elif self.direction == self.DIRECTION_DIAGONAL_135:
                return working_img[k - 1, l - 1]
        img_out = np.zeros((self.num_levels, self.num_levels), dtype=np.int_)
        start_i, stop_i, start_j, stop_j = _get_start_stop(self.direction)
        for i in range(start_i, stop_i):
            for j in range(start_j, stop_j):
                value1 = working_img[i, j]
                value2 = _get_next(i, j)
                img_out[value1, value2] += 1
        img_out = img_out + img_out.T
        img_out = img_out / np.sum(img_out)
        return img_out

    def get_co_occurrence_matrix(self):
        """
        Returns the co-occurrence matrix.
        :return: 2D numpy array representing the co-occurrence matrix
        """
        return self.co_occurrence_matrix

    @staticmethod
    def update_any_is_none(func):
        """
        Decorator to update the anyIsNone flag in the metrics after calling a metric function.
        It checks if any of the metrics is None and updates the flag accordingly.
        """
        @functools.wraps(func)
        def decorated(inst, *args, **kwargs):
            result = func(inst, *args, **kwargs)
            is_none = False
            for metric in inst.metrics:
                if metric is None:
                    is_none = True
                    break
            inst.metrics.anyIsNone = is_none
            return result
        return decorated

    @update_any_is_none
    def expected_value(self):
        """
        Calculates the expected value of the co-occurrence matrix.
        :return: float representing the expected value
        """
        return np.sum(self.co_occurrence_matrix * np.arange(self.num_levels))

    def get_expected_value(self):
        """
        Returns the expected value of the co-occurrence matrix.
        If the expected value is not calculated yet, it calculates it.
        :return: float representing the expected value
        """
        if self.metrics.expected_value is None:
            self.metrics.expected_value = self.expected_value()
        return self.metrics.expected_value

    @update_any_is_none
    def variance(self):
        """
        Calculates the variance of the co-occurrence matrix.
        :return: float representing the variance
        """
        if self.metrics.expected_value is None:
            mu = self.expected_value()
        else:
            mu = self.metrics.expected_value
        return np.sum(self.co_occurrence_matrix * (np.arange(self.num_levels) - mu) ** 2)

    def get_variance(self):
        """
        Returns the variance of the co-occurrence matrix.
        If the variance is not calculated yet, it calculates it.
        :return: float representing the variance
        """
        if self.metrics.variance is None:
            self.metrics.variance = self.variance()
        return self.metrics.variance

    @update_any_is_none
    def energy(self):
        """
        Calculates the energy of the co-occurrence matrix.
        :return: float representing the energy
        """
        return np.sum(self.co_occurrence_matrix ** 2)

    def get_energy(self):
        """
        Returns the energy of the co-occurrence matrix.
        If the energy is not calculated yet, it calculates it.
        :return: float representing the energy
        """
        if self.metrics.energy is None:
            self.metrics.energy = self.energy()
        return self.metrics.energy

    @update_any_is_none
    def entropy(self):
        """
        Calculates the entropy of the co-occurrence matrix.
        :return: float representing the entropy
        """
        elements = self.co_occurrence_matrix[self.co_occurrence_matrix != 0].flatten()
        return -np.sum(elements * np.log2(elements))

    def get_entropy(self):
        """
        Returns the entropy of the co-occurrence matrix.
        If the entropy is not calculated yet, it calculates it.
        :return: float representing the entropy"""
        if self.metrics.entropy is None:
            self.metrics.entropy = self.entropy()
        return self.metrics.entropy

    @update_any_is_none
    def correlation(self):
        """
        Calculates the correlation of the co-occurrence matrix.
        :return: float representing the correlation
        """
        dim = self.num_levels
        if self.metrics.expected_value is None:
            mu = self.expected_value()
        else:
            mu = self.metrics.expected_value
        if self.metrics.variance is None:
            sigma = self.variance()
        else:
            sigma = self.metrics.variance
        if sigma == 0:
            return 0
        correlation_value = 0
        for i in range(dim):
            for j in range(dim):
                correlation_value += (i - mu) * (j - mu) * self.co_occurrence_matrix[i, j]
        return correlation_value / (sigma ** 2)

    def get_correlation(self):
        """
        Returns the correlation of the co-occurrence matrix.
        If the correlation is not calculated yet, it calculates it.
        :return: float representing the correlation
        """
        if self.metrics.correlation is None:
            self.metrics.correlation = self.correlation()
        return self.metrics.correlation

    @update_any_is_none
    def inverse_difference_moment(self):
        """
        Calculates the inverse difference moment of the co-occurrence matrix.
        :return: float representing the inverse difference moment
        """
        dim = self.num_levels
        inverse_difference_moment = 0
        for i in range(dim):
            for j in range(dim):
                inverse_difference_moment += \
                    1 / (1 + (i - j) ** 2) * self.co_occurrence_matrix[i, j]
        return inverse_difference_moment

    def get_inverse_difference_moment(self):
        """
        Returns the inverse difference moment of the co-occurrence matrix.
        If the inverse difference moment is not calculated yet, it calculates it.
        :return: float representing the inverse difference moment
        """
        if self.metrics.inverse_difference_moment is None:
            self.metrics.inverse_difference_moment = self.inverse_difference_moment()
        return self.metrics.inverse_difference_moment

    @update_any_is_none
    def contrast(self):
        """
        Calculates the contrast of the co-occurrence matrix.
        :return: float representing the contrast
        """
        dim = self.num_levels
        contrast_value = 0
        for i in range(dim):
            for j in range(dim):
                contrast_value += (i - j) ** 2 * self.co_occurrence_matrix[i, j]
        return contrast_value

    def get_contrast(self):
        """
        Returns the contrast of the co-occurrence matrix.
        If the contrast is not calculated yet, it calculates it.
        :return: float representing the contrast
        """
        if self.metrics.contrast is None:
            self.metrics.contrast = self.contrast()
        return self.metrics.contrast

    @update_any_is_none
    def cluster_shade(self):
        """
        Calculates the cluster shade of the co-occurrence matrix.
        :return: float representing the cluster shade
        """
        dim = self.num_levels
        if self.metrics.expected_value is None:
            mu = self.expected_value()
        else:
            mu = self.metrics.expected_value
        cluster_shade = 0
        for i in range(dim):
            for j in range(dim):
                cluster_shade += \
                    ((i - mu) + (j - mu)) ** 3 * self.co_occurrence_matrix[i, j]
        return cluster_shade

    def get_cluster_shade(self):
        """
        Returns the cluster shade of the co-occurrence matrix.
        If the cluster shade is not calculated yet, it calculates it."""
        if self.metrics.cluster_shade is None:
            self.metrics.cluster_shade = self.cluster_shade()
        return self.metrics.cluster_shade

    @update_any_is_none
    def calculate_metrics(self):
        """
        Calculates all the metrics of the co-occurrence matrix.
        It updates the metrics object with the calculated values.
        """
        self.metrics.expected_value = self.expected_value()
        self.metrics.variance = self.variance()
        self.metrics.energy = self.energy()
        self.metrics.entropy = self.entropy()
        self.metrics.correlation = self.correlation()
        self.metrics.inverse_difference_moment = self.inverse_difference_moment()
        self.metrics.contrast = self.contrast()
        self.metrics.cluster_shade = self.cluster_shade()
        return self.metrics

    def get_metrics(self):
        """
        Returns the metrics of the co-occurrence matrix.
        If the metrics are not calculated yet, it calculates them.
        :return: CoOccurrenceMatrixMetrics object containing the metrics
        """
        if self.metrics.any_is_none:
            self.calculate_metrics()
        return self.metrics
