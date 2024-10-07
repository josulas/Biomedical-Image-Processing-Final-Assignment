import numpy as np
from numpy.typing import ArrayLike
import functools


class CoOccurrenceMatrixMetrics:
        def __init__(self):
            self.expectedValue = None
            self.variance = None
            self.energy = None
            self.entropy = None
            self.correlation = None
            self.inverseDifferenceMoment = None
            self.contrast = None
            self.clusterShade = None
            self.anyIsNone = True

        def __iter__(self):
            yield self.expectedValue
            yield self.variance
            yield self.energy
            yield self.correlation
            yield self.inverseDifferenceMoment
            yield self.contrast
            yield self.clusterShade
            yield self.anyIsNone


class CoOccurrence:
    DIRECTION_HORIZONTAL = 0
    DIRECTION_DIAGONAL_45 = 1
    DIRECTION_VERTICAL = 2
    DIRECTION_DIAGONAL_135 = 3
    VALID_DIRECTIONS = [DIRECTION_HORIZONTAL, DIRECTION_DIAGONAL_45, DIRECTION_VERTICAL, DIRECTION_DIAGONAL_135]

    def __init__(self, img: ArrayLike, direction: int, calculate_metrics: bool = True):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
            raise TypeError("'img' must be a single channel image, i.e., a two dimensional array with integer values")
        if not isinstance(direction, int):
            raise TypeError("'direction' must be an integer")
        if direction not in [0, 1, 2, 3]:
            raise ValueError("'direction' must be one of the following: 0, 1, 2, 3)")

        self.direction = direction
        self.minVal = np.min(img)
        self.maxVal = np.max(img)
        self.numLevels = int(self.maxVal - self.minVal + 1)
        self.coOccurrenceMatrix = self.calculate_co_occurrence_matrix(img)
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
        return f"Co-occurrence matrix for direction {directions[self.direction]}: {str(self.coOccurrenceMatrix)}"

    def __repr__(self):
        if self.metrics.anyIsNone:
            self.calculate_metrics()
        title = str(self)
        lines = [title, f"Expected value: {self.metrics.expectedValue}", f"Variance: {self.metrics.variance}",
                 f"Energy: {self.metrics.energy}", f"Entropy: {self.metrics.entropy}",
                 f"Correlation: {self.metrics.correlation}",
                 f"Inverse difference moment: {self.metrics.inverseDifferenceMoment}",
                 f"Contrast: {self.metrics.contrast}", f"Cluster shade: {self.metrics.clusterShade}"]
        return "\n".join(lines)

    def calculate_co_occurrence_matrix(self, img):
        working_img = img - self.minVal
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
        img_out = np.zeros((self.numLevels, self.numLevels), dtype=np.int_)
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
        return self.coOccurrenceMatrix

    @staticmethod
    def update_any_is_none(func):
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
        return np.sum(self.coOccurrenceMatrix * np.arange(self.numLevels))

    def get_expected_value(self):
        if self.metrics.expectedValue is None:
            self.metrics.expectedValue = self.expected_value()
        return self.metrics.expectedValue

    @update_any_is_none
    def variance(self):
        if self.metrics.expectedValue is None:
            mu = self.expected_value()
        else:
            mu = self.metrics.expectedValue
        return np.sum(self.coOccurrenceMatrix * (np.arange(self.numLevels) - mu) ** 2)

    def get_variance(self):
        if self.metrics.variance is None:
            self.metrics.variance = self.variance()
        return self.metrics.variance

    @update_any_is_none
    def energy(self):
        return np.sum(self.coOccurrenceMatrix ** 2)

    def get_energy(self):
        if self.metrics.energy is None:
            self.metrics.energy = self.energy()
        return self.metrics.energy

    @update_any_is_none
    def entropy(self):
        elements = self.coOccurrenceMatrix[self.coOccurrenceMatrix != 0].flatten()
        return -np.sum(elements * np.log2(elements))

    def get_entropy(self):
        if self.metrics.entropy is None:
            self.metrics.entropy = self.entropy()
        return self.metrics.entropy

    @update_any_is_none
    def correlation(self):
        dim = self.numLevels
        if self.metrics.expectedValue is None:
            mu = self.expected_value()
        else:
            mu = self.metrics.expectedValue
        if self.metrics.variance is None:
            sigma = self.variance()
        else:
            sigma = self.metrics.variance
        if sigma == 0:
            return 0
        correlation_value = 0
        for i in range(dim):
            for j in range(dim):
                correlation_value += (i - mu) * (j - mu) * self.coOccurrenceMatrix[i, j]
        return correlation_value / (sigma ** 2)

    def get_correlation(self):
        if self.metrics.correlation is None:
            self.metrics.correlation = self.correlation()
        return self.metrics.correlation

    @update_any_is_none
    def inverse_difference_moment(self):
        dim = self.numLevels
        inverse_difference_moment = 0
        for i in range(dim):
            for j in range(dim):
                inverse_difference_moment += 1 / (1 + (i - j) ** 2) * self.coOccurrenceMatrix[i, j]
        return inverse_difference_moment

    def get_inverse_difference_moment(self):
        if self.metrics.inverseDifferenceMoment is None:
            self.metrics.inverseDifferenceMoment = self.inverse_difference_moment()
        return self.metrics.inverseDifferenceMoment

    @update_any_is_none
    def contrast(self):
        dim = self.numLevels
        contrast_value = 0
        for i in range(dim):
            for j in range(dim):
                contrast_value += (i - j) ** 2 * self.coOccurrenceMatrix[i, j]
        return contrast_value

    def get_contrast(self):
        if self.metrics.contrast is None:
            self.metrics.contrast = self.contrast()
        return self.metrics.contrast

    @update_any_is_none
    def cluster_shade(self):
        dim = self.numLevels
        if self.metrics.expectedValue is None:
            mu = self.expected_value()
        else:
            mu = self.metrics.expectedValue
        cluster_shade = 0
        for i in range(dim):
            for j in range(dim):
                cluster_shade += ((i - mu) + (j - mu)) ** 3 * self.coOccurrenceMatrix[i, j]
        return cluster_shade

    def get_cluster_shade(self):
        if self.metrics.clusterShade is None:
            self.metrics.clusterShade = self.cluster_shade()
        return self.metrics.clusterShade

    @update_any_is_none
    def calculate_metrics(self):
        self.metrics.expectedValue = self.expected_value()
        self.metrics.variance = self.variance()
        self.metrics.energy = self.energy()
        self.metrics.entropy = self.entropy()
        self.metrics.correlation = self.correlation()
        self.metrics.inverseDifferenceMoment = self.inverse_difference_moment()
        self.metrics.contrast = self.contrast()
        self.metrics.clusterShade = self.cluster_shade()
        return self.metrics

    def get_metrics(self):
        if self.metrics.anyIsNone:
            self.calculate_metrics()
        return self.metrics