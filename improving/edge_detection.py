from filtering import *
from numbers import Real
import cv2


def _validate_parameters(img: NDArray, mode: str, smooth: bool, sigma: Real) -> None:
    if mode not in ['same', 'valid']:
        raise ValueError("Mode must be 'same' or 'valid'")
    if not issubclass(img.dtype.type, numpy_real):
        raise ValueError(f"Image values must be a numpy real numeric type, got {img.dtype.type} instead")
    if not img.ndim == 2:
        raise ValueError("Image must be a 2D array")
    if not smooth and sigma is not None:
        raise ValueError("Sigma must be None if smooth is False")
    if sigma is not None and (not isinstance(sigma, Real) or sigma <= 0):
        raise ValueError(f"Sigma must be a positive real number, got {sigma} instead")


def _smooth_image(img: NDArray, sigma: Real) -> NDArray:
    if sigma is None:
        sigma = 1
    kernel_size = int(np.ceil(sigma * 6) + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    gaussian_filter = np.dot(cv2.getGaussianKernel(kernel_size, sigma),
                             np.transpose(cv2.getGaussianKernel(kernel_size, sigma)))
    return conv2d(img, gaussian_filter)


def zero_crossing(img: ArrayLike,
                  threshold: Real | None = None,
                  mode: str = 'same', smooth: bool = True, sigma: None | Real = None) -> NDArray[np.uint8]:
    """
    Computes the image borders based on the Zero Crossing algorithm
    @param img: input image
    @param threshold: threshold value, default None
    @param mode: same or valid, default same
    @param smooth: smooth the image before computing the borders, default True
    @param sigma: sigma value, default None
    @return: output image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    _validate_parameters(img, mode, smooth, sigma)
    if threshold is not None and not isinstance(threshold, Real):
        raise ValueError(f"Threshold must be a real numeric type, got {threshold} instead")

    img = img.astype(np.float64)
    if mode == 'same':
        img = extension_pad_img(img, 1)
    if smooth:
       img = _smooth_image(img, sigma)
    result = np.zeros((img.shape[0] - 1, img.shape[1] - 1), dtype=np.uint8)
    second_derivative = conv2d(img, laplacian, depth=np.float64)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            patch = second_derivative[i - 1:i + 2, j - 1:j + 2]
            if patch[0,0] * patch[2,2] < 0 or patch[0,2] * patch[2,0] < 0 or \
            patch[0,1] * patch[2,1] < 0 or patch [1,0] * patch [1, 2] < 0:
                if threshold is None:
                    result[i - 1, j - 1] = 1
                elif second_derivative[i, j] > threshold:
                    result[i - 1, j - 1] = 1
            else:
                result[i - 1, j - 1] = 0
    return result

def canny(img: ArrayLike, t1: Real, t2: Real, mode: str = 'same',
          smooth=True, sigma: None | Real = None, fast_module: bool = True) -> NDArray[np.uint8]:
    """
    Computes the image borders based on the Canny algorithm
    @param img: input image
    @param t1: first threshold value
    @param t2: second threshold value
    @param mode: same or valid, default same
    @param smooth: bool, default True
    @param sigma: sigma value, default None
    @param fast_module: bool, default True
    @return: output image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    _validate_parameters(img, mode, smooth, sigma)

    # Non-maximal suppression functions. The angle is clockwise
    def horizontal(n8_neighborhood):  # -
        return n8_neighborhood[1, 1] >= n8_neighborhood[1, 0] and n8_neighborhood[1, 1] >= n8_neighborhood[1, 2]
    def vertical(n8_neighborhood):    # |
        return n8_neighborhood[1, 1] >= n8_neighborhood[0, 1] and n8_neighborhood[1, 1] >= n8_neighborhood[2, 1]
    def diag_ul_br(n8_neighborhood):  # \
        return n8_neighborhood[1, 1] >= n8_neighborhood[0, 0] and n8_neighborhood[1, 1] >= n8_neighborhood[2, 2]
    def diag_ur_bl(n8_neighborhood):  # /
        return n8_neighborhood[1, 1] >= n8_neighborhood[0, 2] and n8_neighborhood[1, 1] >= n8_neighborhood[2, 0]
    nms_functions = [horizontal, diag_ul_br, vertical, diag_ur_bl] # 0, 45, 90, and 135 degrees

    # edge detection indexes
    def advance_index(angle):
        # The direction now is perpendicular. Remember the angle is clockwise
        if angle == 0:      # |
            return 1, 0
        elif angle == 1:    # /
            return - 1, 1
        elif angle == 2:    # -
            return 0, 1
        elif angle == 3:    # \
            return 1, 1
        else:
            raise NotImplementedError

    def connect_weak_edges(edges_matrix, gradient_magnitude, gradient_direction, k, l):
        # connect recursively all weak edges
        sum_i, sum_j = advance_index(gradient_direction[k, l])
        i_plus, j_plus = k + sum_i, l + sum_j
        i_minus, j_minus = k - sum_i, l - sum_j
        try:
            if not edges_matrix[i_plus, j_plus] and gradient_magnitude[i_plus, j_plus] > t1:
                edges_matrix[i_plus, j_plus] = 1
                connect_weak_edges(edges_matrix, gradient_magnitude, gradient_direction, i_plus, j_plus)
        except IndexError:
            pass
        try:
            if not edges_matrix[i_minus, j_minus] and gradient_magnitude[i_minus, j_minus] > t1:
                edges_matrix[i_minus, j_minus] = 1
                connect_weak_edges(edges_matrix, gradient_magnitude, gradient_direction, i_minus, j_minus)
        except IndexError:
            pass

    # Work with a floating image for precision and avoiding future casting, the result is uint8 to save space

    if mode == 'same':
        img = extension_pad_img(img, 1)
    img = img.astype(np.float64)

    # Smoothing the image. Default kernel is Gaussian 7x7 @σ=1
    if smooth:
       img = _smooth_image(img, sigma)

    # Compute the image gradient module and angle
    gradient_x = conv2d(img, sobel_x)
    gradient_y = conv2d(img, sobel_y)
    if not fast_module:
        gradient_module = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    else:
        gradient_module = np.abs(gradient_x) + np.abs(gradient_y)

    # Angles are defined clockwise, since the y-axis points downwards
    gradient_angle = ((1 * (gradient_y < 0) + np.arctan2(gradient_y, gradient_x) / np.pi) * 4).astype(np.uint8)
    gradient_angle[gradient_angle == 4] = 0

    # Non-maximal suppression
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            patch = gradient_module[i - 1:i + 2, j - 1:j + 2]
            if not nms_functions[gradient_angle[i, j]](patch):
                gradient_module[i, j] = 0

    gradient_module = gradient_module[1:-1,1:-1]
    gradient_angle = gradient_angle[1:-1,1:-1]
    edges = np.zeros_like(gradient_module, dtype=np.uint8)

    # Edge hysteresis assignment
    for i in range(gradient_module.shape[0]):
        for j in range(gradient_module.shape[1]):
            if gradient_module[i, j] > t2:
                if not edges[i, j]:
                    edges[i, j] = 1
                    connect_weak_edges(edges, gradient_module, gradient_angle, i, j)
    return edges