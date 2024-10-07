from common import *


def build_projection_matrix(img_dim: int, n_projections: int):
    """
    Computes the projection matrix for the ART algorithm
    @param img_dim: int, dimension of the image
    @param n_projections: int, number of projections
    @return: projection matrix
    """
    if img_dim < 1:
        raise ValueError("imgDim must be a positive integer")
    if n_projections < 1:
        raise ValueError("NProjections must be a positive integer")

    def _get_floating_pixel_indexes(dim, p_i, a):
        center = dim / 2
        p = p_i - center
        theta_rad = np.radians(-a)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        floating_coords = []
        for i in range(dim):
            x_f = center + p * cos_theta + (i - center) * sin_theta
            y_f = center + p * sin_theta - (i - center) * cos_theta
            if 0 <= x_f < dim and 0 <= y_f < dim:
                floating_coords.append((x_f, y_f))
        return floating_coords

    def _bi_linear_interpolation_weights(f_pixel):
        x_f, y_f = f_pixel
        x0, y0 = int(np.floor(x_f)), int(np.floor(y_f))
        x1, y1 = x0 + 1, y0 + 1
        dx = x_f - x0
        dy = y_f - y0
        w00 = (1 - dx) * (1 - dy)
        w01 = (1 - dx) * dy
        w10 = dx * (1 - dy)
        w11 = dx * dy
        return [((x0, y0), w00), ((x0, y1), w01), ((x1, y0), w10), ((x1, y1), w11)]

    def _get_interpolated_pixel_weights(dim, p_ind, a):
        floating_coords = _get_floating_pixel_indexes(dim, p_ind, a)
        interpolated = []
        for floating_pixel in floating_coords:
            interpolated.extend(_bi_linear_interpolation_weights(floating_pixel))
        return interpolated

    projection_matrix = np.zeros((img_dim * n_projections, img_dim * img_dim))
    theta = np.linspace(0.0, 180.0, n_projections, endpoint=False)
    radius = img_dim // 2
    img_shape = np.array([img_dim, img_dim])

    # Create the mask for valid pixels inside the reconstruction circle
    coords = np.array(np.ogrid[: img_dim, : img_dim], dtype=object)
    dist = ((coords - (img_shape // 2)) ** 2).sum(0)
    inside_reconstruction_circle = dist <= radius**2

    # Loop through each projection and pixel
    for p_index in range(img_dim):
        for angle_index, angle in enumerate(theta):
            interpolated_pixels = _get_interpolated_pixel_weights(img_dim, p_index, angle)
            for pixel, weight in interpolated_pixels:
                x, y = pixel
                if 0 <= x < img_dim and 0 <= y < img_dim and inside_reconstruction_circle[y, x]:
                    # Flatten the pixel coordinates and assign the weight to the projection matrix
                    projection_matrix[angle_index * img_dim + p_index, y * img_dim + x] += weight
    return projection_matrix


def kaczmarz_art(a: ArrayLike, b: ArrayLike, lambda_conv: float = 1,
                 epsilon: float = 1e-4, max_iter: int = 100) -> NDArray[np.floating]:
    """
    Solve the least squares problem Ax = b using the Kaczmarz algorithm.
    @param a: two-dimensional numeric array
    @param b: one dimensional numeric array, should have A's first dimension as length
    @param lambda_conv: relaxation parameters
    @param epsilon: tolerance for the algorithm
    @param max_iter: maximum number of iterations per row of A
    @return: one dimensional numeric array
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not issubclass(a.dtype.type, numpy_real) or not a.ndim == 2:
        raise ValueError("A must be a real-valued 2D array")
    n_row_a, n_col_a = a.shape
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if not issubclass(b.dtype.type, numpy_real) or not b.ndim == 1:
        raise ValueError("b must be a real-valued 1D array")
    if not n_row_a == b.shape[0]:
        raise ValueError("A and b must have the same number of rows")
    if not epsilon > 0:
        raise ValueError("epsilon must be a positive number")
    if not max_iter > 0:
        raise ValueError("maxIter must be a positive integer")
    if not 0 < lambda_conv <= 1:
        raise ValueError("lambda must be a positive number less or equal than 1")
    x = np.ones(n_col_a)
    row_norms = np.sum(a ** 2, axis=1)
    row_norms = np.where(row_norms, row_norms, 1)
    for _ in range(max_iter):
        x_old = np.zeros(n_col_a)
        # all = A * ((b - A @ x).reshape(-1, 1))
        # normalized = all / row_norms.reshape(-1, 1)
        # x += lambdaConv * np.sum(normalized, axis=0)
        for i in range(n_row_a):
            a_i = a[i, :]
            x += lambda_conv * (b[i] - np.dot(a_i, x)) * a_i / row_norms[i]
        if np.linalg.norm(x - x_old) < epsilon:
            break
    return x


def art_reconstruction(sinogram: ArrayLike, a: ArrayLike | None = None,
                       circle: bool=False) -> NDArray[np.floating]:
    """
    Reconstruction using the ART algorithm
    @param sinogram: numeric array containing the sinogram
    @param a: reconstruction matrix
    @param circle: bool, whether the final image is a circle
    @return: numeric array containing the reconstructed image
    """
    if not isinstance(sinogram, np.ndarray):
        sinogram = np.array(sinogram)
    if not issubclass(sinogram.dtype.type, numpy_real) or not sinogram.ndim == 2:
        raise ValueError("Sinogram must be a real-valued 2D array")
    if a is None:
        a = build_projection_matrix(sinogram.shape[0], sinogram.shape[1])
    else:
        if not isinstance(a, np.ndarray):
            a = np.array(a)
    img_dim, n_projections = sinogram.shape
    b = sinogram.T.ravel()
    reconstruction = kaczmarz_art(a, b).reshape((img_dim, img_dim))
    if not circle:
        reconstruction = cut_square(reconstruction)
    else:
        cut_circle(reconstruction)
    return reconstruction