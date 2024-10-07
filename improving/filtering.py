from common.padding import *


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def conv2d(img: ArrayLike, kernel: ArrayLike,
           mode: str = 'same',
           depth: numpy_real_type | None = None)\
            -> NDArray[numpy_real]:
    """
    Takes a one-channel image and a kernel and returns a filtered image.
    @param img: image to be filtered
    @param kernel: kernel to be used for filtering
    @param mode: mode of filtering, 'same' or 'valid'. Default is 'same'.
    @param depth: depth of filtering (data type of the values in the return image)
    @return: filtered image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(kernel, np.ndarray):
        kernel = np.array(kernel)
    if not isinstance(mode, str) or mode not in ['same', 'valid']:
        raise ValueError("Mode must be 'same' or 'valid'")
    if not issubclass(img.dtype.type, numpy_real):
        raise ValueError(f"Image values must be a real numpy type, got {img.dtype.type} instead")
    if not issubclass(kernel.dtype.type, numpy_real):
        raise ValueError(f"Kernel values must be a real numpy type, got {kernel.dtype.type} instead")
    if not img.ndim == 2:
        raise ValueError("Image must be a 2D array")
    if not kernel.ndim == 2:
        raise ValueError("Kernel must be a 2D array")
    if depth is not None and not issubclass(depth, numpy_real):
        raise ValueError(f"Depth must be a real numpy type, got {depth} instead")
    if img.shape[0] < kernel.shape[0] or img.shape[1] < kernel.shape[1]:
        raise ValueError("Image size must be greater or equal than kernel size")

    def get_min_max(dtype):
        info = np.iinfo(dtype)
        return info.min, info.max

    if depth is None:
        depth = img.dtype.type

    working_img = img.astype(np.float64)
    kernel = kernel.astype(np.float64)
    kernel_size = kernel.shape[0]

    if mode == 'same':
        reduction = (kernel_size // 2) * 2
        working_img = zero_pad_img(working_img, kernel_size // 2)
    else:
        reduction = ((kernel_size + 1)// 2) * 2
    img_filtered = np.zeros((working_img.shape[0] - reduction, working_img.shape[1] - reduction), dtype=depth)

    # Convolution. Note that for asymmetric convolutions, the last row and column of the zero-padded image is never used
    if issubclass(depth, np.integer):
        min_value, max_value = get_min_max(depth)
        for i in range(img_filtered.shape[0]):
            for j in range(img_filtered.shape[1]):
                img_filtered[i, j] = \
                np.clip(np.sum(working_img[i:i + kernel_size,
                                   j:j + kernel_size] * kernel),
                        min_value, max_value)
    else:
        for i in range(img_filtered.shape[0]):
            for j in range(img_filtered.shape[1]):
                img_filtered[i, j] = \
                np.sum(working_img[i:i + kernel_size,
                                   j:j + kernel_size] * kernel)

    return img_filtered


def median_filter(img: ArrayLike, mask_size: int = 3, mode: str = 'same') -> NDArray[numpy_real]:
    """
    Takes a one-channel image and a kernel size and returns a filtered image.
    @param img: image to be filtered
    @param mask_size: kernel size to be used for filtering
    @param mode: mode of filtering, 'same' or 'valid'
    @return: filtered image
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim != 2 or not issubclass(img.dtype.type, numpy_real):
        ValueError('Image must be a 2D array containing real values')
    if not isinstance(mask_size, int) or mask_size < 2:
        ValueError(f'Mask size must be an integer greater or equal than 2, got {mask_size} instead')

    pad = mask_size // 2
    if mode == 'same':
        img = extension_pad_img(img, pad)
    result = np.zeros((img.shape[0] - pad * 2, img.shape[1] - pad * 2), dtype=img.dtype)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.median(img[i: i + mask_size, j: j + mask_size])
    return result