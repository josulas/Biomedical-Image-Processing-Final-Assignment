from typing import Iterable, Any
from numpy.typing import ArrayLike, NDArray
import numpy as np


def rl_encode(array: Iterable) -> tuple[dict[int, Any], NDArray]:
    elements = set(array)
    decoding, encoding = {}, {}
    for index, element in enumerate(elements):
        decoding[index] = element
        encoding[element] = index
    listaRLE = []
    actual_element = array[0]
    count = 1
    for element in array[1:]:
        if element == actual_element:
            count += 1
        else:
            listaRLE.append(count)
            listaRLE.append(encoding[element])
            actual_element = element
            count = 1
    listaRLE.append(count)
    listaRLE.append(encoding[element])
    return decoding, np.array(listaRLE)


def rl_decode(decoding: dict, encoded_array: NDArray) -> list:
    decoded = []
    for pair_index in range(0, len(encoded_array), 2):
        count = encoded_array[pair_index]
        element = decoding[encoded_array[pair_index + 1]]
        for _ in range(count):
            decoded.append(element)
    return decoded


def rl_encode_bytes(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    encoded = []
    actual_element = array[0]
    count = 1
    for element in array[1:]:
        if element != actual_element or count == 255:
            encoded.append(count)
            encoded.append(actual_element)
            actual_element = element
            count = 1
        else:
            count += 1
    encoded.append(count)
    encoded.append(actual_element)
    return np.array(encoded, dtype=np.uint8)


def rl_decode_bytes(encoded: NDArray[np.uint8]) -> NDArray[np.uint8]:
    decoded = []
    for pair_index in range(0, len(encoded), 2):
        count = encoded[pair_index]
        element = encoded[pair_index + 1]
        for _ in range(count):
            decoded.append(element)
    return np.array(decoded, dtype=np.uint8)


def get_diagonal_path(img_shape: tuple[int, int]) -> list[tuple[int, int]]:
    path = []
    total_elements = img_shape[0] * img_shape[1]
    diag_plus = True
    top_right_corner_reached = False
    bottom_left_corner_reached = False
    current_y, current_x = 0, 0
    for _ in range(total_elements):
        path.append((current_y, current_x))
        if diag_plus:
            if current_y > 0 and current_x < img_shape[1] - 1:
                # Go up and right
                current_y -= 1
                current_x += 1
            else:
                diag_plus = False
                if current_x == img_shape[1] - 1:
                   top_right_corner_reached = True
                if top_right_corner_reached:
                    # Go down
                    current_y += 1
                else:
                    # Go right
                    current_x += 1
        else:
            if current_y < img_shape[0] - 1 and current_x > 0:
                # Go down and left
                current_y += 1
                current_x -= 1
            else:
                diag_plus = True
                if current_y == img_shape[0] - 1:
                    bottom_left_corner_reached = True
                if bottom_left_corner_reached:
                    # Go right
                    current_x += 1
                else:
                    # Go down
                    current_y += 1
    return path


class RLEImageEncoder:
    # Modes
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2
    def __init__(self, mode: int | None = None):
        self.mode = mode
    def __iter__(self):
        yield RLEImageEncoder.HORIZONTAL
        yield RLEImageEncoder.VERTICAL
        yield RLEImageEncoder.DIAGONAL
    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.mode is None:
            raise ValueError("RLEncodingImageMode has not been set.")
        if image.dtype != np.uint8:
            raise ValueError("Image must have dtype np.uint8")
        if image.ndim != 2:
            raise ValueError("Image must have 2 dimensions")
        shape = image.shape
        if self.mode == RLEImageEncoder.HORIZONTAL:
            encoded_data = rl_encode_bytes(image.flatten())
        elif self.mode == RLEImageEncoder.VERTICAL:
            encoded_data = rl_encode_bytes(image.T.flatten())
        elif self.mode == RLEImageEncoder.DIAGONAL:
            pixels = get_diagonal_path(shape)
            elements = np.zeros(image.size, np.uint8)
            for index, pixel in enumerate(pixels):
                elements[index] = image[pixel]
            encoded_data = rl_encode_bytes(elements)
        else:
            raise NotImplementedError("RLEncodingImageMode has not been implemented.")
        return encoded_data


class ImageIndex:
    def __init__(self, y, x):
        self.y = y
        self.x = x
    def __add__(self, other):
        return ImageIndex(self.y + other.y, self.x + other.x)
    def __lt__(self, other):
        return self.y < other.y and self.x < other.x
    def __eq__(self, other):
        return self.y == other.y and self.x == other.x


def rl_encode_image(img: ArrayLike) -> tuple[tuple[int], int, NDArray[np.uint8]]:
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    shape = img.shape
    encoded_candidates = []
    encoder = RLEImageEncoder()
    for mode in list(encoder):
        encoder.mode = mode
        encoded_data = encoder.encode(img)
        encoded_candidates.append((encoded_data, mode, len(encoded_data)))
        encoded_candidates.append()
    encoded_candidates.sort(key = lambda el: el[-1])
    return shape, encoded_candidates[0][1], encoded_candidates[0][0]


def rl_decode_image(shape: tuple, mode: int, encoded_data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    decoded_bytes = rl_decode_bytes(encoded_data)
    if mode == RLEImageEncoder.HORIZONTAL:
        return decoded_bytes.reshape(shape)
    elif mode == RLEImageEncoder.VERTICAL:
        return decoded_bytes.reshape(shape).T
    elif mode == RLEImageEncoder.DIAGONAL:
        img = np.zeros(shape, dtype=np.uint8)
        indexes = get_diagonal_path(shape)
        for index, value in zip(indexes, decoded_bytes):
            img[index] = value
        return img
    else:
        raise ValueError("Invalid mode")
