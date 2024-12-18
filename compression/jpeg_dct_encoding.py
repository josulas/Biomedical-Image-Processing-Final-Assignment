from collections import defaultdict
import numpy as np
from scipy.fft import dctn
from numpy.typing import NDArray
from bitarray import bitarray
from compression.rle import get_diagonal_path
from compression.huffman import (index_to_bit_array,
                                 get_huffman_tree,
                                 limit_huffman_code_length,
                                 get_abstract_tree,
                                 build_sorted_codebook_from_abstraction)


def escape_ff(scan_data: bytearray | bytes) -> bytearray:
    escaped_data = bytearray()
    for byte in scan_data:
        escaped_data.append(byte)
        if byte == 0xFF:
            escaped_data.append(0x00)  # Escape with a zero byte
    return escaped_data


BLOCK_PATH = get_diagonal_path((8, 8))      # Diagonal ordered pixel indexes for a 8x8 chunk
def get_dct_transformed_block_elements(channel: NDArray, chunk_indexes: tuple[int, int],
                                       quantization_matrix: NDArray[np.float64]) -> NDArray[np.int64]:
    start_y, start_x = chunk_indexes[0] * 8, chunk_indexes[1] * 8
    end_y, end_x = start_y + 8, start_x + 8
    chunk = channel[start_y:end_y, start_x:end_x].astype(np.float64) - 128
    chunk_transformed = dctn(chunk, type=2, norm='ortho')
    chunk_truncated = np.round(chunk_transformed / quantization_matrix).astype(np.int64)
    elements = np.zeros(64, np.int64)
    for pixel_index, pixel in enumerate(BLOCK_PATH):
        elements[pixel_index] = chunk_truncated[pixel]
    return elements


def get_channels_blocks_paths(channels: list[NDArray[np.uint8]]) -> list[NDArray[np.uint16]]:
    # First channel is assumed to have widths and heights multiples of following channels
    # Remaining channels are assumed to have the same shape
    initial_channel_shape = channels[0].shape
    initial_channels_y_chunks = initial_channel_shape[0] // 8
    initial_channels_x_chunks = initial_channel_shape[1] // 8
    remaining_channels_shape = channels[-1].shape
    base_y_chunks = remaining_channels_shape[0] // 8
    base_x_chunks = remaining_channels_shape[1] // 8
    y_spacing = initial_channels_y_chunks // base_y_chunks
    x_spacing = initial_channels_x_chunks // base_x_chunks
    initial_channel_path = np.zeros((base_y_chunks * base_x_chunks,
                                     y_spacing * x_spacing, 2), dtype=np.uint16)
    remaining_channels_path = np.zeros((base_y_chunks * base_x_chunks,
                                     1, 2), dtype=np.uint16)
    mcu_x_indexes = np.arange(base_x_chunks, dtype=np.uint16)
    for base_y_index in range(base_y_chunks):
        row_indexes = mcu_x_indexes + base_y_index * base_x_chunks
        remaining_channel_row_values = np.stack((np.ones(base_x_chunks, dtype=np.uint16) * base_y_index,
                                                mcu_x_indexes), axis=1)
        remaining_channels_path[row_indexes, 0] = remaining_channel_row_values
    y_indexes_basic_unit = np.arange(y_spacing,
                                     dtype=np.uint16).reshape((y_spacing, 1)) @ np.ones((1, x_spacing),
                                                                                                   dtype=np.uint16)
    x_indexes_basic_unit = np.ones((y_spacing, 1),
                                   dtype=np.uint16) @ np.arange(x_spacing,
                                                                dtype=np.uint16).reshape((1, x_spacing))
    initial_channel_basic_unit = np.stack((y_indexes_basic_unit, x_indexes_basic_unit),
                                          axis=-1).reshape((y_spacing * x_spacing, 2))
    initial_channel_path[:] = initial_channel_basic_unit
    for base_y_index in range(base_y_chunks):
        for base_x_index in range(base_x_chunks):
            unit_index = base_y_index * base_x_chunks + base_x_index
            initial_channel_path[unit_index, :, 0] += base_y_index * y_spacing
            initial_channel_path[unit_index, :, 1] += base_x_index * x_spacing
    channels_paths = [initial_channel_path]
    channels_paths.extend([remaining_channels_path for _ in range(len(channels) - 1)])
    return channels_paths


class DCTEncodedBlock:
    EOB = 0  # End Of Block byte, (0, 0) nimble pair
    ZRL = 15 << 4  # Zero Run Length byte, (15, 0) nimble pair
    CATEGORY_AND_INDEX = {0: (0, 0)}  # Encoding
    for i in range(1, 16):
        categories_numbers = np.concatenate(
            (np.arange(-2 ** i + 1, - 2 ** (i - 1) + 1), np.arange(2 ** (i - 1), 2 ** i)))
        for index, number in enumerate(categories_numbers):
            CATEGORY_AND_INDEX[number] = (i, index)
    def __init__(self, block_elements: NDArray[np.int64], last_dc: np.int64):
        dc_coefficient = block_elements[0] - last_dc
        dc_coefficient_category, dc_coefficient_index = DCTEncodedBlock.CATEGORY_AND_INDEX[dc_coefficient]
        self.dc_class = dc_coefficient_category
        self.dc_coefficient_encoded_index = index_to_bit_array(dc_coefficient_index, dc_coefficient_category)
        self.ac_classes, self.ac_coefficient_encoded_indexes = self._get_ac_encoded_els(block_elements[1:])
    @staticmethod
    def _get_ac_encoded_els(ac_coefficients) -> tuple[list[np.int64], list[bitarray]]:
        ac_categories = []
        ac_coefficient_encoded_indexes = []
        n_zeros = 0
        for element_index, element in enumerate(ac_coefficients):
            if element == 0:
                n_zeros += 1
            else:
                while n_zeros > 15:
                    ac_categories.append(DCTEncodedBlock.ZRL)
                    ac_coefficient_encoded_indexes.append(bitarray())
                    n_zeros -= 16
                category, element_index = DCTEncodedBlock.CATEGORY_AND_INDEX[element]
                ac_categories.append((n_zeros << 4) | category)
                ac_coefficient_encoded_indexes.append(index_to_bit_array(element_index, category))
                n_zeros = 0
        if n_zeros:
            ac_categories.append(DCTEncodedBlock.EOB)
            ac_coefficient_encoded_indexes.append(bitarray())
        return ac_categories, ac_coefficient_encoded_indexes
    def get_encoded_block(self, dc_codebook: defaultdict, ac_codebook: defaultdict) -> bitarray:
        assert len(self.ac_classes) == len(self.ac_coefficient_encoded_indexes), "AC coefficients badly encoded"
        encoded_block = bitarray()
        encoded_block.extend(dc_codebook[self.dc_class])
        encoded_block.extend(self.dc_coefficient_encoded_index)
        for ac_class, ac_coefficient_encoded_index in zip(self.ac_classes, self.ac_coefficient_encoded_indexes):
            encoded_block.extend(ac_codebook[ac_class])
            encoded_block.extend(ac_coefficient_encoded_index)
        return encoded_block


def get_dct_encoded_channels(channels: list[NDArray[np.uint8]],
                         quantization_matrices: list[NDArray[np.uint8]],
                         quantization_matrices_ids: list[int]) -> list[NDArray[DCTEncodedBlock]]:
    channels_blocks_paths = get_channels_blocks_paths(channels)
    n_mcus = channels_blocks_paths[0].shape[0]
    dct_encoded_channels = [np.zeros(path.shape[:-1], dtype=DCTEncodedBlock) for path in channels_blocks_paths]
    for channel_index, channel in enumerate(channels):
        quantization_matrix = quantization_matrices[quantization_matrices_ids[channel_index]].astype(np.float64)
        last_dc = np.int64(0)
        for mcu_index in range(n_mcus):
            for block_index, block_indexes in enumerate(channels_blocks_paths[channel_index][mcu_index]):
                dct_encoded_elements = get_dct_transformed_block_elements(channel,
                                                                          block_indexes,
                                                                          quantization_matrix)
                encoded_block = DCTEncodedBlock(dct_encoded_elements, last_dc)
                dct_encoded_channels[channel_index][mcu_index][block_index] = encoded_block
                last_dc = dct_encoded_elements[0]
    return dct_encoded_channels


def get_channel_codebooks(channel_blocks: NDArray[DCTEncodedBlock]) -> tuple[defaultdict, defaultdict]:
    dc_classes = [block.dc_class for block in channel_blocks]
    ac_classes = []
    for block in channel_blocks:
        ac_classes.extend(block.ac_classes)
    dc_data_huffman_tree = get_huffman_tree(dc_classes)
    ac_data_huffman_tree = get_huffman_tree(ac_classes)
    abstracted_dc_tree = limit_huffman_code_length(get_abstract_tree(dc_data_huffman_tree))
    abstracted_ac_tree = limit_huffman_code_length(get_abstract_tree(ac_data_huffman_tree))
    dc_data_codebook = build_sorted_codebook_from_abstraction(abstracted_dc_tree)
    ac_data_codebook = build_sorted_codebook_from_abstraction(abstracted_ac_tree)
    return dc_data_codebook, ac_data_codebook


def encode_channel_blocks(channel_blocks: NDArray[DCTEncodedBlock],
                          dc_codebook: defaultdict,
                          ac_codebook: defaultdict) -> NDArray[bitarray]:
    encoded_channel_mcus = np.zeros(channel_blocks.shape, dtype=bitarray)
    for mcu_index, mcu_block in enumerate(channel_blocks):
        for block_index, block in enumerate(mcu_block):
            encoded_channel_mcus[mcu_index, block_index] = block.get_encoded_block(dc_codebook, ac_codebook)
    return encoded_channel_mcus


def get_encoded_scan(bit_encoded_channels_mcus: list[NDArray[bitarray]]) -> bytearray:
    scan = bitarray()
    for channels_mcus in zip(*bit_encoded_channels_mcus):
        for mcus in channels_mcus:
            for mcu in mcus:
                scan.extend(mcu)
    return escape_ff(scan.tobytes())