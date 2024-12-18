from numbers import Real
from numpy.typing import NDArray, ArrayLike
import numpy as np
import cv2
from compression.jfif_definitions import (JFIFMarkers,
                                          JFIFHeader,
                                          JPEGMode,
                                          JFIFHTClass,
                                          JPEGStandardTables)
from compression.jpeg_subsampling import JPEGSubSampleModes, JPEGSubSampler
from compression.jfif_block_building import build_dqt_segment, build_sof_segment, build_dht_segment, build_sos_segment
from compression.jpeg_dct_encoding import (get_dct_encoded_channels,
                                           get_channel_codebooks,
                                           encode_channel_blocks,
                                           get_encoded_scan)

class ImgType:
    GRAYSCALE = 0
    RGB = 1
    RGBA = 2
    def __iter__(self):
        yield ImgType.GRAYSCALE
        yield ImgType.RGB
        yield ImgType.RGBA


def encode_baseline(channels: list[NDArray[np.uint8]],
                    quantization_matrices: list[NDArray[np.uint8]],
                    quantization_matrices_ids: list[int],
                    dc_range: tuple[int] = (-2047, 2047),
                    ac_range: tuple[int] = (-1023, 1023)) -> bytearray:
    # Apply DCT and RL Encoding
    dct_encoded_channels = get_dct_encoded_channels(channels, quantization_matrices, quantization_matrices_ids)
    # Build Huffman's codebooks
    first_channel_codebook = get_channel_codebooks(dct_encoded_channels[0].flatten())
    channels_codebooks = [first_channel_codebook]
    if len(dct_encoded_channels) > 1:
        remaining_channels_codebook = get_channel_codebooks(np.concatenate([channel_blocks.flatten() for channel_blocks in dct_encoded_channels[1:]]))
        channels_codebooks.extend([remaining_channels_codebook for _ in range(len(dct_encoded_channels) - 1)])
    # channels_codebooks = [get_channel_codebooks(channel_blocks.flatten()) for channel_blocks in dct_encoded_channels]
    codebook_indexes = [0]
    codebook_indexes.extend([1 for _ in range(len(dct_encoded_channels) - 1)])
    # Encode data per channel per mcu
    encoded_channels_mcus = [encode_channel_blocks(channel_blocks, *codebooks)
                             for channel_blocks, codebooks in zip(dct_encoded_channels, channels_codebooks)]
    # Get the encoded scan
    encoded_scan = get_encoded_scan(encoded_channels_mcus)
    # Write data
    data = bytearray()
    n_channels = len(channels)
    encoded_tables = bytearray()
    # Write DHT segments
    written_huffman_tables = []
    for codebook_index, (dc_data_codebook, ac_data_codebook) in zip(codebook_indexes, channels_codebooks):
        if codebook_index not in written_huffman_tables:
            dht_segment_dc = build_dht_segment(dc_data_codebook, JFIFHTClass.DC, codebook_index)
            encoded_tables.extend(dht_segment_dc)
            dht_segment_ac = build_dht_segment(ac_data_codebook, JFIFHTClass.AC, codebook_index)
            encoded_tables.extend(dht_segment_ac)
            written_huffman_tables.append(codebook_index)
    # Write Scan
    data.extend(encoded_tables)
    huffman_tables_identifiers = [(codebook_index, codebook_index) for codebook_index in codebook_indexes]
    sos_segment = build_sos_segment(n_channels, huffman_tables_identifiers)
    data.extend(sos_segment)
    data.extend(encoded_scan)
    return data


def encode_progressive(channels: list[NDArray[np.uint8]],
                       quantization_matrices: list[NDArray[np.uint8]],
                       quantization_matrices_ids: list[int]) -> bytearray:
    spectral_selectors = [(0, 0), (1, 5), (6, 26), (27, 63)]
    # Apply DCT and RL Encoding
    dct_encoded_channels = get_dct_encoded_channels(channels, quantization_matrices, quantization_matrices_ids)
    data = bytearray()
    # NOT implemented yet
    return data


def pad_channel(channel_array: NDArray, custom_multiples: tuple[int, int] | None = None):
    height, width = channel_array.shape
    if custom_multiples is None:
        multiple_y, multiple_x = 8, 8
    else:
        multiple_y, multiple_x = custom_multiples
    pad_y = multiple_y - height % multiple_y if height % multiple_y else 0
    pad_x = multiple_x - width % multiple_x if width % multiple_x else 0
    return np.pad(channel_array, ((0, pad_y), (0, pad_x)), mode='edge')


def pad_multichannel_image(multichannel_image_array: NDArray, custom_multiples: tuple[int, int] | None = None):
    height, width, depth = multichannel_image_array.shape
    padded_image = []
    for channel in range(depth):
        padded_image.append(pad_channel(multichannel_image_array[:, :, channel], custom_multiples))
    return np.stack(padded_image, axis=2)


def jpeg_encode(img: ArrayLike, mode: int = JPEGMode.BASELINE,
                quality_factor: Real = 75,
                thumbnail: NDArray | None | bool = None,
                convert_to_rgb: None | bool = None,
                subsampling: None | int = None) -> bytearray:
    # Input Validation
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.uint8):
        raise NotImplementedError("Only unsigned 8-bit integers are supported as image levels")
    if img.ndim == 2:
        if convert_to_rgb is None:
            convert_to_rgb = True
        if convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_type = ImgType.RGB
        elif subsampling is not None:
            raise ValueError("Subsampling is not applied to grayscale images, \
            leave it None or select 'convert_to_RGB' as True")
        else:
            img_type = ImgType.GRAYSCALE
    elif img.ndim == 3:
        if convert_to_rgb is not None:
            raise ValueError("'convert_to_RGB' must be left None if image is not grayscale")
        if img.shape[2] == 3:
            img_type = ImgType.RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img_type = ImgType.RGBA
        else:
            raise ValueError("Image must be a 2D array or 3D array with 3 or 4 channels")
    else:
        raise ValueError("Image must be a 2D or 3D array")
    if not 10 <= quality_factor <= 100:
        raise ValueError("Quality factor must be between 10 and 100")
    mode = JPEGMode(mode) # Errors handled internally

    # Getting Everything to start encoding
    jpeg_data = bytearray()
    channels = []
    sampling_factors = []
    channels_quantization_tables_ids = []
    quantization_tables_ids = []
    quantization_tables = []
    if quality_factor > 50:
        scaling_factor = (200 - 2 * quality_factor) / 100.0
    else:
        scaling_factor = 5000 / (quality_factor * 100.0)
    y_table = np.clip(JPEGStandardTables.Y.astype(np.int64) * scaling_factor, 1, 255).astype(np.uint8)
    c_table = np.clip(JPEGStandardTables.C.astype(np.int64) * scaling_factor, 1, 255).astype(np.uint8)
    original_height, original_width = img.shape[:2]
    if img_type == ImgType.GRAYSCALE:
        img = pad_channel(img)
        channels.append(img)
        sampling_factors.append((1 << 4) + 1)
        channels_quantization_tables_ids.append(0)
        quantization_tables.append(y_table)
        quantization_tables_ids.append(0)
    elif img_type == ImgType.RGB:
        if subsampling is None:
            sub_sampler = JPEGSubSampler()
        else:
            sub_sampler = JPEGSubSampler(subsampling)
        match sub_sampler.mode:
            case (JPEGSubSampleModes.S4_4_4):
                padding_lengths = (8, 8)
            case (JPEGSubSampleModes.S4_2_2):
                padding_lengths = (8, 16)
            case (JPEGSubSampleModes.S4_2_0):
                padding_lengths = (16, 16)
            case (_):
                raise NotImplementedError("Non Implemented type of sampling for the encoder")
        y_cr_cb_img = pad_multichannel_image(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB), padding_lengths)
        channels_tuple, factors_tuple = sub_sampler(y_cr_cb_img)
        for channel, factor in zip(channels_tuple, factors_tuple):
            channels.append(channel)
            sampling_factors.append(factor)
        # Exchange Chrominance channels (for some reason they should be Cb and Cr)
        # channels[1], channels[2] = channels[2], channels[1]
        # Pad Chrominance channels:
        channels_quantization_tables_ids.extend([0, 1, 1])
        quantization_tables.extend([y_table, c_table])
        quantization_tables_ids.extend([0, 1])
    else:
        raise NotImplementedError("Only grayscale and RGB images are supported (for now)")
    n_channels = len(channels)
    channel_indexes = np.arange(n_channels)
    jpeg_data.extend(JFIFMarkers.SOI)
    jpeg_data.extend(JFIFMarkers.APP0)
    if thumbnail is not None:
        NotImplementedError("Thumbnails are not supported yet, leave it None")
    header = JFIFHeader(thumbnail=thumbnail)()
    jpeg_data.extend(header)
    # Write DQT, if applicable
    if quantization_tables:
        for table, table_index in zip(quantization_tables, quantization_tables_ids):
            dqt_segment = build_dqt_segment([table], [table_index])
            jpeg_data.extend(dqt_segment)
    # Frame
    sof_segment = build_sof_segment(mode(),
                                    original_width,
                                    original_height,
                                    8,
                                    sampling_factors,
                                    channels_quantization_tables_ids)
    jpeg_data.extend(sof_segment)
    match mode:
        case JPEGMode.BASELINE:
            jpeg_data.extend(encode_baseline(channels, quantization_tables, channels_quantization_tables_ids))
        case JPEGMode.EXTENDED_SEQUENTIAL:
            raise NotImplementedError(f"{JPEGMode.EXTENDED_SEQUENTIAL} mode is not implemented yet.")
        case JPEGMode.PROGRESSIVE:
            jpeg_data.extend(encode_progressive(channels, quantization_tables, channels_quantization_tables_ids))
        case JPEGMode.LOSSLESS:
            raise NotImplementedError(f"{JPEGMode.LOSSLESS} mode is not implemented yet.")
        case JPEGMode.DIFFERENTIAL_SEQUENTIAL:
            raise NotImplementedError(f"{JPEGMode.DIFFERENTIAL_SEQUENTIAL} mode is not implemented yet.")
        case JPEGMode.DIFFERENTIAL_PROGRESSIVE:
            raise NotImplementedError(f"{ JPEGMode.DIFFERENTIAL_PROGRESSIVE} mode is not implemented yet.")
        case JPEGMode.DIFFERENTIAL_LOSSLESS:
            raise NotImplementedError(f"{JPEGMode.DIFFERENTIAL_LOSSLESS} mode is not implemented yet.")
    jpeg_data.extend(JFIFMarkers.EOI)
    return jpeg_data