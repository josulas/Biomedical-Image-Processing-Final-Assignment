"""
This module contains constructors for building JPEG/JFIF segments.
"""


from collections import defaultdict

import numpy as np

from libraries.compression.jfif_definitions import JFIFMarkers


def build_sof_segment(marker: bytes,
                      width: int,
                      height: int,
                      sample_precision: int,
                      sampling_factors: list,
                      quantization_table_ids: list) -> bytearray:
    """
    Constructs a Start of Frame (SOF) segment for JPEG/JFIF.
    :param marker: The marker for the SOF segment (e.g., JFIFMarkers.SOF0).
    :param width: The width of the image in pixels.
    :param height: The height of the image in pixels.
    :param sample_precision: The sample precision (e.g., 8 bits per sample).
    :param sampling_factors: A list of horizontal and vertical sampling factors for each component.
                            For grayscale, this is typically [1], for RGB it could be [2, 2, 1].
    :param quantization_table_ids: A list of quantization table IDs corresponding to each component.
                                   For grayscale, this is [0], for RGB it could be [0, 1, 1].
    :return: A bytearray containing the SOF segment.
    """
    segment = bytearray()
    segment.extend(marker)
    segment_length = 8 + 3 * len(sampling_factors)
    segment.extend(segment_length.to_bytes(2, 'big'))  # Segment length (variable)
    segment.append(sample_precision)  # Sample precision (e.g., 8 bits per sample)
    segment.extend(height.to_bytes(2, 'big'))
    segment.extend(width.to_bytes(2, 'big'))
    segment.append((len(sampling_factors)))  # Number of components (1 for grayscale, 3 for RGB)
    # Component specification for each color channel
    # (if RGB, this part needs to be repeated per channel)
    for factor_index, factor in enumerate(sampling_factors):
        # Component ID: 1 for Y (luminance), 2 for Cb (chrominance), 3 for Cr (chrominance)
        segment.append(factor_index + 1)
        # Horizontal and vertical sampling factors
        segment.append(factor)
        # Quantization table ID
        segment.append(quantization_table_ids[factor_index])
    return segment


def build_dqt_segment(tables: list, table_indexes: list[int]) -> bytearray:
    """
    Constructs a Define Quantization Table (DQT) segment for JPEG/JFIF.
    :param tables: A list of quantization tables, each as a numpy array of unsigned 8-bit integers.
    :param table_indexes: A list of quantization table indexes corresponding to each table.
                          These should be unique and typically range from 0 to 3.
    :return: A bytearray containing the DQT segment.
    """
    segment = bytearray()
    segment.extend(JFIFMarkers.DQT)
    segment_length = 2
    table_segments = []
    for table_index, table in zip(table_indexes, tables):
        if table.dtype != np.uint8:
            raise ValueError("Only unsigned 8-bit integers are supported as quantization tables.")
        precision = 8
        array = bytearray()
        array.append(precision >> 4 | table_index)
        array.extend(table.tobytes())
        table_segments.append(array)
        segment_length += len(array)
    segment.extend(segment_length.to_bytes(2, 'big'))
    for table_segment in table_segments:
        segment.extend(table_segment)
    return segment


def build_dht_segment(codebook: defaultdict, table_class: int, table_identifier: int) -> bytearray:
    """
    Constructs a Define Huffman Table (DHT) segment for JPEG/JFIF.
    :param codebook: A defaultdict where keys are symbols (integers) and values 
    are Huffman codes (bytearrays).
                The keys should be unique symbols, and the values should be byte arrays representing
                the Huffman codes for those symbols.
    :param table_class: The class of the Huffman table (0 for DC, 1 for AC).
                        This determines whether the table is for DC coefficients or AC coefficients.
    :param table_identifier: The identifier for the Huffman table (0-3).
                             This should be unique for each table.
    :return: A bytearray containing the DHT segment.
    """
    segment = bytearray()
    segment.extend(JFIFMarkers.DHT)
    sorted_codes = sorted(list(codebook.items()), key=lambda x: len(x[1]))
    number_of_symbols = len(sorted_codes)
    table_length = 2 + 1 + 16 + number_of_symbols
    segment.extend(table_length.to_bytes(2, 'big'))       # Segment length
    segment.append(table_identifier | table_class << 4)   # Table Identifier & Class
    code_lengths = [0 for _ in range(16)]
    for symbol, code in sorted_codes:
        code_lengths[len(code) - 1] += 1
    for code_length in code_lengths:
        segment.extend(code_length.to_bytes(1, 'big'))
    for symbol, code in sorted_codes:
        segment.append(symbol)
    return segment


def build_sos_segment(n_segments: int, entropy_table_ids: list[tuple],
                      spectral_selection: tuple[int, int] = (0, 63),
                      successive_approximation: int = 0) -> bytearray:
    """
    Constructs a Start of Scan (SOS) segment for JPEG/JFIF.
    :param n_segments: The number of components in the scan (e.g., 1 for grayscale, 3 for RGB).
    :param entropy_table_ids: A list of tuples where each tuple contains the DC and AC 
                              entropy table IDs for each component in the scan.
                              For example, [(0, 0), (1, 1), (1, 1)] for RGB with different tables.
    :param spectral_selection: A tuple (Ss, Se) indicating the start and end of spectral selection.
                               Default is (0, 63) for full range.
    :param successive_approximation: A byte indicating the successive approximation bits.
                                     Default is 0, which means no successive approximation.
    :return: A bytearray containing the SOS segment.
    """
    segment = bytearray()
    segment.extend(JFIFMarkers.SOS)
    # Segment length: 6 bytes + 2 bytes per component
    segment.extend((6 + 2 * n_segments).to_bytes(2, 'big'))
    # Number of components in scan
    segment.append(n_segments)
    # Component selector
    for segment_index in range(n_segments):
        # Component ID
        segment.append(segment_index + 1)
        dc_table_index, ac_table_index = entropy_table_ids[segment_index]
        # DC entropy table ID + AC entropy table ID
        segment.append(dc_table_index << 4 | ac_table_index)
    # Spectral selection and point transform / predictor
    # Start of spectral selection (Ss) = 0
    segment.append(spectral_selection[0])
    # End of spectral selection (Se) = 0
    segment.append(spectral_selection[1])
    # Successive approximation bits (1st and 2nd nibble),
    # or predictor mode for differential modes
    segment.append(successive_approximation)
    return segment
