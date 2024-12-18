import numpy as np
from collections import defaultdict
from compression.jfif_definitions import JFIFMarkers


def build_sof_segment(marker: bytes,
                      width: int,
                      height: int,
                      sample_precision: int,
                      sampling_factors: list,
                      quantization_table_ids: list) -> bytearray:
    segment = bytearray()
    segment.extend(marker)
    segment_length = 8 + 3 * len(sampling_factors)
    segment.extend(segment_length.to_bytes(2, 'big'))  # Segment length (variable)
    segment.append(sample_precision)  # Sample precision (e.g., 8 bits per sample)
    segment.extend(height.to_bytes(2, 'big'))
    segment.extend(width.to_bytes(2, 'big'))
    segment.append((len(sampling_factors)))  # Number of components (1 for grayscale, 3 for RGB)
    # Component specification for each color channel (if RGB, this part needs to be repeated per channel)
    for factor_index, factor in enumerate(sampling_factors):
        segment.append(factor_index + 1)                        # Component ID
        segment.append(factor)                                  # Horizontal and vertical sampling factors
        segment.append(quantization_table_ids[factor_index])    # Quantization Table ID
    return segment


def build_dqt_segment(tables: list, table_indexes: list[int]) -> bytearray:
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
    segment = bytearray()
    segment.extend(JFIFMarkers.DHT)
    sorted_codes = sorted(list(codebook.items()), key=lambda x: len(x[1]))
    number_of_symbols = len(sorted_codes)
    table_length = 2 + 1 + 16 + number_of_symbols
    segment.extend(table_length.to_bytes(2, 'big'))       # Segment length
    segment.append(table_identifier | table_class << 4)     # Table Identifier & Class
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
    segment = bytearray()
    segment.extend(JFIFMarkers.SOS)
    # Segment length: 6 bytes + 2 bytes per component
    segment.extend((6 + 2 * n_segments).to_bytes(2, 'big'))
    segment.append(n_segments)              # Number of components in scan
    # Component selector
    for segment_index in range(n_segments):
        segment.append(segment_index + 1)               # Component ID
        dc_table_index, ac_table_index = entropy_table_ids[segment_index]
        segment.append(dc_table_index << 4 | ac_table_index)    # DC entropy table ID + AC entropy table ID
    # Spectral selection and point transform / predictor
    segment.append(spectral_selection[0])                       # Start of spectral selection (Ss) = 0
    segment.append(spectral_selection[1])                       # End of spectral selection (Se) = 0
    # Successive approximation bits (1st and 2nd nibble), or predictor mode for differential modes
    segment.append(successive_approximation)
    return segment