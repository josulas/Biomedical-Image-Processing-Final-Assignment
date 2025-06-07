"""
This file contains JIFF (JPEG File Interchange Format) definitions and classes.
"""


import numpy as np
from numpy.typing import NDArray


class JFIFMarkers:
    """
    This class contains the standard JPEG/JFIF markers.
    These markers are used to identify different segments in a JPEG file.
    """
    SOI = b'\xFF\xD8'         # Start of Image
    APP0 = b'\xFF\xE0'        # Application
    DHT = b'\xFF\xC4'         # Define Huffman Table
    DQT = b'\xFF\xDB'         # Define Quantization Table
    SOS = b'\xFF\xDA'         # Start of Scan
    EOI = b'\xFF\xD9'         # End of Image


class JFIFHTClass:
    """
    This class defines the Huffman Table classes for JFIF.
    The classes are used to differentiate between DC and AC coefficients.
    """
    DC = 0
    AC = 1
    def __iter__(self):
        yield JFIFHTClass.AC
        yield JFIFHTClass.DC


class JFIFHeaderVersion:
    """
    This class defines the JFIF header version and revision.
    The version is always 1, and the revision can be 0, 1, or 2.
    """
    VERSION_1 = 1
    REVISION_0 = 0
    REVISION_1 = 1
    REVISION_2 = 2
    def __init__(self, version: int = 1, revision: int = 1):
        if version != 1:
            raise NotImplementedError("Version must be 1")
        self.version = version
        if revision not in range(3):
            raise NotImplementedError("Revision must be between 0 and 2")
        self.revision = revision
    def get_bytes(self):
        """Returns the version and revision as two bytes."""
        return self.version.to_bytes(1, 'big') + self.revision.to_bytes(1, 'big')
    def __call__(self):
        return self.get_bytes()


class JFIFHeaderUnits:
    """
    This class defines the units for the JFIF header density.
    The units can be:
    - 0: No units
    - 1: Dots per inch (DPI)
    - 2: Dots per centimeter (DPCM)
    """
    NONE = 0
    DOTS_PER_INCH = 1
    DOTS_PER_CM = 2
    def __init__(self, units: int = 0):
        if units not in range(3):
            raise NotImplementedError("Units must be between 0 and 2")
        self.units = units
    def get_bytes(self):
        """Returns the units as a single byte."""
        return self.units.to_bytes(1, 'big')
    def __call__(self):
        return self.get_bytes()


class JFIFHeaderDensity:
    """
    This class defines the pixel density for the JFIF header.
    The density is defined by the x and y values, which are both 16-bit integers.
    """
    def __init__(self, x: int = 1, y: int = 1):
        self.x = x
        self.y = y
    def get_bytes(self):
        """Returns the density as two 16-bit integers."""
        return self.x.to_bytes(2, 'big') + self.y.to_bytes(2, 'big')
    def __call__(self):
        return self.get_bytes()


class JFIFHeader:
    """
    This class defines the JFIF header structure.
    The header includes the version, units, density, and an optional thumbnail.
    The thumbnail is a small image that can be included in the JFIF file.
    The header is 16 bytes long, plus the size of the thumbnail if it is included.
    """
    IDENTIFIER = b'\x4A\x46\x49\x46\x00'
    def __init__(self, version: JFIFHeaderVersion = JFIFHeaderVersion(),
                 units: JFIFHeaderUnits = JFIFHeaderUnits(),
                 density: JFIFHeaderDensity = JFIFHeaderDensity(),
                 thumbnail: NDArray | None = None):
        self.version = version
        self.identifier = JFIFHeader.IDENTIFIER
        self.units = units
        self.density = density
        if thumbnail is not None:
            if not thumbnail.ndim == 3 or thumbnail.shape[2] != 3 or \
            thumbnail.shape[0] > 255 or thumbnail.shape[1] > 255 or \
            thumbnail.dtype != np.uint8:
                raise ValueError("Thumbnail must be a RGB standard image if it \
                                 is not None, with equal or less than 255\
                                 pixels per dimension")
            x_thumbnail = thumbnail.shape[1]
            y_thumbnail = thumbnail.shape[0]
            self.thumbnail = thumbnail.tobytes()
        else:
            self.thumbnail = None
            x_thumbnail = 0
            y_thumbnail = 0
        self.x_thumbnail = x_thumbnail
        self.y_thumbnail = y_thumbnail
        self.length = 16 + 3 * x_thumbnail * y_thumbnail
        self.header = None
    def get_bytes(self):
        """Constructs the JFIF header as a bytearray."""
        self.header = bytearray()
        self.header += self.length.to_bytes(2, 'big')
        self.header += self.identifier
        self.header += self.version()
        self.header += self.units()
        self.header += self.density()
        self.header += self.x_thumbnail.to_bytes(1, 'big')
        self.header += self.y_thumbnail.to_bytes(1, 'big')
        if self.thumbnail is not None:
            self.header += self.thumbnail
        if len(self.header) != self.length:
            raise ValueError("Header length does not match")
        return self.header
    def __call__(self):
        return self.get_bytes()


class JPEGMode:
    """
    This class defines the JPEG modes.
    Each mode corresponds to a specific JPEG frame type.
    The modes are:
    - BASELINE: Baseline DCT
    - EXTENDED_SEQUENTIAL: Extended sequential DCT
    - PROGRESSIVE: Progressive DCT
    - LOSSLESS: Lossless DCT
    - DIFFERENTIAL_SEQUENTIAL: Differential sequential DCT
    - DIFFERENTIAL_PROGRESSIVE: Differential progressive DCT
    - DIFFERENTIAL_LOSSLESS: Differential lossless DCT
    Each mode has a corresponding SOF marker.
    """
    BASELINE = 0
    EXTENDED_SEQUENTIAL = 1
    PROGRESSIVE = 2
    LOSSLESS = 3
    DIFFERENTIAL_SEQUENTIAL= 5
    DIFFERENTIAL_PROGRESSIVE = 6
    DIFFERENTIAL_LOSSLESS = 7
    SOF0 = b'\xFF\xC0'
    SOF1 = b'\xFF\xC1'
    SOF2 = b'\xFF\xC2'
    SOF3 = b'\xFF\xC3'
    SOF5 = b'\xFF\xC5'
    SOF6 = b'\xFF\xC6'
    SOF7 = b'\xFF\xC7'
    frame_dictionary = {
        BASELINE: SOF0,
        EXTENDED_SEQUENTIAL: SOF1,
        PROGRESSIVE: SOF2,
        LOSSLESS: SOF3,
        DIFFERENTIAL_SEQUENTIAL: SOF5,
        DIFFERENTIAL_PROGRESSIVE: SOF6,
        DIFFERENTIAL_LOSSLESS: SOF7
    }
    mode_list = [
            BASELINE,
            EXTENDED_SEQUENTIAL,
            PROGRESSIVE,
            LOSSLESS,
            DIFFERENTIAL_SEQUENTIAL,
            DIFFERENTIAL_PROGRESSIVE,
            DIFFERENTIAL_LOSSLESS
        ]
    def __init__(self, mode: int = BASELINE):
        if mode not in JPEGMode.mode_list:
            raise ValueError(f"Mode should be one of the following: \
                             {JPEGMode.mode_list}, got {mode} instead")
        self.mode = mode
    @staticmethod
    def get_frame(mode):
        """Returns the SOF marker for the given mode."""
        return JPEGMode.frame_dictionary[mode]
    def __eq__(self, other):
        if isinstance(other, JPEGMode):
            return self.mode == other.mode
        elif isinstance(other, int):
            return self.mode == other
        else:
            return False
    def __ne__(self, other):
        return not self.__eq__(other)
    def __call__(self):
        return self.get_frame(self.mode)


class JPEGStandardTables:
    """
    This class contains the standard JPEG quantization tables.
    These tables are used for lossy compression in JPEG images.
    The tables are defined for luminance (Y) and chrominance (C).
    """
    Y = np.array([[16, 11, 10, 16, 24, 40, 51,61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]],
                 dtype=np.uint8)
    C = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99]],
                  dtype=np.uint8)
