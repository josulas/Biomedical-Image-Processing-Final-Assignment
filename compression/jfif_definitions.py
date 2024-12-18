import numpy as np
from numpy.typing import NDArray


class JFIFMarkers:
    SOI = b'\xFF\xD8'         # Start of Image
    APP0 = b'\xFF\xE0'        # Application
    DHT = b'\xFF\xC4'         # Define Huffman Table
    DQT = b'\xFF\xDB'         # Define Quantization Table
    SOS = b'\xFF\xDA'         # Start of Scan
    EOI = b'\xFF\xD9'         # End of Image


class JFIFHTClass:
    DC = 0
    AC = 1
    def __iter__(self):
        yield JFIFHTClass.AC
        yield JFIFHTClass.DC


class JFIFHeaderVersion:
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
        return self.version.to_bytes(1, 'big') + self.revision.to_bytes(1, 'big')
    def __call__(self):
        return self.get_bytes()


class JFIFHeaderUnits:
    NONE = 0
    DOTS_PER_INCH = 1
    DOTS_PER_CM = 2
    def __init__(self, units: int = 0):
        if units not in range(3):
            raise NotImplementedError("Units must be between 0 and 2")
        self.units = units
    def get_bytes(self):
        return self.units.to_bytes(1, 'big')
    def __call__(self):
        return self.get_bytes()


class JFIFHeaderDensity:
    def __init__(self, x: int = 1, y: int = 1):
        self.x = x
        self.y = y
    def get_bytes(self):
        return self.x.to_bytes(2, 'big') + self.y.to_bytes(2, 'big')
    def __call__(self):
        return self.get_bytes()


class JFIFHeader:
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
                raise ValueError("Thumbnail must be a RGB standard image if it is not None, with equal or less than 255\
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
            raise ValueError(f"Mode should be one of the following: {JPEGMode.mode_list}, got {mode} instead")
        self.mode = mode
    @staticmethod
    def get_frame(mode):
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