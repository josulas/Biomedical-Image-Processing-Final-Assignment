from numpy.typing import NDArray
from skimage.transform import resize


class JPEGSubSampleModes:
    S4_4_4 = 0
    S4_2_2 = 1
    S4_2_0 = 2
    def __iter__(self):
        yield JPEGSubSampleModes.S4_4_4
        yield JPEGSubSampleModes.S4_2_2
        yield JPEGSubSampleModes.S4_2_0


class JPEGSubSampler:
    def __init__(self, mode: int = JPEGSubSampleModes.S4_4_4):
        if mode not in list(JPEGSubSampleModes()):
            raise ValueError("Invalid subsampling mode")
        self.mode = mode
    def subsample(self, y_cr_cb_image: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        shape = y_cr_cb_image.shape
        width, height = shape[1], shape[0]
        y_channel = y_cr_cb_image[:, :, 0]
        match self.mode:
            case (JPEGSubSampleModes.S4_4_4):
                cr_channel = y_cr_cb_image[:, :, 1]
                cb_channel = y_cr_cb_image[:, :, 2]
            case (JPEGSubSampleModes.S4_2_2):
                cr_channel = resize(y_cr_cb_image[:, :, 1], (height, width // 2), preserve_range=True)
                cb_channel = resize(y_cr_cb_image[:, :, 2], (height, width // 2), preserve_range=True)
            case (JPEGSubSampleModes.S4_2_0):
                cr_channel = resize(y_cr_cb_image[:, :, 1], (height // 2, width // 2), preserve_range=True)
                cb_channel = resize(y_cr_cb_image[:, :, 2], (height // 2, width // 2), preserve_range=True)
            case (_):
                raise NotImplementedError("Subsampling is not implemented yet")
        return y_channel, cr_channel, cb_channel
    def get_subsampling_factors(self):
        match self.mode:
            case (JPEGSubSampleModes.S4_4_4):
                return (1 << 4) | 1, (1 << 4) | 1, (1 << 4) | 1
            case (JPEGSubSampleModes.S4_2_2):
                return (2 << 4) | 1, (1 << 4) | 1, (1 << 4) | 1
            case (JPEGSubSampleModes.S4_2_0):
                return (2 << 4) | 2, (1 << 4) | 1, (1 << 4) | 1
            case (_):
                raise NotImplementedError("Subsampling is not implemented yet")
    def __call__(self, y_cr_cb_image: NDArray):
        frames =  self.subsample(y_cr_cb_image)
        factors = self.get_subsampling_factors()
        return frames, factors
