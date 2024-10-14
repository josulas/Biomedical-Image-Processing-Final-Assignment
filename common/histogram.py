import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


class ImgHist:
    """
    Class to store the histogram of an image
    """
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.hist = np.zeros(n_bins, dtype=np.int_)
        self.bin_intervals = np.zeros((n_bins, 2), dtype=np.int_)
    def __str__(self):
        return f"ImgHist(n_bins={self.n_bins}, hist={self.hist}, bin_intervals={self.bin_intervals})"
    def __repr__(self):
        return f"ImgHist(n_bins={self.n_bins}, hist={self.hist}, bin_intervals={self.bin_intervals})"
    def __eq__(self, other):
        return self.n_bins == other.n_bins and np.array_equal(self.hist, other.hist) and np.array_equal(self.bin_intervals, other.bin_intervals)
    def set_hist(self, hist, bin_intervals):
        self.hist = hist
        self.bin_intervals = bin_intervals
    def get_hist(self):
        return self.hist, self.bin_intervals


def img2hist(img: ArrayLike, n_bins: int | None = None, v_min: int = None, v_max: int | None = None) \
-> ImgHist:
    """
    Takes a one-channel image and returns a histogram of the image.
    @param img: grayscale image, a 2D numpy array with integer values
    @param n_bins: number of bins in the histogram, a positive integer
    @param v_min: minimum value of the bins, a positive integer
    @param v_max: maximum value of the image, a positive integer
    @return: histogram of the image: a tuple containing the bin counts and the bin intervals
    """
    # Assertions
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    info = np.iinfo(img.dtype)
    abs_max, abs_min = info.max, info.min
    if n_bins is None:
        n_bins = abs_max - abs_min + 1
    if v_min is None:
        v_min = abs_min
    if v_max is None:
        v_max = abs_max
    if v_max - v_min < n_bins - 1:
        raise ValueError("Number of bins must be lower or equal than the range given by v_max and v_min")

    hist = np.zeros(n_bins, dtype=np.int_)
    hist_bin_intervals = np.zeros((n_bins, 2), dtype=np.int_)
    bin_width = (v_max + 1 - v_min) / n_bins
    for i in range(n_bins):
        hist_bin_intervals[i, 0] = v_min + int(i * bin_width)
        hist_bin_intervals[i, 1] = v_min + int((i + 1) * bin_width)
    flatten_img = np.clip(np.ravel(img), v_min, v_max)
    for pixel in flatten_img:
        hist[int(pixel / bin_width) - v_min] += 1
    img_hist = ImgHist(n_bins)
    img_hist.set_hist(hist, hist_bin_intervals)
    return img_hist


def plot_hist(hist: ImgHist, **kwargs):
    """
    Plot the histogram of an image
    @param hist: histogram of the image
    """
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (10, 5)
    if 'title' not in kwargs:
        kwargs['title'] = 'Histogram'
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Pixel value'
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Frequency'
    if 'figure' not in kwargs:
        kwargs['figure'], kwargs['ax'] = plt.subplots(figsize=kwargs['figsize'])
    if 'color' not in kwargs:
        kwargs['color'] = kwargs['ax']._get_lines.get_next_color()
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 1
    if 'legend' not in kwargs:
        kwargs['legend'] = None
    if 'show' not in kwargs:
        kwargs['show'] = True

    fig, ax = kwargs['figure'], kwargs['ax']
    title, x_label, y_label= kwargs['title'], kwargs['xlabel'], kwargs['ylabel']
    color, alpha = kwargs['color'], kwargs['alpha']
    legend = kwargs['legend']
    show = kwargs['show']

    hist_values, bin_intervals = hist.get_hist()
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if legend is not None:
        ax.bar(bin_intervals[:, 0], hist_values, width=bin_intervals[:, 1] - bin_intervals[:, 0], color=color, alpha=alpha,
           label=legend)
        ax.legend()
    else:
        ax.bar(bin_intervals[:, 0], hist_values, width=bin_intervals[:, 1] - bin_intervals[:, 0], color=color, alpha=alpha)
    if show:
        fig.tight_layout()
        fig.show()