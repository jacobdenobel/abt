import numpy as np
import scipy

def q(*args, **kwargs):
    return np.quantile(*args, q=.90, **kwargs)


def bin_over_y(data, src_y, tgt_y, agg = np.max):
    data_binned = np.zeros((len(tgt_y), data.shape[1]))
    bins = np.digitize(src_y, tgt_y)

    for i in range(len(tgt_y)):
        if not any(bins == i): continue
        data_binned[i] = agg(data[bins == i], axis=0)
    return data_binned



def min_max_scale(data, a=-80, b=0, data_min=None, data_max=None):
    data_min = data_min or np.min(data)
    data_max = data_max or np.max(data)
    return a + (data - data_min) * (b - a) / (data_max - data_min)


def fix_audio_scaling(idata, lb, ub):
    data = idata.copy()
    data[data > 0] = min_max_scale(data[data > 0], 0, ub)
    data[data < 0] = min_max_scale(data[data < 0], lb, 0)
    return data

def make_bins(n, data):
    if n == 1:
        return data
    return data[:, : len(data[0]) // n * n].reshape(data.shape[0], -1, n).sum(axis=2)


def smooth(
    data: np.ndarray,
    window_type: str = "hann",
    window_size: int = 2048,
    hop_length: int = None
) -> np.ndarray:
    
    hop_length = hop_length or max(window_size // 4, 1)
    window = scipy.signal.get_window(window_type, window_size)
    wsum = window.sum()
    data = np.vstack(
        [
            (np.convolve(data[i], window, mode="same") / wsum)[::hop_length]
            for i in range(data.shape[0])
        ]
    )
    return data

def apply_filter(
    data: np.ndarray,
    window_type: str = "hann",
    window_size: int = 2048,
    hop_length: int = None,
    scale: bool = True,
    clip_outliers: float = 0.0,
    n_bins: int = 0,
    resample_to: int = None,
    data_min=None, 
    data_max=None
):
    data = data.copy()
    if clip_outliers != 0.0:
        q99 = np.quantile(data.ravel(), clip_outliers)
        data[data > q99] = q99

    if n_bins != 0:
        data = make_bins(n_bins, data)
    
    data = smooth(data, window_type, window_size, hop_length)

    if resample_to:
        data = np.array([scipy.signal.resample(x, resample_to) for x in data])
        
    if scale:
        data = min_max_scale(data, data_min=data_min, data_max=data_max)

    return data


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]
