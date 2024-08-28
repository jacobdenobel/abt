import numpy as np

import phast


def create_neurogram(fiber_stats, selected_fibers, binsize, duration):
    bins = np.arange(0, duration, binsize)
    data = np.zeros((len(selected_fibers), len(bins) + 1))
    for i, fiber_idx in enumerate(selected_fibers):
        fs = [fs for fs in fiber_stats if fs.fiber_id == fiber_idx]
        spike_times = phast.spike_times(fs)
        idx = np.digitize(spike_times, bins)
        values, counts = np.unique_counts(idx)
        data[i, values] += counts
    return data


def bin_over_y(data, src_y, tgt_y, agg = np.max):
    data_binned = np.zeros((len(tgt_y), data.shape[1]))
    bins = np.digitize(src_y, tgt_y)

    for i in range(len(tgt_y)):
        if not any(bins == i): continue
        data_binned[i] = agg(data[bins == i], axis=0)
    return data_binned


# def bin_over_y(data, n_bins):
#     data_binned = np.zeros((data.shape[0], n_bins))

#     for i in range(data.shape[0]):
#         bins = np.digitize(src_y, tgt_y)
        
#         data_binned[i] = np.max(data[bins == i], axis=0)
#     return data_binned