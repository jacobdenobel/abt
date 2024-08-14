import numpy as np

DEFAULT_BINS = np.array([2, 2, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 10, 56])

DEFAULT_BIN_TO_LOC_MAP = (
    np.concatenate(
        (
            np.zeros(
                6,
            ),
            np.array(
                [
                    256,
                    640,
                    896,
                    1280,
                    1664,
                    1920,
                    2176,  # 1 x nBin vector of nominal cochlear locations for the center frequencies of each STFT bin
                    2432,
                    2688,
                    2944,
                    3157,
                    3328,
                    3499,
                    3648,
                    3776,
                    3904,
                    4032,  # values from 0 .. 15 in Q9 format
                    4160,
                    4288,
                    4416,
                    4544,
                    4659,
                    4762,
                    4864,
                    4966,
                    5069,
                    5163,  # corresponding to the nominal steering location for each
                    5248,
                    5333,
                    5419,
                    5504,
                    5589,
                    5669,
                    5742,
                    5815,
                    5888,
                    5961,  # FFT bin
                    6034,
                    6107,
                    6176,
                    6240,
                    6304,
                    6368,
                    6432,
                    6496,
                    6560,
                    6624,
                    6682,
                    6733,
                    6784,
                    6835,
                    6886,
                    6938,
                    6989,
                    7040,
                    7091,
                    7142,
                    7189,
                    7232,
                    7275,
                    7317,
                    7360,
                    7403,
                    7445,
                    7488,
                    7531,
                    7573,
                    7616,
                    7659,
                ]
            ),
            7679 * np.ones((53,)),
        )
    )
    / 512
)


DEFAULT_CHANNEL_ORDER = np.array([1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12])


DEFAULT_COEFFS = np.array(
    [
        -19,
        55,
        153,
        277,
        426,
        596,
        784,
        983,
        1189,
        1393,
        1587,
        1763,
        1915,
        2035,
        2118,
        2160,
        2160,
        2118,
        2035,
        1915,
        1763,
        1587,
        1393,
        1189,
        983,
        784,
        596,
        426,
        277,
        153,
        55,
        -19,
    ]
) / (2**16)
