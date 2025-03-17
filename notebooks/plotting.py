import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
from matplotlib.colors import LogNorm, Normalize


def frequency_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_yscale("symlog", linthresh=1000.0, base=2)
    ax.yaxis.set_major_formatter(mplticker.ScalarFormatter())
    ax.yaxis.set_major_locator(
        mplticker.SymmetricalLogLocator(ax.yaxis.get_transform())
    )
    ax.yaxis.set_label_text("frequency [Hz]")


def time_vs_freq(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("time [s]")
    frequency_ax(ax)


def plot_heatmap(
    t, y, data, 
    ax=None, fig=None, 
    show_bands: bool = True, 
    pad_idx: bool = False,
    figsize=(9, 4),
    logcolors: bool = False
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    norm = LogNorm(vmin=data.min(), vmax=data.max()) if logcolors else Normalize(vmin=data.min(), vmax=data.max())

    if pad_idx:
        n_idx = np.nonzero(data.sum(axis=0))[0]
        n_idx = np.unique(np.c_[n_idx - 1, n_idx, n_idx + 1].ravel())
        n_idx = n_idx[n_idx < t.size]
        img = ax.pcolormesh(
            t[n_idx], y, data[:, n_idx], cmap="inferno", norm=norm
            
        )
    else:
        img = ax.pcolormesh(
            t[:], y, data[:, :], cmap="inferno", norm=norm
        )
    time_vs_freq(ax)
    ax.set_xlabel("time [s]")
    fig.colorbar(img, ax=ax)
    
    if show_bands:
        for f in y:
            ax.plot([0, t[-1]], [f, f], color="white", alpha=0.3)
        ax.set_xlim(0, t[-1])
        

def plot_pulse_train_stats(pulse_train, audio_signal, M, T, pw, IDET, FS, n_electrodes: int = 15):
    f, (ax1,ax2)  = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(np.linspace(0, n_electrodes, pulse_train.shape[0]), pulse_train.max(axis=1), label="max stim")
    ax1.plot(np.linspace(0, n_electrodes, IDET.shape[1]), IDET.min(axis=0), label="min thres")
    ax1.plot(M, label="M")
    ax1.plot(T, label="T")
    ax1.legend()
    ax1.grid()
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Electrode")
    # ax1.set_yscale("symlog")

    ax2.plot(np.linspace(0, n_electrodes, pulse_train.shape[0]), pulse_train.max(axis=1) / IDET.min(axis=0))
    ax2.set_ylabel("max stim / min thres")
    ax2.set_xlabel("Electrode")
    ax2.grid();

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,  figsize=(20, 5))

    t_audio = np.arange(len(audio_signal)) * 1 / FS
    ax1.plot(t_audio, audio_signal)
    ax1.set_xlabel("time [s]")
    ax1.grid()

    e, t = np.where((pulse_train) > 0)
    ax2.scatter(t * pw, e + 1, s=np.abs(pulse_train[e, t]))
    ax2.set_ylim(1, pulse_train.shape[0])
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("electrode")
    ax2.set_ylim(0, pulse_train.shape[0] + 1)

    ax3.hist(np.abs(pulse_train[pulse_train > 0]), bins=50)
    ax3.set_xlabel("stimulus [A]")
    ax3.grid()

    ax4.boxplot(IDET)
    ax4.set_ylabel("threshold [A]")
    ax4.set_xlabel("electrode")
    ax4.grid()


    density = scipy.signal.savgol_filter(pulse_train.sum(axis=0), 2048 * 4, 1).clip(0)
    ax5 = ax2.twinx()
    ax5.plot(np.arange(density.shape[0]) * pw, density, alpha=.4, color="black", zorder=-1)
    ax5.set_ylabel("pulse density")
    plt.tight_layout();
