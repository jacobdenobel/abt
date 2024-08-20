import scipy
import numpy as np
import matplotlib.pyplot as plt


def plot_pulse_train_stats(pulse_train, audio_signal, MAT, IDET, FS):
    f, (ax1,ax2)  = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(np.linspace(0, 15, pulse_train.shape[0]), pulse_train.max(axis=1), label="max stim")
    ax1.plot(np.linspace(0, 15, IDET.shape[1]), IDET.min(axis=0), label="min thres")
    ax1.plot(MAT.M, label="M")
    ax1.plot(MAT.T, label="T")
    ax1.legend()
    ax1.grid()
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Electrode")
    # ax1.set_yscale("symlog")

    ax2.plot(np.linspace(0, 15, pulse_train.shape[0]), pulse_train.max(axis=1) / IDET.min(axis=0))
    ax2.set_ylabel("max stim / min thres")
    ax2.set_xlabel("Electrode")
    ax2.grid();

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,  figsize=(20, 5))

    t_audio = np.arange(len(audio_signal)) * 1 / FS
    ax1.plot(t_audio, audio_signal)
    ax1.set_xlabel("time [s]")
    ax1.grid()

    e, t = np.where((pulse_train) > 0)
    ax2.scatter(t * MAT.pw, e + 1, s=np.abs(pulse_train[e, t]))
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
    ax5.plot(np.arange(density.shape[0]) * MAT.pw, density, alpha=.4, color="black", zorder=-1)
    ax5.set_ylabel("pulse density")
    plt.tight_layout();
