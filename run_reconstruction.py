import os
import argparse
import glob

import librosa
import scipy
import numpy as np
from scipy.signal import lfilter
import soundfile as sf
from noisereduce import reduce_noise
import matplotlib.pyplot as plt


import phast
import bruce

from notebooks import utils, plotting


N_FFT = 512
N_HOP = 32
REF_DB = 65.0
OUTPUT_FOLDER = "/scratch/jacob/output_folder"
TESTING = False

def echo(*args, **kwargs):
    if TESTING:
        print(*args, **kwargs)
        

def generate_ace():
    raise NotImplementedError()
    tp = phast.load_cochlear()
    (audio_signal, FS), pulse_train, neurogram = phast.ace_e2e(
        path,
        tp=tp,
        scaling_factor=scaling_factor,
        n_trials=n_trials,
    )


def pre_emphasis(signal: np.ndarray, factor: float = 0.97):
    return np.append(signal[0], signal[1:] - (factor * signal[:-1]))


def select_fibers(fiber_freq, mel_scale):
    grouped = np.digitize(fiber_freq, mel_scale, True)
    selected_fibers = []
    for fbin, nf in zip(*np.unique(grouped, return_counts=True)):
        selected_fibers.extend(
            sorted(
                np.random.choice(
                    np.where(grouped == fbin)[0], min(10, nf), replace=False
                )
            )
        )
    return np.array(selected_fibers)


def get_fiber_freq_specres(tp, max_freq, power=25):
    channel_freq = phast.scs.ab.defaults.virtual_channel_frequencies(
        tp.i_det.shape[1], max_freq + 500
    )
    w = (-tp.i_det / tp.i_det.sum(axis=1).reshape(-1, 1)) + (2 / len(channel_freq))
    w = np.power(w, power) / np.power(w, power).sum(axis=1).reshape(-1, 1)
    fiber_freq = w @ channel_freq
    return fiber_freq


def generate_specres(path):
    n_trials = 20
    cs = True
    apply_premph = False
    min_freq = 450
    max_freq = 5500
    n_mels = 64
    mel_scale = librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)

    tp = phast.load_df120()
    fiber_freq = get_fiber_freq_specres(tp, max_freq)
    selected_fibers = select_fibers(fiber_freq, mel_scale)
    fiber_freq = fiber_freq[selected_fibers]

    audio_signal, audio_fs = phast.scs.ab.frontend.read_wav(path, stim_db=REF_DB)
    audio_signal += np.random.normal(0, 1e-20, size=len(audio_signal))
    if apply_premph:
        audio_signal = pre_emphasis(audio_signal[0]).reshape(1, -1)

    (audio_signal, FS), pulse_train, neurogram = phast.ab_e2e(
        audio_signal=audio_signal,
        audio_fs=audio_fs,
        tp=tp,
        current_steering=cs,
        scaling_factor=1.4,
        ramp_duration=(audio_signal.size / audio_fs) * 0.05,
        n_trials=n_trials,
        accommodation_amplitude=0.07,
        adaptation_amplitude=7.142,
        accommodation_rate=2,
        adaptation_rate=19.996,
        selected_fibers=selected_fibers,
    )
    audio_signal = audio_signal[0]
    neurogram_data = neurogram.data / n_trials
    binned_data = utils.bin_over_y(
        neurogram_data, fiber_freq, mel_scale, agg=np.mean
    )
    dm = binned_data.max()
    
    data = utils.smooth(binned_data, "hann", 128, 1)
    data *= dm / data.max()
    echo(data.min(), data.mean(), data.max())
    
    return data, neurogram.binsize, min_freq, max_freq, n_mels


def generate_bruce(path):
    bruce.set_seed(31)
    stim = bruce.stimulus.from_file(path, False)

    n_trials = 20
    n_fibers = 10
    
    n_low_med = int(np.floor(n_fibers / 5))
    n_high = n_fibers - (2 * n_low_med)
    
    min_freq = 150
    max_freq = 10_500
    n_bins = 1
    n_mels = 64 
    
    mel_scale = librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)

    ng = bruce.Neurogram(
        mel_scale,
        n_low=n_low_med,
        n_med=n_low_med,
        n_high=n_high,
    )
    # Takes more time
    ng.bin_width = 1 / (max_freq * 2 * n_bins)
    from time import perf_counter

    time = perf_counter()
    ng.create(stim, n_rep=1, n_trials=n_trials)
    echo(perf_counter() - time)

    data = ng.get_output()
    echo("n spikes", data.sum())
    echo(data.min(), data.max())
    # breakpoint()
    
    breakpoint()
    data /= 0.8 * (n_high * n_trials)
    dm = data.max()

    data = utils.smooth(data, "hann", 2048, n_bins)
    data *= dm / data.max()
    echo(data.min(), data.max())
    
    return data, n_bins * ng.bin_width, min_freq, max_freq, n_mels


def neurogram_to_wav(data, binsize, audio_size, min_freq, max_freq):
    # Clip outliers
    data = data.clip(0, np.quantile(data.ravel(), 0.999))
    echo(data.min(), data.max())
    
    # Resample
    n_s = int(np.ceil(data.shape[1] / N_HOP))
    data = np.array([scipy.signal.resample(x, n_s) for x in data]).clip(0, 1)
    echo(data.min(), data.max())
    
    # Scale to -80, 0
    data = utils.min_max_scale(data, data_min=0, data_max=1)
    echo(data.min(), data.max())
    
    # Convert to power
    data = librosa.db_to_power(data, ref=REF_DB)
    echo(data.min(), data.max())
    
    reconstructed = librosa.feature.inverse.mel_to_audio(
        data,
        n_fft=N_FFT,
        hop_length=N_HOP,
        sr=np.ceil(1 / binsize),
        fmin=min_freq,
        fmax=max_freq,
    )
    return scipy.signal.resample(reconstructed, audio_size)


def plot(audio_signal, reconstructed_signal, audio_fs, n_mels, min_freq, max_freq, title):
    fig = plt.figure()
    fig.suptitle(title)
    gs = plt.GridSpec(2, 2, wspace=0.25, hspace=0.25) # 2x2 grid
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    
    for i, (ax, sig) in enumerate(zip((ax0, ax1), (audio_signal, reconstructed_signal))):
        S = librosa.feature.melspectrogram(
            y=sig, 
            sr=audio_fs,
            n_mels=n_mels, 
            fmin=min_freq, 
            fmax=max_freq
        )
      
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=audio_fs,
                            fmin=min_freq, 
                            fmax=max_freq, ax=ax)
        ax.set_xlabel(None)
        if i == 1:
            ax.set_title("reconstructed")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
        else:
            ax.set_title("original")

    t = np.arange(audio_signal.size) / audio_fs
    ax2.plot(t, audio_signal, label='original', alpha=.6)
    ax2.plot(t, reconstructed_signal, label='reconstructed', alpha=.6)
    ax2.grid()
    ax2.legend()
    ax2.set_xlabel("time [s]")
    plt.savefig("x.pdf")
    
    
def reconstruct(path, output_path, neurogram_generator):
    path = os.path.realpath(path)
    assert os.path.isfile(path)
    name = os.path.basename(path).split(".")[0]

    neurogram_generator = NEUROGRAM_GENERATORS[neurogram_generator]
    method_name = neurogram_generator.__name__.replace("generate_", "")
    
    # output_path = os.path.join(output_path, method_name)
    output_file = os.path.join(output_path, f"{name}.wav")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.exists(output_file):
        if not TESTING:
            return
        os.remove(output_file)

    audio_signal, audio_fs = librosa.load(path, sr=None)

    neurogram_freq_bin, binsize, min_freq, max_freq, n_mels = neurogram_generator(path)
    reconstructed = neurogram_to_wav(
        neurogram_freq_bin, binsize, audio_signal.size, min_freq, max_freq
    )
    if TESTING:
        plot(audio_signal, reconstructed, audio_fs, n_mels, min_freq, max_freq, f"{method_name}_{name}")
    sf.write(output_file, reconstructed, audio_fs, subtype="PCM_32")


NEUROGRAM_GENERATORS = [
    generate_specres,
    generate_bruce,
    generate_ace,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--neurogram_generator",
        default=0,
        choices=range(len(NEUROGRAM_GENERATORS)),
        type=int,
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--testing", action='store_true')
    args = parser.parse_args()
    TESTING = args.testing
    path = os.path.normpath(args.path)

    if os.path.isdir(path) and path.endswith("corpus"):
        files = glob.glob(f"{args.path}/*/*/*/*/*.wav")
        for i, file in enumerate(files, 1):
            echo(f"{i}/{len(files)}: {file.replace(path, '')}")
            output_folder = os.path.join(OUTPUT_FOLDER, os.path.dirname(file))
            reconstruct(file, output_folder, args.neurogram_generator)

    elif os.path.isfile(path):
        reconstruct(
            args.path, args.output_path or OUTPUT_FOLDER, args.neurogram_generator
        )
