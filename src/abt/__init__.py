import os

import numpy as np
from scipy.signal import lfilter

from . import frontend
from . import buffer
from . import automatic_gain_control
from . import filterbank
from . import noise_reduction
from . import post_filterbank
from . import mapping
from . import electrodogram

# from . import vocoder
from . import audiomixer
from . import audio_ramp


base_dir = os.path.realpath(os.path.dirname(__file__))
sound_dir = os.path.join(base_dir, "data/sounds")

sounds = {
    x[:-4]: os.path.join(sound_dir, x)
    for x in os.listdir(sound_dir)
    if x.endswith(".wav")
}


def pulse_train_to_virtual(
    pulse_train: np.ndarray,
    weights_matrix: np.ndarray,
    n_virtual: int = 8,
    charge_balanced: bool = False,
):
    (num_electrodes, num_samples) = weights_matrix.shape
    n_virtual_channels = (num_electrodes - 1) * n_virtual + 1
    pulse_times, pulse_electrodes = np.where(pulse_train.T < 0)
    pulse_train_virtual = np.zeros((n_virtual_channels, num_samples))

    weights_map = (
        {(0.5, 0.5): 1}
        if n_virtual == 1
        else {
            (float(x), 1 - float(x)): e
            for e, x in enumerate(np.arange(0, 1.1, 1 / n_virtual)[::-1], 1)
        }
    )

    for el in range(num_electrodes):
        pulse_times_electrode = pulse_times[pulse_electrodes == el]

        if el == 15:
            el -= 1  # only loop over electrode, don't add to count
            el_pair = [14, 15]
        else:
            el_pair = [el, el + 1]

        for pt in pulse_times_electrode:
            weights_pair = tuple(map(float, weights_matrix[el_pair, pt]))
            if weights_pair not in weights_map:
                continue
            virtual_channel_id = int(weights_map[weights_pair] + el * n_virtual - 1)
            pulse_pair = pulse_train[el_pair, pt]
            pulse_train_virtual[virtual_channel_id, pt] = np.sum(pulse_pair)

    if charge_balanced:
        return lfilter(np.array([1, -1]), 1, pulse_train_virtual)
    return pulse_train_virtual


# perhaps we can get some data on which electrode array the patients have and such? Or too few people?
# overview of system: https://www.youtube.com/watch?v=6Xq29_ci6Ko&t=596s
# according to link: Fs = 17400 Hz
# (So Nyquist frequency = 8700 Hz)
# frequency resolution = 68 Hz
# Maximum channel stimulation rate (CSR) = 2899 Hz, but elsewhere in the article they say its 2320 Hz?
def wav_to_electrodogram(
    wav_file: str,
    apply_audiomixer=True,
    virtual_channels=True,
    charge_balanced=True,
    **kwargs,
):
    audio_signal, *_ = frontend.read_wav(wav_file, **kwargs)

    if apply_audiomixer:
        audio_signal, *_ = audiomixer.audiomixer(audio_signal, **kwargs)

    # add ramp to the stimulus to avoid any un-smoothness/discontinuity at onset
    audio_signal = audio_ramp.apply_ramp_type(audio_signal, **kwargs)

    # Applies an IIR (actually highpass filter) to the audio signal, but this works as a pre-emphasis filter:
    # (1) balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies,
    # (2) avoid numerical problems during the Fourier transform operation and
    # (3) may also improve the Signal-to-Noise Ratio (SNR). See https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    # Another way to apply pre-emphasis:
    # pre_emphasis = 0.97  # will become a hyperparameter for Jacob
    # emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    signal = frontend.td_filter(audio_signal, **kwargs)
    # AGC to reduce signals with too much strength, how can we alter this so that it can be optimized? Moreover, there is also another FIR
    # in this application?
    # dual loop for fast and slow acting part
    signal, agc = automatic_gain_control.dual_loop_td_agc(signal, **kwargs)
    # windowing to reduce spectral leakage
    signal = buffer.window_buffer(signal, **kwargs)
    # Makes the audio input of lenght 256, so fft over 0.0147 seconds. FFT length gives good compromise between spectral and temporal resolution
    signal_fft = filterbank.fft_filterbank(signal, **kwargs)
    # calculate the envelope per frequency band of each channel?
    signal_hilbert = filterbank.hilbert_envelope(signal_fft, **kwargs)
    # calculate amplitude just by taking the square root of the power per channel? Why first squaring + absolute and then square root?
    signal_energy = filterbank.channel_energy(signal_fft, agc.smpGain, **kwargs)
    # Compute channel-by-channel noise reduction gains.
    signal_energy, *_ = noise_reduction.noise_reduction(signal_energy, **kwargs)
    # why would you sum the envelope with the energy?
    signal = signal_hilbert + signal_energy
    # Find frequency band with largest amplitude of subsample (every third FFT input frame)
    peak_freq, peak_loc = post_filterbank.spec_peak_locator(
        signal_fft[:, 2::3], **kwargs
    )
    # upsample back to full framerate and add padding
    peak_freq = post_filterbank.upsample(peak_freq, signal_fft.shape[1], **kwargs)
    peak_loc = post_filterbank.upsample(peak_loc, signal_fft.shape[1], **kwargs)

    # # CS provides 8 "virtual" electrodes so 8*15 = 120 "virtual" electrodes in total
    weights = post_filterbank.current_steering_weights(peak_loc, **kwargs)

    # Create carrier function with period of 1/peak_freq, maximum depends on implant's maximal stimulation rate
    carrier, audio_idx = post_filterbank.carrier_synthesis(peak_freq, **kwargs)

    signal = mapping.f120(carrier, signal, weights, audio_idx, **kwargs)

    pulse_train, weights_matrix = electrodogram.f120(
        signal, weights=weights[:, audio_idx], **kwargs
    )

    if virtual_channels:
        n_virtual_channels = 8 if kwargs.get("current_steering", True) else 1
        pulse_train = pulse_train_to_virtual(
            pulse_train, weights_matrix, n_virtual_channels, charge_balanced
        )

    if not charge_balanced:
        pulse_train[pulse_train > 0] = 0

    pulse_train *= 1e-6

    return pulse_train, audio_signal
