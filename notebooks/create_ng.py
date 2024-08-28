import os
import glob
import librosa
import matplotlib
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
from scipy.signal import lfilter

import soundfile as sf

import phast
import abt

from matlab import MatlabData
from utils import create_neurogram, bin_over_y
from plotting import plot_pulse_train_stats

FS = 17400
PATH = glob.glob("/home/nobeljpde1/hearing_model/data/data/original/79*")[0]
PATH = abt.sounds["defineit"]
NAME = os.path.basename(PATH).split(".")[0]
OUTPUT_NAME = f"{NAME}_reconstructed_cs"

N_MELS = 128
N_FFT = 512
N_HOP = 32

MIN_FREQ = 450
MAX_FREQ = 5500

MEL_SCALE = librosa.filters.mel_frequencies(N_MELS, fmin=MIN_FREQ, fmax=MAX_FREQ) 

MAT = MatlabData.read()
CS = True

IDET = MAT.I_det[:, 4::9] if not CS else MAT.I_det

# IDET_FILT = np.zeros(MAT.I_det.shape)
# for e in range(135):
#     IDET_FILT[:, e] = scipy.signal.savgol_filter((MAT.I_det[:, e]), 64, 2) 

# IDET = IDET_FILT 

scaling_factor = 1

n_rep = 1

pulse_train, audio_signal = abt.wav_to_electrodogram(
    PATH, 
    current_steering=CS, 
    charge_balanced=False,  
    virtual_channels=True,
    apply_audiomixer=False,
    ramp_type = 'cs',
    M=(MAT.M * 1e6) / scaling_factor,
    T=(MAT.T * 1e6) / scaling_factor,
    Fs = FS,
    stim_db = 65.0,
    ramp_duration=0.01,
    n_rep=n_rep,
    maxModDepth=0.
)
audio_signal = audio_signal[0]#[start_st:]
duration = len(audio_signal) * (1 / FS)

n_channels = pulse_train.shape[0]
channel_freq = abt.utils.virtual_channel_frequencies(n_channels, MAX_FREQ + 500)
power = 25
w = (-IDET / IDET.sum(axis=1).reshape(-1, 1)) + (2 / len(channel_freq))
w = np.power(w, power) / np.power(w, power).sum(axis=1).reshape(-1, 1)

estimated_freq = w @ channel_freq

stimulus = phast.PulseTrain(pulse_train, time_step=MAT.pw)

selected_fibers = np.arange(3200)

phast.set_seed(1)
fibers = []
for fiber_idx in selected_fibers:
    fibers.append(
        phast.Fiber(
            i_det=IDET[fiber_idx],
            spatial_constant=MAT.spatial_factor(fiber_idx, IDET),
            sigma=MAT.sigma(fiber_idx, IDET),  
            sigma_rs = 0.04, 
            fiber_id=fiber_idx,
            store_stats=False,
            refractory_period = phast.RefractoryPeriod(
                absolute_refractory_period = 4e-4,
                relative_refractory_period = 8e-4,
                sigma_absolute_refractory_period = 0.1e-3,
                sigma_relative_refractory_period = 0.5e-3
            ),
            decay = phast.LeakyIntegratorDecay(
                accommodation_amplitude=0.072, 
                adaptation_amplitude=7.142, 
                accommodation_rate=0.014,
                adaptation_rate=19.996,
                sigma_amp = 0.6e-2, 
                sigma_rate = 0.6e-2, 
            )
        )
    )
    

fiber_stats = phast.phast(fibers, stimulus, n_trials = 20, n_jobs=-1)

max_t = max(fs.spikes[-1] for fs in fiber_stats if len(fs.spikes) > 0)
breakpoint()
BINSIZE = MAT.pw * 2
ng = phast.Neurogram(fiber_stats, BINSIZE, duration, MAT.pw)  

print(ng.data.shape)
print(ng.data)

del ng

print('asdsa')