# plt.plot(density)

N = density.shape[0]
fft_signal = np.fft.fft(density)
frequencies = np.fft.fftfreq(N, BINSIZE)

positive_frequencies = frequencies[1:N // 2]
positive_magnitude = np.abs(fft_signal[1:N // 2])

dominating_frequency = positive_frequencies[np.argmax(positive_magnitude)]

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
ax1.plot(positive_frequencies, positive_magnitude)

t = np.arange(0, N) * BINSIZE
sine_wave = np.sin(2 * np.pi * dominating_frequency * t)
ax2.plot(t, sine_wave)
ax3.plot(t, density)
# ax2.plot(t,scipy.signal.savgol_filter(density, 8, 1))
dominating_frequency

# filtered_data = abt.utils.apply_filter(neurogram_freq_bin, window_size=15*20, hop_length=15, scale=True, clip_outliers=.95)

# mel_spec_ng = librosa.db_to_power(filtered_data, ref=65.0)

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
# t = np.linspace(0, duration, len(filtered_data[0]))
# abt.utils.plot_heatmap(t, MEL_SCALE, filtered_data, ax1, f)

# t = np.linspace(0, duration, len(filtered_data[0]))
# abt.utils.plot_heatmap(t, MEL_SCALE, mel_spec_ng, ax2, f)

# # fs = int(1 / (duration / mel_spec_ng.shape[1]))

# s1 = (duration / (1 / FS)) / N_HOP

# mel_inversed = librosa.feature.inverse.mel_to_audio(mel_spec_ng, n_fft=N_FFT, hop_length=int(N_HOP * (s1 / filtered_data.shape[1])), sr=FS, fmin=MIN_FREQ, fmax=MAX_FREQ)

# mel_inversed = scipy.signal.resample(mel_inversed, audio_signal.size)

# mel_inversed.shape

# t_audio = np.arange(len(audio_signal)) * 1 / FS
# plt.figure(figsize=(20, 5))
# plt.plot(t_audio, audio_signal, label="original signal", color="black", alpha=.5)
# plt.xlabel("time [s]")

# t_mel = np.arange(len(mel_inversed)) * 1 / FS

# plt.plot(
#     t_mel, 
#     mel_inversed,
#     label="inverted spectrogram", 
#     color ="red", 
#     linestyle="dashed", 
#     alpha=.5
# )
# sf.write(
#     f"output/reconstructed_{NAME}_ci2.wav", mel_inversed, FS, subtype='PCM_24'
# )