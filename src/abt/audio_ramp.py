import numpy as np
from abt.frontend import read_wav


# wav_file = 'C:\\python\\temporal\\abt\\sounds\\tone_1kHz.wav'
# audio_signal, *_ = read_wav(wav_file)
# import matplotlib.pyplot as plt


def linear_ramp(audio_signal, ramp_bool=True, ramp_duration=0.05, Fs=17400, **kwargs):
    if ramp_bool:
        if len(audio_signal.shape)>1:
            audio_signal = np.squeeze(audio_signal)
        ramp_length = int(np.ceil(ramp_duration*Fs))
        ramp_vector = np.linspace(0, ramp_duration, ramp_length)

        ramped_audio_signal = audio_signal.copy()
        for ii in np.arange(ramp_length):
            # at beginning
            ramped_audio_signal[ii] = audio_signal[ii]*ramp_vector[ii]/ramp_vector.max()
            # at end
            ramped_audio_signal[-1-ii] = audio_signal[-1-ii]*ramp_vector[ii]/ramp_vector.max()

        # needs to be back into shape of (1, audio_length)
        ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
    else:
        ramped_audio_signal = audio_signal

    return ramped_audio_signal


def cosine_ramp(audio_signal, ramp_bool=True, ramp_duration=0.05, Fs=17400, **kwargs):
    if ramp_bool:
        if len(audio_signal.shape)>1:
            audio_signal = np.squeeze(audio_signal)
        ramp_length = int(np.ceil(ramp_duration*Fs))
        ramp_vector = np.linspace(0, np.pi, ramp_length)

        ramped_audio_signal = audio_signal.copy()

        # at beginning
        ramp_up = -0.5 * np.cos(ramp_vector) + 0.5 
        ramped_audio_signal[:ramp_length] = ramped_audio_signal[:ramp_length] * ramp_up
        # at end 
        ramp_down = ramp_up[::-1]
        ramped_audio_signal[-1-ramp_length:-1] = ramped_audio_signal[-1-ramp_length:-1] * ramp_down
        ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
    else:
        ramped_audio_signal = audio_signal
    
    return ramped_audio_signal



def cosine_squared_ramp(audio_signal, ramp_bool=True, ramp_duration=0.05, Fs=17400, **kwargs):
    if ramp_bool:
        if len(audio_signal.shape)>1:
            audio_signal = np.squeeze(audio_signal)
        ramp_length = int(np.ceil(ramp_duration*Fs))
        ramp_vector = np.linspace(0, np.pi, ramp_length)

        ramped_audio_signal = audio_signal.copy()

        # at beginning
        ramp_up = -0.5 * np.cos(ramp_vector) + 0.5 
        ramped_audio_signal[:ramp_length] = ramped_audio_signal[:ramp_length] * ramp_up**2
        # at end 
        ramp_down = ramp_up[::-1]
        ramped_audio_signal[-1-ramp_length:-1] = ramped_audio_signal[-1-ramp_length:-1] * ramp_down**2
        ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
    else:
        ramped_audio_signal = audio_signal
    
    return ramped_audio_signal

# linear_ramped_audio_signal = linear_ramp(audio_signal)
# cosine_ramped_audio_signal = cosine_ramp(audio_signal)
# cosine_s_ramped_audio_signal = cosine_squared_ramp(audio_signal)


# plt.figure()
# plt.plot(np.squeeze(audio_signal), 'k', label='original')
# plt.plot(np.squeeze(linear_ramped_audio_signal), 'g', label='linear')
# plt.plot(np.squeeze(cosine_ramped_audio_signal), 'b', label='cosine')
# plt.plot(np.squeeze(cosine_s_ramped_audio_signal), 'r', label='cosine squared')
# plt.legend()
# plt.show()


########################################################################
# OOP
import abc
from typing import Any

class AudioRamp(abc.ABC):
    def __init__(self, Fs: float, ramp_bool: bool, ramp_duration:float) -> None: # what if I remove the assigned output?
        self.Fs = Fs
        self.ramp_bool = ramp_bool
        self.ramp_duration = ramp_duration

    @abc.abstractmethod
    def __call__(self, audio_signal: np.ndarray, **kwds: Any) -> Any: # what if I remove the assigned output?
        "Input of time domain filters must be a vector"

class LinearRamp(AudioRamp):
    def __init__(self, Fs: float, ramp_bool: bool, ramp_duration:float, **kwds: Any):
        """""" 
        super().__init__(Fs, ramp_bool, ramp_duration)

    def __call__(self,  audio_signal: np.ndarray, **kwds: Any) -> Any:
        if self.ramp_bool:
            if len(audio_signal.shape)>1:
                audio_signal = np.squeeze(audio_signal)
            ramp_length = int(np.ceil(self.ramp_duration*self.Fs))
            ramp_vector = np.linspace(0, self.ramp_duration, ramp_length)

            ramped_audio_signal = audio_signal.copy()

            for ii in np.arange(ramp_length):
                # at beginning
                ramped_audio_signal[ii] = audio_signal[ii]*ramp_vector[ii]/ramp_vector.max()
                # at end
                ramped_audio_signal[-1-ii] = audio_signal[-1-ii]*ramp_vector[ii]/ramp_vector.max()

            # needs to be back into shape of (1, audio_length)
            ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
        else:
            ramped_audio_signal = audio_signal

        return ramped_audio_signal

class CosineRamp(AudioRamp):
    def __init__(self, Fs: float, ramp_bool: bool, ramp_duration:float, **kwds: Any):
        """""" 
        super().__init__(Fs, ramp_bool, ramp_duration)

    def __call__(self,  audio_signal: np.ndarray, **kwds: Any) -> Any:
        if self.ramp_bool:
            if len(audio_signal.shape)>1:
                audio_signal = np.squeeze(audio_signal)
            ramp_length = int(np.ceil(self.ramp_duration*self.Fs))
            ramp_vector = np.linspace(0, np.pi, ramp_length)

            ramped_audio_signal = audio_signal.copy()

            # at beginning
            ramp_up = -0.5 * np.cos(ramp_vector) + 0.5 
            ramped_audio_signal[:ramp_length] = ramped_audio_signal[:ramp_length] * ramp_up
            # at end 
            ramp_down = ramp_up[::-1]
            ramped_audio_signal[-1-ramp_length:-1] = ramped_audio_signal[-1-ramp_length:-1] * ramp_down
            # needs to be back into shape of (1, audio_length)
            ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
        else:
            ramped_audio_signal = audio_signal

        return ramped_audio_signal

class CosineSquaredRamp(AudioRamp):
    def __init__(self, Fs: float, ramp_bool: bool, ramp_duration:float, **kwds: Any):
        """""" 
        super().__init__(Fs, ramp_bool, ramp_duration)

    def __call__(self,  audio_signal: np.ndarray, **kwds: Any) -> Any:
        if self.ramp_bool:
            if len(audio_signal.shape)>1:
                audio_signal = np.squeeze(audio_signal)
            ramp_length = int(np.ceil(self.ramp_duration*self.Fs))
            ramp_vector = np.linspace(0, np.pi, ramp_length)

            ramped_audio_signal = audio_signal.copy()

            # at beginning
            ramp_up = -0.5 * np.cos(ramp_vector) + 0.5 
            ramped_audio_signal[:ramp_length] = ramped_audio_signal[:ramp_length] * ramp_up**2
            # at end 
            ramp_down = ramp_up[::-1]
            ramped_audio_signal[-1-ramp_length:-1] = ramped_audio_signal[-1-ramp_length:-1] * ramp_down**2
            # needs to be back into shape of (1, audio_length)
            ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
        else:
            ramped_audio_signal = audio_signal

        return ramped_audio_signal


def apply_ramp_type(audio_signal, ramp_type='cs', Fs=17400, ramp_bool=False, ramp_duration=0.05, **kwargs):
    if ramp_bool:
        if ramp_type == 'linear' or ramp_type == 'l':
            ramped_audio_function = LinearRamp(ramp_bool=ramp_bool, ramp_duration=ramp_duration, Fs=Fs, **kwargs)
        elif ramp_type == 'cosine' or ramp_type == 'c':
            ramped_audio_function = CosineRamp(ramp_bool=ramp_bool, ramp_duration=ramp_duration, Fs=Fs, **kwargs)
        elif ramp_type == 'cosine_squared' or ramp_type == 'cs':
            ramped_audio_function = CosineSquaredRamp(ramp_bool=ramp_bool, ramp_duration=ramp_duration, Fs=Fs, **kwargs)
        else:
            ValueError('This ramp type is not an option. Choose from: linear, cosine, cosine_squared')

        return ramped_audio_function(audio_signal)
    else:
        return audio_signal