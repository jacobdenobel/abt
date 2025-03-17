import os
import argparse

import numpy as np
import soundfile as sf
import librosa

# Constants
P_REF = 20e-6  # Reference sound pressure in Pa (20 μPa)


def scale_to_spl(waveform, spl_target, current_spl=None):
    """
    Scale a waveform to the desired SPL.

    Parameters:
    - waveform: numpy array, the input sound waveform
    - spl_target: float, target SPL in dB
    - current_spl: float (optional), current SPL if known

    Returns:
    - scaled_waveform: numpy array, scaled waveform
    """

    # Compute current RMS of the waveform
    rms_current = np.sqrt(np.mean(waveform**2))

    if current_spl is None:
        # Assume the waveform's current SPL is relative to P_REF
        current_spl = 20 * np.log10(rms_current / P_REF)

    # Calculate target RMS
    rms_target = P_REF * 10 ** (spl_target / 20)

    # Compute the scaling factor
    scaling_factor = rms_target / rms_current

    # Scale the waveform
    scaled_waveform = waveform * scaling_factor

    return scaled_waveform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--target_spl", type=int, default=130)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    path = os.path.realpath(args.path)
    assert os.path.isfile(path)

    print(args.path)
    wav, sr = librosa.load(path)
    scaled_waveform = scale_to_spl(wav, args.target_spl)

    tgt = args.path
    if not args.overwrite:
        tgt = os.path.join(os.path.dirname(args.path), f"rescaled_{os.path.basename(args.path)}")
    sf.write(tgt, scaled_waveform, sr, subtype="PCM_32")