import unittest

import numpy as np
import matplotlib.pyplot as plt

import abt

TOTAL_INJECTED_CURRENT = 3.6952946437


class TestPackage(unittest.TestCase):
    def evaluate(
        self,
        expected_n_channels: tuple,
        expected_output: float = TOTAL_INJECTED_CURRENT,
        **kwargs,
    ):
        name = "tone_1kHz"
        pulse_train, audio_signal = abt.wav_to_electrodogram(abt.sounds[name], **kwargs)
        self.assertEqual(pulse_train.shape[0], expected_n_channels)
        self.assertEqual(pulse_train.shape[1], 111090)
        self.assertAlmostEqual(np.abs(pulse_train).sum(), expected_output)
        return pulse_train, audio_signal

    def test_wav_to_electrodogram_current_steering_no_virtual(self):
        self.evaluate(16, current_steering=True, virtual_channels=False)

    def test_wav_to_electrodogram_no_current_steering_no_virtual(self):
        pulse_train, audio_signal = self.evaluate(
            16, current_steering=False, virtual_channels=False
        )
        self.assertTrue((np.abs(pulse_train).sum(axis=1) > 0).all())

    def test_virtual_current_steering(self):
        self.evaluate(121, current_steering=True, virtual_channels=True)

    def test_virtual_no_current_steering(self):
        self.evaluate(15, current_steering=False, virtual_channels=True)

    def test_wav_to_electrodogram_not_balanced(self):
        self.evaluate(
            16,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=True,
            virtual_channels=False,
            charge_balanced=False,
        )
        self.evaluate(
            16,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=False,
            virtual_channels=False,
            charge_balanced=False,
        )
        self.evaluate(
            121,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=True,
            virtual_channels=True,
            charge_balanced=False,
        )
        self.evaluate(
            15,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=False,
            virtual_channels=True,
            charge_balanced=False,
        )

    def test_ordering_virtual_channels(self):
        for cs in (False, True):
            n_channels = 121 if cs else 15
            pulse_train, _ = self.evaluate(
                n_channels,
                TOTAL_INJECTED_CURRENT / 2,
                current_steering=cs,
                virtual_channels=True,
                charge_balanced=False,
            )
            
            channel_freq = abt.utils.virtual_channel_frequencies(n_channels)
            max_freq = channel_freq[pulse_train.sum(axis=1).argmax()]
            closest_to_1kHz = abt.utils.find_nearest(channel_freq, 1000)
            self.assertEqual(max_freq, closest_to_1kHz)
            
if __name__ == "__main__":
    unittest.main()
