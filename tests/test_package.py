import unittest

import numpy as np

import abt


class TestPackage(unittest.TestCase):
    def evaluate(self, expected_n_channels: tuple, expected_value: float, **kwargs):
        name = "tone_1kHz"
        pulse_train, audio_signal = abt.wav_to_electrodogram(abt.sounds[name], **kwargs)
        self.assertEqual(pulse_train.shape[0], expected_n_channels)
        self.assertEqual(pulse_train.shape[1], 111090)
        self.assertTrue(np.isclose(np.abs(pulse_train).sum(), expected_value))
        return pulse_train, audio_signal

    def test_wav_to_electrodogram_current_steering_no_virtual(self):
        self.evaluate(16, 3.6952946437, current_steering=True, virtual_channels=False)

    def test_wav_to_electrodogram_no_current_steering_no_virtual(self):
        pulse_train, audio_signal = self.evaluate(
            16, 3.6952946437, current_steering=False, virtual_channels=False
        )
        self.assertTrue((np.abs(pulse_train).sum(axis=1) > 0).all())
        
        
    def test_wav_to_electrodogram_current_steering(self):
        pulse_train, audio_signal = self.evaluate(121, 3.6952946437, current_steering=True, virtual_channels=True)
        breakpoint()

    def test_wav_to_electrodogram_no_current_steering(self):
        self.evaluate(16, 3.6952946437, current_steering=False, virtual_channels=True)

    def test_wav_to_electrodogram_not_balanced(self):
        self.evaluate(
            16,
            3.6952946437 / 2,
            current_steering=True,
            virtual_channels=False,
            charge_balanced=False,
        )
        self.evaluate(
            16,
            3.6952946437 / 2,
            current_steering=False,
            virtual_channels=False,
            charge_balanced=False,
        )
        self.evaluate(
            121,
            3.6952946437 / 2,
            current_steering=True,
            virtual_channels=True,
            charge_balanced=False,
        )
        self.evaluate(
            16,
            3.6952946437 / 2,
            current_steering=False,
            virtual_channels=True,
            charge_balanced=False,
        )


if __name__ == "__main__":
    unittest.main()
