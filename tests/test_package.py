import unittest

import numpy as np

import abt

VIRTUAL     = 2.7438820088
NO_VIRTUAL  = 3.6952946437


class TestPackage(unittest.TestCase):
    def evaluate(self, expected_n_channels: tuple, expected_value: float, **kwargs):
        name = "tone_1kHz"
        pulse_train, audio_signal = abt.wav_to_electrodogram(abt.sounds[name], **kwargs)
        self.assertEqual(pulse_train.shape[0], expected_n_channels)
        self.assertEqual(pulse_train.shape[1], 111090)
        self.assertTrue(np.isclose(np.abs(pulse_train).sum(), expected_value))
        return pulse_train, audio_signal

    # def test_wav_to_electrodogram_current_steering_no_virtual(self):
    #     self.evaluate(16, NO_VIRTUAL, current_steering=True, virtual_channels=False)

    # def test_wav_to_electrodogram_no_current_steering_no_virtual(self):
    #     pulse_train, audio_signal = self.evaluate(
    #         16, NO_VIRTUAL, current_steering=False, virtual_channels=False
    #     )
    #     self.assertTrue((np.abs(pulse_train).sum(axis=1) > 0).all())
        
    def test_virtual_current_steering(self):
        self.evaluate(121, VIRTUAL, current_steering=True, virtual_channels=True)
        
    def test_virtual_no_current_steering(self):
        self.evaluate(15, VIRTUAL, current_steering=False, virtual_channels=True)
        
    # def test_wav_to_electrodogram_virtual_on_off(self):
    #     pt_virt, _ = self.evaluate(16, EXPECTED_OUTPUT_VIRTUAL / 2, current_steering=True, virtual_channels=True, charge_balanced=False)
    #     pt_no_virt, _ = self.evaluate(16, EXPECTED_OUTPUT_VIRTUAL / 2, current_steering=False, virtual_channels=False, charge_balanced=False)
        
        # self.assertTrue((pt_virt.sum(axis=1) == pt_no_virt.sum(axis=1)).all())
        

    # def test_wav_to_electrodogram_not_balanced(self):
    #     self.evaluate(
    #         16,
    #         EXPECTED_OUTPUT_VIRTUAL / 2,
    #         current_steering=True,
    #         virtual_channels=False,
    #         charge_balanced=False,
    #     )
    #     self.evaluate(
    #         16,
    #         EXPECTED_OUTPUT_VIRTUAL / 2,
    #         current_steering=False,
    #         virtual_channels=False,
    #         charge_balanced=False,
    #     )
    #     self.evaluate(
    #         121,
    #         EXPECTED_OUTPUT_VIRTUAL / 2,
    #         current_steering=True,
    #         virtual_channels=True,
    #         charge_balanced=False,
    #     )
    #     self.evaluate(
    #         16,
    #         EXPECTED_OUTPUT_VIRTUAL / 2,
    #         current_steering=False,
    #         virtual_channels=True,
    #         charge_balanced=False,
    #     )


if __name__ == "__main__":
    unittest.main()
