import unittest

from src.simulation.simulate_pra import simulate_rir, load_antenna
import src.sfw
import numpy as np


class TestGamma(unittest.TestCase):
    def test_gamma1(self):
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        origin = np.array([0.89, 1, 1.1])
        param_dict1 = dict(mic_array=mic_array1 + origin[np.newaxis, :], src_pos=[1, 2., 0.5],
                           room_dim=[2, 3, 1.5], fs=16000, max_order=1, origin=origin)
        measurements1, N1, src1, ampl1, mic_array1 = simulate_rir(param_dict1)

        sfw = src.sfw.SFW(y=measurements1, mic_pos=mic_array1, fs=16000, N=N1)
        reconstr_rir = sfw.gamma(ampl1, src1)
        diff = np.linalg.norm(reconstr_rir/np.max(reconstr_rir) - measurements1/np.max(measurements1))
        self.assertLessEqual(diff, 0.5)

    def test_gamma2(self):
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=1.)
        origin = np.array([0.5, 1, 0.89754])
        param_dict1 = dict(mic_array=mic_array1 + origin[np.newaxis, :], src_pos=[1.754201, 9.2114, 0.51213],
                           room_dim=[3, 13.2, 1.5], fs=8000, max_order=2, origin=origin)
        measurements1, N1, src1, ampl1, mic_array1 = simulate_rir(param_dict1)
        sfw = src.sfw.SFW(y=measurements1, mic_pos=mic_array1, fs=8000, N=N1)
        reconstr_rir = sfw.gamma(ampl1, src1)
        diff = np.linalg.norm(reconstr_rir/np.max(reconstr_rir) - measurements1/np.max(measurements1))
        self.assertLessEqual(diff, 1)


if __name__ == '__main__':
    unittest.main()
