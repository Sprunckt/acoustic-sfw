import unittest
from src.simulation.simulate_pra import simulate_rir, load_antenna
import numpy as np


class TestSim(unittest.TestCase):
    """Compute two RIRs"""

    def test_sim1(self):
        mic_array = load_antenna("data/eigenmike32_cartesian.csv", mic_size=2.)
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1 = simulate_rir(mic_array=mic_array + origin[np.newaxis, :],
                                                                  src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5],
                                                                  fs=16000, max_order=1)
        measurements2, N2, src2, ampl2, mic_array2 = simulate_rir(mic_array=mic_array + origin[np.newaxis, :],
                                                                  src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5], fs=16000,
                                                                  max_order=1, origin=origin)
        self.assertEqual(N1, N2)
        self.assertEqual(ampl1.tolist(), ampl2.tolist())
        self.assertEqual(measurements1.tolist(), measurements2.tolist())
        np.testing.assert_array_almost_equal(src2, src1 - origin)
        np.testing.assert_array_almost_equal(mic_array1, mic_array2 + origin)
        try:
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_sim2(self):
        mic_array2 = np.array([[1.1, 2.1, 1],
                               [1, 0.5, 4]])
        measurements2, N2, src2, ampl2, mic_array2 = simulate_rir(mic_array=mic_array2, src_pos=[2, 1., 0.765],
                                                                  room_dim=[2.74, 3.14, 4.278], fs=8000, max_order=2)


if __name__ == '__main__':
    unittest.main()
