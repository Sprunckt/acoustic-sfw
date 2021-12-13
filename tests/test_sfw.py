import unittest

from src.simulation.simulate_pra import simulate_rir, load_antenna
import src.sfw
import numpy as np
from src.simulation.utils import c, create_grid_spherical


def gammaj(x, fs, n, posm, filt):
    dist = np.linalg.norm(x - posm)
    return filt(n / fs - dist / c) / 4 / np.pi / dist


def plain_gamma(a, x, fs, N, mic_pos, filt):
    n_spikes = len(a)
    M = len(mic_pos)
    J = M*N
    res = np.zeros(J)
    for m in range(M):
        for n in range(N):
            j = N * m + n
            for k in range(n_spikes):
                res[j] += a[k] * gammaj(x[k], fs, n, mic_pos[m], filt)
    return res


def plain_eta(r, res, fs, N, mic_pos, filt):
    M = len(mic_pos)
    y = np.zeros([M, N])
    for m in range(M):
        for n in range(N):
            y[m, n] = gammaj(r, fs, n, mic_pos[m], filt)
    return np.sum(res * y.flatten())


class TestGamma(unittest.TestCase):
    """Testing compliance to PRA simulations and to the plain written formula (with explicit loops)"""
    def test_gamma1(self):
        """Testing compliance to PRA simulations"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 16000
        origin = np.array([0.89, 1, 1.1])
        param_dict1 = dict(mic_array=mic_array1 + origin[np.newaxis, :], src_pos=[1, 2., 0.5],
                           room_dim=[2, 3, 1.5], fs=fs, max_order=1, origin=origin)
        measurements1, N1, src1, ampl1, mic_array1 = simulate_rir(param_dict1)

        sfw = src.sfw.SFW(y=measurements1, mic_pos=mic_array1, fs=16000, N=N1)

        # check that the vectorized code matches the PRA simulations
        reconstr_rir = sfw.gamma(ampl1, src1)
        diff = np.linalg.norm(reconstr_rir/np.max(reconstr_rir) - measurements1/np.max(measurements1))
        self.assertLessEqual(diff, 0.5)

        # check that  the vectorized code matches the explicit formula
        reconstr_rir2 = plain_gamma(ampl1, src1, fs, N=N1, mic_pos=mic_array1, filt=sfw.sinc_filt)
        diff2 = np.linalg.norm(reconstr_rir/np.max(reconstr_rir) - reconstr_rir2/np.max(reconstr_rir2))
        self.assertAlmostEqual(diff2, 0., places=12)

    def test_gamma2(self):
        fs = 4000
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=1.)
        origin = np.array([0.5, 1, 0.89754])
        param_dict1 = dict(mic_array=mic_array1 + origin[np.newaxis, :], src_pos=[1.754201, 9.2114, 0.51213],
                           room_dim=[3, 13.2, 1.5], fs=fs, max_order=2, origin=origin)
        measurements1, N1, src1, ampl1, mic_array1 = simulate_rir(param_dict1)
        sfw = src.sfw.SFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)

        # check that the vectorized code matches the PRA simulations
        reconstr_rir = sfw.gamma(ampl1, src1)
        diff = np.linalg.norm(reconstr_rir/np.max(reconstr_rir) - measurements1/np.max(measurements1))
        self.assertLessEqual(diff, 1)

        # check that  the vectorized code matches the explicit formula
        reconstr_rir2 = plain_gamma(ampl1, src1, fs, N=N1, mic_pos=mic_array1, filt=sfw.sinc_filt)
        diff2 = np.linalg.norm(reconstr_rir / np.max(reconstr_rir) - reconstr_rir2 / np.max(reconstr_rir2))
        self.assertAlmostEqual(diff2, 0., places=12)


class TestEta(unittest.TestCase):
    """Testing compliance to PRA simulations and to the plain written formula (with explicit loops)"""
    def test_eta1(self):
        """Testing compliance to PRA simulations"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 4000
        origin = np.array([0.89, 1, 1.1])
        param_dict1 = dict(mic_array=mic_array1 + origin[np.newaxis, :], src_pos=[1, 2., 0.5],
                           room_dim=[2, 3, 1.5], fs=fs, max_order=1, origin=origin)
        measurements1, N1, src1, ampl1, mic_array1 = simulate_rir(param_dict1)

        sfw = src.sfw.SFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)

        import time

        grid, sph_grid, n_per_sphere = create_grid_spherical(1.124, 3, 0.541, 47, 47, verbose=False)

        # check that the vectorized code matches the explicit formula
        for k in range(1, len(ampl1) - 3):
            a, x = ampl1[:k], src1[:k, :]
            res = sfw.y - sfw.gamma(a, x)
            sfw.res = res
            for r in grid:
                self.assertAlmostEqual(sfw.etak(r.flatten()),
                                       -np.abs(plain_eta(r, res, fs, N1, mic_array1, sfw.sinc_filt)) / sfw.lam)


if __name__ == '__main__':
    unittest.main()
