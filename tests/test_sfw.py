import unittest

from src.simulation.simulate_pra import simulate_rir, load_antenna
import src.sfw
import numpy as np
from src.simulation.utils import c, create_grid_spherical


def gammaj(x, fs, n, posm, filt):
    dist = np.linalg.norm(x - posm)
    return filt(n / fs - dist / c) / 4 / np.pi / dist


def gammaj_der(x, fs, n, posm, filt, filt_der):
    dist = np.linalg.norm(x - posm)
    diff = x - posm
    int_term = n/fs - dist/c
    return diff * (-dist*filt_der(int_term)/c - filt(int_term)) / 4 / np.pi / dist**3


def gammaj_der_norm2(x, fs, n, m, allposm, filt, filt_der, M, N, norm):
    common_term = gammaj(x, fs, n, allposm[m], filt) * 4 * np.pi / norm**3
    common_add = - 1 / norm / np.linalg.norm(x-allposm[m])**2
    res = np.zeros(3)
    for mm in range(M):
        diff = x - allposm[mm]
        dist2 = np.linalg.norm(diff)

        for nn in range(N):
            int_term2 = nn / fs - dist2 / c

            if nn == n and mm == m:
                add_term = common_add
            else:
                add_term = 0

            res += (diff * (filt_der(int_term2)/c + filt(int_term2) / dist2)
                    * (common_term * 4*np.pi*gammaj(x, fs, nn, allposm[mm], filt) / dist2 ** 2
                    + add_term))
    return res


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


def plain_gamma_norm2(a, x, fs, N, mic_pos, filt):
    n_spikes = len(a)
    M = len(mic_pos)
    J = M*N
    res = np.zeros([J, n_spikes])
    norms = np.zeros(n_spikes)
    for k in range(n_spikes):
        for m in range(M):
            for n in range(N):
                j = N * m + n
                gammaval = gammaj(x[k], fs, n, mic_pos[m], filt)
                norms[k] += gammaval**2
                res[j, k] += gammaval

    norms = np.sqrt(norms)

    return np.sum(a[np.newaxis, :] * res / norms[np.newaxis, :], axis=1)


def plain_eta(r, res, fs, N, mic_pos, filt):
    M = len(mic_pos)
    y = np.zeros([M, N])
    for m in range(M):
        for n in range(N):
            y[m, n] = gammaj(r, fs, n, mic_pos[m], filt)
    return -np.sum(res * y.flatten())


def plain_slide_jac(a, x, y, fs, N, mic_pos, filt, filt_der, lam):
    K, M = len(a), len(mic_pos)
    jac = np.zeros(4*K)
    residue = plain_gamma(a, x, fs, N, mic_pos, filt) - y
    for k in range(K):  # derivative in ak
        for m in range(M):
            for n in range(N):
                jac[k] += gammaj(x[k], fs, n, mic_pos[m], filt) * residue[m*N + n]
        jac[k] += lam * np.sign(a[k])
    for k in range(K):  # derivative in xk
        main_term = np.zeros(3)
        for m in range(M):
            for n in range(N):
                main_term += gammaj_der(x[k], fs, n, mic_pos[m], filt, filt_der) * residue[m*N + n]
        jac[K + 3*k: K + 3*(k+1)] = a[k] * main_term

    return jac


def plain_slide_jac_norm2(a, x, y, fs, N, mic_pos, filt, filt_der, lam):
    K, M = len(a), len(mic_pos)
    jac = np.zeros(4*K)
    residue = plain_gamma_norm2(a, x, fs, N, mic_pos, filt) - y
    norms = np.zeros(K)

    for k in range(K):  # derivative in ak
        for m in range(M):
            for n in range(N):
                gamma_val = gammaj(x[k], fs, n, mic_pos[m], filt) * 4 * np.pi
                norms[k] += gamma_val**2
                jac[k] += gamma_val * residue[m*N + n]

    norms = np.sqrt(norms)
    jac[:K] = jac[:K] / norms + lam*np.sign(a)

    for k in range(K):  # derivative in xk
        main_term = np.zeros(3)
        for m in range(M):
            for n in range(N):
                main_term += (gammaj_der_norm2(x[k], fs, n, m, mic_pos, filt, filt_der, M, N, norms[k])
                              * residue[m*N + n])
        jac[K + 3*k: K + 3*(k+1)] = a[k] * main_term

    return jac


def plain_eta_jac(x, res, fs, N, mic_pos, filt, filt_der, lam):
    M = len(mic_pos)
    jac = np.zeros(3)
    for m in range(M):
        for n in range(N):
            jac += gammaj_der(x, fs, n, mic_pos[m], filt, filt_der) * res[m*N + n]
    return -jac


class TestGamma(unittest.TestCase):
    """Testing compliance to PRA simulations and to the plain written formula (with explicit loops)"""
    def test_gamma1(self):
        """Testing compliance to PRA simulations"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 16000
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5], fs=fs,
                                                                     max_order=1, origin=origin)

        sfw = src.sfw.TimeDomainSFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)

        # check that the vectorized code matches the PRA simulations
        reconstr_rir = sfw.gamma(ampl1, src1)
        diff = np.linalg.norm(reconstr_rir - measurements1)
        self.assertLessEqual(diff, 0.1)

        # check that  the vectorized code matches the explicit formula
        reconstr_rir2 = plain_gamma(ampl1, src1, fs, N=N1, mic_pos=mic_array1, filt=sfw.sinc_filt)
        diff2 = np.linalg.norm(reconstr_rir - reconstr_rir2)
        self.assertAlmostEqual(diff2, 0., places=12)

    def test_gamma2(self):
        fs = 4000
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=1.)
        origin = np.array([0.5, 1, 0.89754])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1.754201, 9.2114, 0.51213],
                                                                     room_dim=[3, 13.2, 1.5], fs=fs,
                                                                     max_order=2, origin=origin)
        sfw = src.sfw.TimeDomainSFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)

        # check that the vectorized code matches the PRA simulations
        reconstr_rir = sfw.gamma(ampl1, src1)
        diff = np.linalg.norm(reconstr_rir - measurements1)
        self.assertLessEqual(diff, 0.1)

        # check that  the vectorized code matches the explicit formula
        reconstr_rir2 = plain_gamma(ampl1, src1, fs, N=N1, mic_pos=mic_array1, filt=sfw.sinc_filt)
        diff2 = np.linalg.norm(reconstr_rir - reconstr_rir2)
        self.assertAlmostEqual(diff2, 0., places=12)

    def test_gamma_norm2(self):
        """Testing compliance to PRA simulations"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)[:10]
        fs = 6000
        N1 = 500
        rand_gen = np.random.RandomState()
        rand_gen.seed(1)

        measurements1 = rand_gen.random(N1 * mic_array1.shape[0])

        sfw = src.sfw.TimeDomainSFWNorm2(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)
        sfw.nk = 10
        ampl1, src1 = rand_gen.random(sfw.nk), rand_gen.random([sfw.nk, 3])
        # check that  the vectorized code matches the explicit formula
        reconstr_rir = sfw.gamma(ampl1, src1)
        reconstr_rir2 = plain_gamma_norm2(ampl1, src1, fs, N=N1, mic_pos=mic_array1, filt=sfw.sinc_filt)
        diff2 = np.linalg.norm(reconstr_rir - reconstr_rir2)
        self.assertAlmostEqual(diff2, 0., places=12)


class TestEta(unittest.TestCase):
    """Testing compliance to PRA simulations and to the plain written formula (with explicit loops)"""
    def test_eta1(self):
        """Testing compliance to PRA simulations"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 4000
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5],
                                                                     fs=fs, max_order=1, origin=origin)

        sfw = src.sfw.TimeDomainSFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)

        grid, sph_grid, n_per_sphere = create_grid_spherical(1.124, 3, 0.541, 47, 47, verbose=False)

        # check that the vectorized code matches the explicit formula
        for k in range(1, len(ampl1) - 3):
            a, x = ampl1[:k], src1[:k, :]
            res = sfw.y - sfw.gamma(a, x)
            sfw.res = res
            for r in grid:
                self.assertAlmostEqual(sfw.etak(r.flatten()),
                                       float(plain_eta(r, res, fs, N1, mic_array1, sfw.sinc_filt)))

    def test_jac_slide(self):
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 4000
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5],
                                                                     fs=fs, max_order=1, origin=origin)

        sfw = src.sfw.TimeDomainSFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)
        sfw.nk = 5  # number of sources considered

        # point used for comparison
        var = np.arange(1, sfw.nk*4 + 1, dtype=float)

        sfw_jac = sfw._jac_slide_obj(var, y=sfw.y, n_spikes=sfw.nk)
        plain_jac = plain_slide_jac(var[:sfw.nk], var[sfw.nk:].reshape(sfw.nk, 3),
                                    y=sfw.y, fs=fs, N=N1, mic_pos=sfw.mic_pos,
                                    filt=sfw.sinc_filt, filt_der=sfw.sinc_der, lam=sfw.lam)
        for t in range(4*sfw.nk):
            self.assertAlmostEqual(sfw_jac[t], plain_jac[t])

    def test_jac_slide_norm2(self):
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)[:5]
        fs = 1000
        N1 = 75
        rand_gen = np.random.RandomState()
        rand_gen.seed(1)

        measurements1 = rand_gen.random(N1*mic_array1.shape[0])

        sfw = src.sfw.TimeDomainSFWNorm2(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)
        sfw.nk = 5 # number of sources considered

        # point used for comparison
        var = rand_gen.random(sfw.nk*4) + 0.25
        var[sfw.nk:] += 0.75

        sfw_jac = sfw._jac_slide_obj(var, y=sfw.y, n_spikes=sfw.nk)
        plain_jac = plain_slide_jac_norm2(var[:sfw.nk], var[sfw.nk:].reshape(sfw.nk, 3),
                                          y=sfw.y, fs=fs, N=N1, mic_pos=sfw.mic_pos,
                                          filt=sfw.sinc_filt, filt_der=sfw.sinc_der, lam=sfw.lam)
        dt = 1e-6

        for t in range(4 * sfw.nk):
            # test compliance to plain written formula
            self.assertAlmostEqual(sfw_jac[t], plain_jac[t])
            varm, varp = var.copy(), var.copy()
            varp[t] += dt
            # test compliance to finite differences
            fd = (sfw._obj_slide(varp, y=sfw.y, n_spikes=sfw.nk) - sfw._obj_slide(varm, y=sfw.y, n_spikes=sfw.nk)) / dt
            self.assertAlmostEqual(sfw_jac[t], fd, places=4)

    def test_jac_eta1(self):
        """Test compliance to the plain written eta jacobian without normalization"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 4000
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5],
                                                                     fs=fs, max_order=1, origin=origin)

        sfw = src.sfw.TimeDomainSFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)
        sfw.nk = 5  # number of sources considered

        # point used for comparison
        lvar = [np.arange(1, 4, dtype=float), np.array([0.321, -4, -1.342])]
        for var in lvar:
            sfw_jac = sfw._jac_etak(var)
            plain_jac = plain_eta_jac(var, res=sfw.y, fs=fs, N=N1, mic_pos=sfw.mic_pos,
                                      filt=sfw.sinc_filt, filt_der=sfw.sinc_der, lam=sfw.lam)
            for t in range(3):
                self.assertAlmostEqual(sfw_jac[t], plain_jac[t])

    def test_jac_eta_norm1(self):
        """Test compliance to the plain written eta jacobian without normalization"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 4000
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5], fs=fs,
                                                                     max_order=1, origin=origin)

        sfw = src.sfw.TimeDomainSFW(y=measurements1, mic_pos=mic_array1, fs=fs, N=N1)
        sfw.nk = 2  # number of sources considered

        # point used for comparison
        lvar = [np.arange(1, 4), np.array([0.321, -4, -1.342])]
        for var in lvar:
            sfw_jac = sfw._jac_etak(var)
            plain_jac = plain_eta_jac(var, res=sfw.y, fs=fs, N=N1, mic_pos=sfw.mic_pos,
                                      filt=sfw.sinc_filt, filt_der=sfw.sinc_der, lam=sfw.lam)
            for t in range(3):
                self.assertAlmostEqual(sfw_jac[t], plain_jac[t])


class TestGammaFreq(unittest.TestCase):
    """Testing compliance between the DFT of the ideal RIR and the ideal FT model"""
    def test_gamma1(self):
        """Testing compliance to PRA simulations"""
        mic_array1 = load_antenna("data/eigenmike32_cartesian.csv", mic_size=3.)
        fs = 16000
        origin = np.array([0.89, 1, 1.1])

        measurements1, N1, src1, ampl1, mic_array1, _ = simulate_rir(mic_array=mic_array1 + origin[np.newaxis, :],
                                                                     src_pos=[1, 2., 0.5], room_dim=[2, 3, 1.5], fs=fs,
                                                                     max_order=1, origin=origin)
        M = len(mic_array1)
        sfw = src.sfw.TimeDomainSFW(y=(ampl1, src1), mic_pos=mic_array1, fs=16000, N=N1)

        fft = np.fft.rfft(sfw.y.reshape(M, N1), axis=-1).flatten() / np.sqrt(2*np.pi)

        freq_sfw = src.sfw.FrequencyDomainSFW(y=sfw.y, N=N1, mic_pos=mic_array1, fs=fs)
        model_ft = freq_sfw.gamma(ampl1, src1)

        # check that the vectorized code matches the PRA simulations
        diff = np.linalg.norm((fft - model_ft)[:-1])
        self.assertLessEqual(diff, 0.2)


if __name__ == '__main__':
    unittest.main()
