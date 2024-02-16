import numpy as np
from src.simulation.utils import c, create_grid_spherical

def filt(t, fs):
    """Return the filter function for the sinc kernel."""
    return np.sinc(t * fs)


def filt_der1(t, fs):
    """Return the first order derivative of the sinc kernel."""
    return (np.pi*fs*t*np.cos(np.pi*fs*t) - np.sin(np.pi*fs*t))/(np.pi*fs*t*t+1e-16)


def filt_der2(t, fs):
    """Return the second order derivative of the sinc kernel."""
    return ((2 - (np.pi*fs*t)**2)*np.sin(np.pi*fs*t) - 2*np.pi*fs*t*np.cos(np.pi*fs*t)) / (np.pi*fs) / (t**3+1e-16)


def gamma_mn(r, N, mic_pos, fs):
    """Takes an array of spike locations r (shape (K, 3)) and returns the matrix (gamma_mn(r_k)) of shape (MN, K) of
    kernel evaluations at each spike."""

    # source-microphone distances, shape (M, K)
    dist = np.sqrt(np.sum((r[np.newaxis, :, :] - mic_pos[:, np.newaxis, :]) ** 2, axis=2))

    # shape(MN, K)
    return np.reshape(filt(np.arange(N)[np.newaxis, :, np.newaxis]/fs - dist[:, np.newaxis, :]/c, fs)
                      / 4 / np.pi / dist[:, np.newaxis, :], [len(mic_pos)*N, len(r)])


def gamma_der1(r, N, mic_pos, fs):
    """Takes an array of spike locations r (shape (K, 3)) and returns the matrix (gamma_mn(r_k)) of shape (MN, 3K) of
    kernel first order derivative evaluations at each spike."""

    diff = r[np.newaxis, :, :] - mic_pos[:, np.newaxis, :]  # shape (M, K, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    t = np.arange(N)[np.newaxis, :, np.newaxis]/fs - dist[:, np.newaxis, :] / c  # shape (M, N, K)
    res = np.zeros([len(mic_pos)*N, len(r)*3])

    for i in range(3):
        res[:, i*len(r):(i+1)*len(r)] = np.reshape(- diff[:, :, i][:, np.newaxis, :] / 4 / np.pi * (
                                                   filt_der1(t, fs)/c/dist[:, np.newaxis, :]**2 -
                                                   filt(t, fs) / dist[:, np.newaxis, :]**3), [len(mic_pos)*N, len(r)])
    return res


def gamma_op(r, N, mic_pos, fs):
    nlines, K = len(mic_pos)*N, len(r)
    gamma_mat = np.zeros([nlines, 4*K])

    gamma_mat[:, :K] = gamma_mn(r, N, mic_pos, fs)
    gamma_mat[:, K:] = gamma_der1(r, N, mic_pos, fs)
    return gamma_mat


def pV(r, N, mic_pos, fs):
    """Return the vector pV of length NM used to compute the precertificate"""
    gamma_mat, K = gamma_op(r, N, mic_pos, fs), len(r)
    vec = np.zeros(4*K)
    vec[:K] = 1.
    return np.linalg.pinv(gamma_mat).T @ vec


def etav(rpos, pvec, N, mic_pos, fs):
    return np.sum(pvec[:, np.newaxis] * gamma_mn(np.reshape(rpos, [-1, 3]), N, mic_pos, fs), axis=0)


def reps_sampling(extent, eps, mic_pos, sample_step):
    """Return a list of points in a grid of size extent, with a step sample_step, hollowing balls of radius eps around
    each microphone position."""
    x, y, z = np.meshgrid(np.arange(extent[0, 0], extent[1, 0], sample_step),
                          np.arange(extent[0, 1], extent[1, 1], sample_step),
                          np.arange(extent[0, 2], extent[1, 2], sample_step))
    grid = np.array([x, y, z]).T.reshape(-1, 3)
    mask = np.ones(len(grid), dtype=bool)
    for mic in mic_pos:
        mask = np.logical_and(mask, np.sum((grid - mic)**2, axis=1) > eps**2)
    return grid[mask]


def rmic_sampling(rmin, rmax, rpos, r_step, theta_step):
    """Sample in balls of radius rmax around each microphone position, with a step r_step and theta_step in spherical
    coordinates."""
    grid_ball = create_grid_spherical(rmin=rmin, rmax=rmax, dr=r_step, dtheta=theta_step, dphi=theta_step)[0]
    return np.stack([rpos[i] + grid_ball for i in range(len(rpos))], axis=0).reshape(-1, 3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mic_pos = np.array([[0, 0., 0], [1.20, 0, 0], [0, 1.1, -2.15], [0, 2.54, 1]])
    r = np.array([[0, 2., .64654], [1.54, -2.564987, 0]])
    N = 500
    fs = 16000

    # check derivatives
    gmat = gamma_op(r, N, mic_pos, fs)
    eps = 1e-10
    reps = r.copy()
    reps[:, 1] += eps
    gammaeps = gamma_op(reps, N, mic_pos, fs)
    fd = (gammaeps - gmat) / eps
    plt.plot(fd[:, 0], label="fd")
    plt.plot(gmat[:, 4], "--", label="gmat")
    plt.legend()
    plt.show()

    # check etaV
    grid = rmic_sampling(0.01, 1., r, 0.01, 2)
    print(grid.shape)
    total_size, batch_size = grid.shape[0], 100000
    etaval = np.zeros(total_size)

    for i in range(0, total_size, batch_size):
        if i%100000 == 0:
            print(i)
        etaval[i:i+batch_size] = etav(grid[i:i+batch_size], pV(r, N, mic_pos, fs), N, mic_pos, fs)
    print(np.max(etaval), np.min(etaval))
