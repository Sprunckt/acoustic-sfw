import numpy as np
from src.simulation.utils import c


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mic_pos = np.array([[0, 0., 0], [1.20, 0, 0], [0, 1, 0], [0, 0, 1]])
    r = np.array([[0, 2., .64654], [1.54, 3.564987, 0]])
    N = 500
    fs = 8000

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

