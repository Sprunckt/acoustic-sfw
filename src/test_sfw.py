
"""
Basic working script for testing the algorithm
"""

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from src.simulation.utils import (multichannel_rir_to_vec, vec_to_rir, create_grid_spherical,
                                  compare_arrays)
from src.sfw import SFW

# Scene Parameters
room_dim = [4, 6, 3.2]
max_order = 1
freq_sampling = 16000  # Hz
all_flat_materials = {
    "east": pra.Material(0.1),
    "west": pra.Material(0.1),
    "north": pra.Material(0.1),
    "south": pra.Material(0.1),
    "ceiling": pra.Material(0.1),
    "floor": pra.Material(0.1),
}

src_pos = [3, 4, 1.8]
mic_pos = [2, 3, 1.5]
mic_size = 5

# Create the Room
room = pra.ShoeBox(room_dim, fs=freq_sampling,
                   materials=all_flat_materials, max_order=max_order)

# Add a source somewhere in the room
room.add_source(src_pos)

# Load the eigenmike32 spherical microphone array
# Source: https://www.locata.lms.tf.fau.de/files/2020/01/Documentation_LOCATA_final_release_V1.pdf
mic_array = np.transpose(np.genfromtxt('data/eigenmike32_cartesian.csv', delimiter=', '))

# Translate to desired position and scale
mic_array = mic_size * mic_array + np.reshape(mic_pos, [3, 1])

room.add_microphone_array(mic_array)

# Simulate RIR with image source method
room.compute_rir()

# assemble the multichannel rir in a single array
measurements, N, M = multichannel_rir_to_vec(room.rir)  # N, M : number of time samples and microphones

# get the image sources and corresponding amplitudes
src = room.sources[0].get_images(max_order=max_order).T
ampl = room.sources[0].get_damping(max_order=1).flatten()

# translate the sources so the origin is at the center of the microphones
src = src - np.reshape(mic_pos, [1, 3])
print("sources :", src)

d = 3  # dimension of the problem
mic_array = mic_array.T  # positions, shape (M, d)

# translate the microphones so the center of the antenna is the origin
mic_array = mic_array - np.reshape(mic_pos, [1, 3])

J = N * M

plot_sources = False
if plot_sources:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(src[:, 0], src[:, 1], src[:, 2])
    ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
    plt.show()

# create a spherical grid
grid, sph_grid, n_sph = create_grid_spherical(1, 15, 0.3, 25, 25)
print("grid shape : ", grid.shape)

s = SFW(measurements/np.max(measurements), mic_pos=mic_array, fs=freq_sampling, N=N, lam=1e-2)

plot_rir = False
if plot_rir:  # plot the simulated RIR and the RIR computed using the measure gamma and the real source positions
    m = 5
    compg = vec_to_rir(s.gamma(ampl, src), m, N)
    plt.plot(np.arange(N)/freq_sampling, compg/np.max(compg), label='gamma')
    rir = vec_to_rir(measurements, m, N)
    plt.plot(np.arange(N)/freq_sampling, rir/np.max(rir), '-.', label='pyroom')
    plt.xlabel("t (s)"), plt.ylabel("p")
    plt.legend()
    plt.show()

load = False
if load:
    res = np.load('../mes.npz')
    x, a, r = res['x'], res['a'], res['rir']
else:
    a, x = s.reconstruct(grid, 8, True, True)
    r = s.gamma(a, x)
    np.savez("../mes.npz", a=a, x=x, rir=r)

plot_reconstr = True
if plot_reconstr:
    m = 0

    r = vec_to_rir(r/np.max(r), m, N)

    plt.plot(np.arange(N)/freq_sampling, r, label='reconstruction')
    rir = vec_to_rir(measurements/np.max(measurements), m, N)
    plt.plot(np.arange(N)/freq_sampling, rir, '-.', label='pyroom')
    plt.legend()
    plt.show()

    ind, dist = compare_arrays(src, x)
    print("distances between real and predicted  sources : \n", dist)

