
"""
Launching multiple random experiments.
"""

import numpy as np
import json
import pandas as pd
import os
import sys
import pyroomacoustics as pra
from utils import (multichannel_rir_to_vec, vec_to_rir, create_grid_spherical,
                   compare_arrays, c)
from sfw import SFW
tol_recov = 2e-2

directory = "sfw_exp1"
if not os.path.exists(directory):
    os.mkdir(directory)

op_path = os.path.join(directory, "operator.json")
if not os.path.exists(op_path):
    fd = open(op_path, "w")
    json.dump(dict(exp_id=0), fd)
    fd.close()


def update_op(idd):
    fd1 = open(op_path, "r")
    curr_dict = json.load(fd1)
    fd1.close()

    fd2 = open(op_path, "w")
    curr_dict["exp_id"] = idd
    json.dump(curr_dict, fd2)
    fd2.close()


def get_id():
    fd1 = open(op_path, "r")
    curr_dict = json.load(fd1)
    idd = curr_dict["exp_id"]
    fd1.close()
    return idd


conf_path = os.path.join(directory, "parameters.json")
if not os.path.exists(conf_path):
    print("Error : the conf file must be created first.")
    param_dict = None
    exit(1)
else:
    fd = open(conf_path, 'r')
    param_dict = json.load(fd)

df_path = os.path.join(directory, "results.csv")
if not os.path.exists(df_path):
    df_res = pd.DataFrame(columns=["exp_id", "min_dist", "mean_dist", "max_dist", "nb_recov"])
else:
    df_res = pd.read_csv(df_path)
    df_res.to_csv(os.path.split(df_path)[0] + "_copy.csv")  # making a copy

new_exp = 20
for i in range(new_exp):
    # defining the room parameters
    room_dim = [np.random.uniform(*param_dict["xlim"]),
                np.random.uniform(*param_dict["ylim"]),
                np.random.uniform(*param_dict["zlim"])]
    max_order = 1
    freq_sampling = param_dict["fs"]  # Hz
    all_flat_materials = {
        "east": pra.Material(0.1),
        "west": pra.Material(0.1),
        "north": pra.Material(0.1),
        "south": pra.Material(0.1),
        "ceiling": pra.Material(0.1),
        "floor": pra.Material(0.1),
    }
    xroom, yroom, zroom = room_dim
    src_pos = np.array([np.random.uniform(param_dict["mic_wall_sep"], xroom - param_dict["mic_wall_sep"]),
                        np.random.uniform(param_dict["mic_wall_sep"], yroom - param_dict["mic_wall_sep"]),
                        np.random.uniform(param_dict["mic_wall_sep"], zroom - param_dict["mic_wall_sep"])])

    mic_pos = src_pos.copy()
    while np.linalg.norm(src_pos - mic_pos) < param_dict["mic_src_sep"]:
        mic_pos = np.array([np.random.uniform(param_dict["mic_wall_sep"], xroom - param_dict["mic_wall_sep"]),
                            np.random.uniform(param_dict["mic_wall_sep"], yroom - param_dict["mic_wall_sep"]),
                            np.random.uniform(param_dict["mic_wall_sep"], zroom - param_dict["mic_wall_sep"])])

    # create the Room
    room = pra.ShoeBox(room_dim, fs=freq_sampling,
                       materials=all_flat_materials, max_order=max_order)

    # add the source
    room.add_source(src_pos)

    # Load the eigenmike32 spherical microphone array
    # Source: https://www.locata.lms.tf.fau.de/files/2020/01/Documentation_LOCATA_final_release_V1.pdf
    mic_array = np.transpose(np.genfromtxt('data/eigenmike32_cartesian.csv', delimiter=', '))

    # Translate to desired position and scale
    mic_array = param_dict["mic_size"] * mic_array + np.reshape(mic_pos, [3, 1])

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

    d = 3  # dimension of the problem
    mic_array = mic_array.T  # positions, shape (M, d)

    # translate the microphones so the center of the antenna is the origin
    mic_array = mic_array - np.reshape(mic_pos, [1, 3])

    J = N * M

    # compute the maximal radius that can be reached depending on tmax
    rmax = c * N / freq_sampling + 0.5
    # create a spherical grid
    grid, sph_grid, n_sph = create_grid_spherical(1, rmax, param_dict["dr"],
                                                  param_dict["dphi"], param_dict["dphi"])
    n_grid = grid.shape[0]

    exp_id = get_id()
    str_id = "exp_" + str(exp_id)

    # saving original source and microphone positions (before relocation) as well as room dimensions
    exp_data = dict(src_pos=src_pos.tolist(), room_dim=room_dim, mic_pos=mic_pos.tolist(), rmax=rmax, N=N)
    fd = open(os.path.join(directory, str_id + "_param.json"), 'w')
    json.dump(exp_data, fd)
    fd.close()

    measurements = measurements/np.max(measurements)
    # applying the algorithm
    s = SFW(measurements, mic_pos=mic_array, fs=freq_sampling, N=N, lam=1e-2)
    stdout = sys.stdout
    sys.stdout = open(os.path.join(directory, str_id + ".out"), 'w')
    a, x = s.reconstruct(grid, 7, True, True)
    sys.stdout.close()
    sys.stdout = stdout

    r = s.gamma(a, x)

    # save the results (original sources/new sources, rirs)
    exp_res = dict(image_pos=src.tolist(), ampl=ampl.tolist(), reconstr_pos=x.tolist(), reconstr_ampl=a.tolist(),
                   rir=measurements.tolist(), reconstr_rir=r.tolist(), N=N)
    fd = open(os.path.join(directory, str_id + "_res.json"), 'w')
    json.dump(exp_res, fd)
    fd.close()

    # compute the distances between real and reconstructed positions and save them in the global csv
    ind, dist = compare_arrays(src, x)
    nb_recov = (dist < tol_recov).sum()

    res = dict(exp_id=exp_id, nb_found=len(a), nb_recov=nb_recov,
               min_dist=np.min(dist), max_dist=np.max(dist), mean_dist=np.mean(dist))
    df_res = df_res.append(res, ignore_index=True)
    df_res.to_csv(df_path)

    fd = open(conf_path, 'w')
    json.dump(param_dict, fd)
    fd.close()
    print("Exp {} done".format(exp_id))
    update_op(exp_id + 1)
