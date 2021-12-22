"""
Creating the parameter files for multiple random experiments and launching simulations
"""

import numpy as np
import json
import pandas as pd
import os
import sys
from src.simulation.utils import (create_grid_spherical,
                                  compare_arrays, c, save_results, dict_to_json)
from src.simulation.simulate_pra import load_antenna, simulate_rir
from src.sfw import TimeDomainSFW

tol_recov = 2e-2

# if set to True : apply SFW at each room. ELse : only create the parameter files after running PRA
reconstruct = False

directory = "room_db2"
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
else:  # loading the global parameters for the experiments
    fd = open(conf_path, 'r')
    param_dict = json.load(fd)

df_path = os.path.join(directory, "results.csv")
if not os.path.exists(df_path):
    df_res = pd.DataFrame(columns=["exp_id", "min_dist", "mean_dist", "max_dist", "nb_recov"])
else:
    df_res = pd.read_csv(df_path)

new_exp = 50
for i in range(new_exp):
    # defining the room dimensions
    room_dim = [np.random.uniform(*param_dict["xlim"]),
                np.random.uniform(*param_dict["ylim"]),
                np.random.uniform(*param_dict["zlim"])]
    max_order = 1
    fs = param_dict["fs"]  # sampling frequency in Hz

    xroom, yroom, zroom = room_dim
    # placing the source randomly
    z_src = param_dict.get("z_src")

    if z_src is None:  # if the parameter is not set, choose a random altitude
        z_src = np.random.uniform(param_dict["mic_wall_sep"], zroom - param_dict["mic_wall_sep"])

    src_pos = np.array([np.random.uniform(param_dict["mic_wall_sep"], xroom - param_dict["mic_wall_sep"]),
                        np.random.uniform(param_dict["mic_wall_sep"], yroom - param_dict["mic_wall_sep"]),
                        z_src])

    mic_pos = src_pos.copy()

    # placing the microphone antenna
    z_mic = param_dict.get("z_mic")
    fixed_z = z_mic is not None

    k = 0
    while np.linalg.norm(src_pos - mic_pos) < param_dict["mic_src_sep"] and k < 100:
        z = z_mic if fixed_z else np.random.uniform(param_dict["mic_wall_sep"], zroom - param_dict["mic_wall_sep"])

        mic_pos = np.array([np.random.uniform(param_dict["mic_wall_sep"], xroom - param_dict["mic_wall_sep"]),
                            np.random.uniform(param_dict["mic_wall_sep"], yroom - param_dict["mic_wall_sep"]),
                            z])

        k += 1
    if k == 100:
        print("Error : failed to place the antenna")
        exit(1)

    # creating the microphone array
    mic_array = load_antenna('data/eigenmike32_cartesian.csv',
                             mic_size=param_dict["mic_size"]) + np.reshape(mic_pos, [1, 3])
    M = len(mic_array)

    sim_param = dict(max_order=1, mic_array=mic_array, src_pos=src_pos, room_dim=room_dim,
                     fs=fs, origin=mic_pos)

    # simulate the RIR, the center of the antenna is choosed as the new origin
    measurements, N, src, ampl, mic_array = simulate_rir(sim_param)

    sim_param["image_pos"] = src
    sim_param["ampl"] = ampl
    sim_param["N"] = N

    exp_id = get_id()
    str_id = "exp_" + str(exp_id)
    # saving the parameters used to run the simulation and the resulting amplitudes and source positions
    sim_param["mic_size"] = param_dict["mic_size"]
    dict_to_json(sim_param, os.path.join(directory, str_id + "_param.json"))

    if reconstruct:
        # compute the maximal radius that can be reached depending on tmax
        rmax = c * N / fs + 0.5
        # create a spherical grid
        grid, sph_grid, n_sph = create_grid_spherical(param_dict["rmin"], rmax, param_dict["dr"],
                                                      param_dict["dphi"], param_dict["dphi"])
        n_grid = grid.shape[0]

        measurements = measurements/np.max(measurements)
        # applying the algorithm
        s = TimeDomainSFW(measurements, mic_pos=mic_array, fs=fs, N=N, lam=1e-2)

        stdout = sys.stdout
        sys.stdout = open(os.path.join(directory, str_id + ".out"), 'w')  # redirecting stdout to capture the prints
        a, x = s.reconstruct(grid=grid, niter=7, use_hard_stop=True, verbose=True)
        sys.stdout.close()
        sys.stdout = stdout

        r = s.gamma(a, x)

        # save the results (original sources/new sources, rirs)
        save_results(os.path.join(directory, str_id + "_res.json"),
                     src, ampl, x, a, measurements, r, N, rmax)

        # compute the distances between real and reconstructed positions and save them in the global csv
        ind, dist = compare_arrays(x, src)
        # number of distinct recovered sources
        nb_recov = np.minimum((dist < tol_recov).sum(), len(np.unique(ind)))
        # mean error for recovered sources
        mean_dist = np.mean(dist[dist < tol_recov])

        res = dict(exp_id=exp_id, nb_found=len(a), nb_recov=nb_recov,
                   min_dist=np.min(dist), max_dist=np.max(dist), mean_dist=mean_dist)
        df_res = df_res.append(res, ignore_index=True)
        df_res.to_csv(df_path, index=False)

    print("Exp {} done".format(exp_id))
    update_op(exp_id + 1)
