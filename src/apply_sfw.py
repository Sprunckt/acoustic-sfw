from src.sfw import SFW
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.utils import (create_grid_spherical, c, compare_arrays, save_results,
                                  json_to_dict, correlation, unique_matches, dict_to_json)
from src.visualization import plot_room
from src.simulation.simulate_pra import simulate_rir, load_antenna
import os
import sys
import pandas as pd
import time
tol_recov = 2e-2

plot = False
ideal = True  # if True, use the observation operator to reconstruct the measure, else use a PRA simulation

# directory where the results should be saved
save_path = "experiments/rdb1b_2"
if not os.path.exists(save_path):
    os.mkdir(save_path)

df_path = os.path.join(save_path, "results.csv")
if not os.path.exists(df_path):
    df_res = pd.DataFrame(columns=["exp_id", "nb_found", "nb_recov", "corr_ampl",
                                   "min_dist", "max_dist_global", "max_dist", "mean_dist"])
else:
    df_res = pd.read_csv(df_path)

# path to the parameter json files used for the simulation
paths = ["room_db1/exp_{}_param.json".format(i) for i in range(0, 50)]

# path to a json file containing additional parameters used for all simulations (eg grid spacing)
meta_param_path = os.path.join(save_path, "parameters.json")
meta_param_dict = json_to_dict(meta_param_path)

if __name__ == "__main__":
    tstart = time.time()
    for path in paths:
        print("Applying SFW to " + os.path.split(path)[-1])
        param_dict = json_to_dict(path)
        lam, ideal = meta_param_dict["lambda"], meta_param_dict["ideal"]
        fs = meta_param_dict["fs"]
        ms = meta_param_dict.get("mic_size")
        if ms is not None:  # overwrite the microphone positions
            mic_pos = load_antenna(mic_size=ms)
        else:
            ms = param_dict["mic_size"]
            mic_pos = param_dict["mic_array"]

        use_two_antennas = meta_param_dict.get("use_two_antennas", False)
        if use_two_antennas:
            normal_antenna = load_antenna(mic_size=ms)
            antenna_rad = np.linalg.norm(normal_antenna[0])
            rescaled_antenna, antenna_rad = normal_antenna*ms, antenna_rad*ms  # rescale the antenna to correct size
            half_sep = (antenna_rad + 0.1)*np.array([1., 0, 0])  # half separation between the antennas
            mic_pos = np.concatenate([rescaled_antenna - half_sep, rescaled_antenna + half_sep], axis=0)

        sim_dict = dict()
        sim_dict["fs"] = fs
        sim_dict["room_dim"] = param_dict["room_dim"]
        sim_dict["src_pos"] = param_dict["src_pos"]

        # translate the microphones back to their original positions
        mic_pos += param_dict["origin"]
        sim_dict["mic_array"] = mic_pos
        sim_dict["origin"] = param_dict["origin"]
        sim_dict["max_order"] = meta_param_dict["max_order"]

        # simulate the RIR, the center of the antenna is chosen as the new origin
        measurements, N, src, ampl, mic_pos = simulate_rir(sim_dict)

        if ideal:  # exact theoretical observations
            s = SFW(y=(ampl, src), mic_pos=mic_pos, fs=fs, N=N, lam=lam)
            measurements = s.y.copy()
        else:  # recreation using pyroom acoustics. The parameters are only taken from the room parameters file
            s = SFW(y=measurements, mic_pos=mic_pos, fs=fs, N=N, lam=lam)

        # maximum reachable distance
        max_norm = c * N / param_dict["fs"] + 0.5
        rmax = meta_param_dict.get("rmax", max_norm)

        grid, sph_grid, n_sph = create_grid_spherical(meta_param_dict["rmin"], rmax, meta_param_dict["dr"],
                                                      meta_param_dict["dphi"], meta_param_dict["dphi"])
        # file name without extension
        file_ind = os.path.splitext((os.path.split(path)[-1]))[0]
        curr_save_path = os.path.join(save_path, file_ind)

        if curr_save_path.endswith("_param"):
            dist_path = curr_save_path[:-6] + "_dist.json"
            res_path = curr_save_path[:-6] + "_res.json"
            out_path = curr_save_path[:-6] + ".out"
        else:
            print("invalid path")
            exit(1)

        normalization = meta_param_dict["normalization"]
        min_norm = meta_param_dict["min_norm"]
        spherical_search = meta_param_dict.get("spherical_search", 0)
        stdout = sys.stdout

        sys.stdout = open(out_path, 'w')  # redirecting stdout to capture the prints
        a, x = s.reconstruct(grid=grid, niter=meta_param_dict["max_iter"], min_norm=min_norm, max_norm=max_norm,
                             max_ampl=200,
                             normalization=normalization, spike_merging=False, spherical_search=spherical_search,
                             use_hard_stop=True, verbose=True, rough_search=True, early_stopping=True)

        reconstr_rir = s.gamma(a, x)
        ind, dist = compare_arrays(x, src)
        print("source matching and distances : \n", ind, dist)
        max_dist_global = np.max(dist)

        dist_dic = dict()
        dist_dic["distances"], dist_dic["matching"], dist_dic["reconstr_pos"], dist_dic["image_pos"] = dist, ind, x, src

        dict_to_json(dist_dic, dist_path)
        save_results(res_path, src, ampl, x, a,
                     measurements, reconstr_rir, N, rmax)

        inda, indb, dist = unique_matches(x, src, ampl=a)
        mean_dist = np.mean(dist[dist < tol_recov])
        sorted_ampl_reconstr = a[inda]
        sorted_ampl_exact = ampl[indb]
        correlation_ampl = correlation(sorted_ampl_exact, sorted_ampl_reconstr)
        # number of distinct recovered sources
        nb_recov = (dist < tol_recov).sum()

        res = dict(exp_id=curr_save_path, nb_found=len(a), nb_recov=nb_recov, corr_ampl=correlation_ampl,
                   min_dist=np.min(dist), max_dist_global=max_dist_global, max_dist=np.max(dist), mean_dist=mean_dist)
        df_res = df_res.append(res, ignore_index=True)
        df_res.to_csv(df_path, index=False)

        sys.stdout.close()
        sys.stdout = stdout

        if plot:
            plt.plot(measurements, label="real rir")
            plt.plot(reconstr_rir / np.max(reconstr_rir), label="reconstructed rir")
            plt.legend()
            plt.show()
            plot_room(mic_pos, src, ampl, x)

        if curr_save_path.endswith("_param"):
            curr_save_path = curr_save_path[:-6] + "_res.json"
    print("total execution time : {} s".format(time.time() - tstart))
