from src.sfw import SFW
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.utils import create_grid_spherical, c, compare_arrays, save_results, json_to_dict, plot_room
from src.simulation.simulate_pra import simulate_rir
import os
import sys
import pandas as pd
tol_recov = 2e-2

plot = False
ideal = True  # if True, use the observation operator to reconstruct the measure, else use a PRA simulation

# directory where the results should be saved
save_path = "experiments/rdb1_1"
if not os.path.exists(save_path):
    os.mkdir(save_path)

df_path = os.path.join(save_path, "results.csv")
if not os.path.exists(df_path):
    df_res = pd.DataFrame(columns=["exp_id", "recov_dist", "max_dist", "nb_recov"])
else:
    df_res = pd.read_csv(df_path)

# path to the parameter json files used for the simulation
paths = ["room_db1/exp_{}_param.json".format(i) for i in range(22)]

# path to a json file containing additional parameters used for all simulations (eg grid spacing)
meta_param_path = os.path.join(save_path, "parameters.json")
meta_param_dict = json_to_dict(meta_param_path)

for path in paths:
    print("Applying SFW to " + os.path.split(path)[-1])
    param_dict = json_to_dict(path)
    lam = 1e-3
    if ideal:
        N, src, ampl = param_dict["N"], param_dict["image_pos"], param_dict["ampl"]
        s = SFW(y=(ampl, src), mic_pos=param_dict["mic_array"], fs=param_dict["fs"], N=N, lam=1e-3)
        measurements = s.y.copy()
    else:
        # translating the microphones back to their original positions
        param_dict["mic_array"] += param_dict["origin"]
        # simulate the RIR, the center of the antenna is choosed as the new origin
        measurements, N, src, ampl, mic_array = simulate_rir(param_dict)
        s = SFW(y=measurements, mic_pos=param_dict["mic_array"], fs=param_dict["fs"], N=N, lam=1e-3)

    rmax = c * N / param_dict["fs"] + 0.5

    grid, sph_grid, n_sph = create_grid_spherical(meta_param_dict["rmin"], rmax, meta_param_dict["dr"],
                                                  meta_param_dict["dphi"], meta_param_dict["dphi"])

    # file name without extension
    file_ind = os.path.splitext((os.path.split(path)[-1]))[0]
    curr_save_path = os.path.join(save_path, file_ind)

    if curr_save_path.endswith("_param"):
        res_path = curr_save_path[:-6] + "_res.json"
        out_path = curr_save_path[:-6] + ".out"
    else:
        print("invalid path")
        exit(1)

    stdout = sys.stdout
    sys.stdout = open(out_path, 'w')  # redirecting stdout to capture the prints
    a, x = s.reconstruct(grid=grid, niter=8, min_norm=1, max_norm=rmax, max_ampl=200, spike_merging=False,
                         use_hard_stop=True, verbose=True, rough_search=True)

    reconstr_rir = s.gamma(a, x)
    ind, dist = compare_arrays(x, src)
    print("source matching and distances : \n", ind, dist)
    sys.stdout.close()
    sys.stdout = stdout

    if plot:
        plt.plot(measurements, label="real rir")
        plt.plot(reconstr_rir / np.max(reconstr_rir), label="reconstructed rir")
        plt.legend()
        plt.show()
        plot_room(mic_array, src, ampl, x)

    if curr_save_path.endswith("_param"):
        curr_save_path = curr_save_path[:-6] + "_res.json"

    save_results(res_path, src, ampl, x, a,
                 measurements, reconstr_rir, N, rmax)

    # number of distinct recovered sources
    nb_recov = np.minimum((dist < tol_recov).sum(), len(np.unique(ind)))
    # mean error for recovered sources
    mean_dist = np.mean(dist[dist < tol_recov])

    res = dict(exp_id=curr_save_path, nb_found=len(a), nb_recov=nb_recov,
               min_dist=np.min(dist), max_dist=np.max(dist), mean_dist=mean_dist)
    df_res = df_res.append(res, ignore_index=True)
    df_res.to_csv(df_path, index=False)
