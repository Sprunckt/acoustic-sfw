from src.sfw import TimeDomainSFW, FrequencyDomainSFW, EpsilonTimeDomainSFW, compute_time_sample
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from src.simulation.utils import (create_grid_spherical, c, compare_arrays, save_results,
                                  json_to_dict, correlation, unique_matches)
from src.visualization import plot_room
from src.simulation.simulate_pra import simulate_rir, load_antenna
import os
import getopt
import sys
import pandas as pd
import time

tol_recov = 2e-2

plot = False
ideal = True  # if True, use the observation operator to reconstruct the measure, else use a PRA simulation

# default directory where the results should be saved (overwritten by command line arguments)
save_path = "experiments/rdb3_freq/eps_0.01"

# path to the parameter json files used for the simulation
exp_paths = "room_db3"

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['path=', 'exp=', 'exp_path='])

    except getopt.GetoptError:
        print("apply_sfw.py [--path=] [--exp=] [--exp_path]\n"
              "--path= : directory where the results will be saved \n"
              "--exp_path= : directory containing the experiment parameters \n"
              "--exp= : two integers, separated by a comma, that give the range of experience IDs that will be"
              "considered in the folder. Ex : 4,9 will apply SFW to the experiences 4 to 8.")
        sys.exit(1)

    ind_start, ind_end = 0, 50
    for opt, arg in opts:
        if opt == '--path':
            save_path = arg
        elif opt == '--exp':
            ind_start, ind_end = [int(t) for t in arg.split(',')]
        elif opt == '--exp_path':
            exp_paths = arg

    paths = [os.path.join(exp_paths, "exp_{}_param.json".format(i)) for i in range(ind_start, ind_end)]
    exp_ids = [i for i in range(ind_start, ind_end)]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    df_path = os.path.join(save_path, "results.csv")
    if not os.path.exists(df_path):  # create the dataframe if it does not exist
        df_res = pd.DataFrame(columns=["exp_id", "nb_found", "nb_recov", "corr_ampl",
                                       "min_dist", "max_dist_global", "max_dist", "mean_dist"])
        df_res.to_csv(df_path, index=False)

    # path to a json file containing additional parameters used for all simulations (eg grid spacing)
    meta_param_path = os.path.join(save_path, "parameters.json")
    meta_param_dict = json_to_dict(meta_param_path)

    # random numbers generator
    rand_gen = np.random.RandomState()

    # global sfw parameters
    algo_start_cb = meta_param_dict.get("start_cb")
    freeze_step = meta_param_dict.get("freeze_step", 0)
    resliding_step = meta_param_dict.get("resliding_step", 0)
    # choose to cut the rir
    cutoff = meta_param_dict.get("cutoff", -1)

    # global parameters for the set of experiments
    lam, ideal = meta_param_dict["lambda"], meta_param_dict["ideal"]
    fs = meta_param_dict["fs"]  # sampling frequency
    ms = meta_param_dict.get("mic_size")  # array size factor
    max_order = meta_param_dict["max_order"]  # maximum order of reflections used

    # peak signal to noise ration for the decorrelated noise
    psnr = meta_param_dict.get("psnr")

    # psnr for the correlated noise
    psnr_corr = meta_param_dict.get("psnr_corr")

    tstart = time.time()
    for exp_ind, path in enumerate(paths):
        print("Applying SFW to " + os.path.split(path)[-1])
        # parameters specific to the current room experience
        param_dict = json_to_dict(path)

        # setting the seed to get consistant results between runs for a same room id
        rand_gen.seed(exp_ids[exp_ind])

        if ms is not None:  # overwrite the microphone positions
            mic_pos = load_antenna(mic_size=ms)
        else:  # use default values
            ms = param_dict["mic_size"]
            mic_pos = param_dict["mic_array"]

        rot = meta_param_dict.get("rotation_mic")
        if rot is not None:  # overwrite the default rotation
            rot_mat = Rotation.from_euler("xyz", rot, degrees=True).as_matrix()
        else:
            rot_mat = Rotation.from_euler("xyz", param_dict["rotation_mic"], degrees=True).as_matrix()

        use_two_antennas = meta_param_dict.get("use_two_antennas", False)
        if use_two_antennas:
            antenna1 = load_antenna(mic_size=ms) @ rot_mat
            antenna2 = antenna1.copy()
            antenna_rad = np.linalg.norm(antenna1[0])

            half_sep = (antenna_rad + 0.1)*np.array([1., 0, 0])  # half separation between the antennas
            mic_pos = np.concatenate([antenna1 - half_sep, antenna2 + half_sep], axis=0)
        else:  # single antenna
            mic_pos = mic_pos @ rot_mat

        sim_dict = dict()

        room_dim = param_dict["room_dim"]
        src_pos = param_dict["src_pos"]

        # translate the microphones back to their original positions
        mic_pos += param_dict["origin"]
        origin = param_dict["origin"]

        # choose to apply varying absorption rates or a default rate for each wall
        use_abs = meta_param_dict.get("use_absorption")
        if use_abs:
            absorptions = param_dict["absorptions"]
        else:
            absorptions = None

        # simulate the RIR, the center of the antenna is chosen as the new origin
        measurements, N, src, ampl, mic_pos, orders = simulate_rir(fs=fs, room_dim=room_dim, src_pos=src_pos,
                                                                   mic_array=mic_pos, origin=origin,
                                                                   max_order=max_order, absorptions=absorptions,
                                                                   cutoff=cutoff)
        domain = meta_param_dict.get("domain")
        if domain is None:
            print("No domain provided, considering the time domain by default")
            domain = "time"

        if domain in ["frequential", "time"]:
            sfw_init_args = dict(mic_pos=mic_pos, fs=fs, N=N, lam=lam)
        elif domain == "time_epsilon":
            eps = meta_param_dict.get("eps")
            if eps is None:
                print("epsilon must be provided")
                exit(1)
            sfw_init_args = dict(mic_pos=mic_pos, fs=fs, N=N, lam=lam, eps=eps)
        else:
            sfw_init_args = {}
            print("invalid domain type")  # should not be reached
            exit(1)

        if ideal:  # exact theoretical observations
            if domain == "frequential":
                s = FrequencyDomainSFW(y=(ampl, src), **sfw_init_args)
                measurements = s.time_sfw.y.copy()
            elif domain == "time_epsilon":
                s = EpsilonTimeDomainSFW(y=(ampl, src), **sfw_init_args)
                measurements = s.y
            else:
                s = TimeDomainSFW(y=(ampl, src), **sfw_init_args)
                measurements = s.y

        else:  # recreation using pyroom acoustics. The parameters are only taken from the room parameters file
            if domain == "frequential":
                s = FrequencyDomainSFW(y=measurements, mic_pos=mic_pos, fs=fs, N=N, lam=lam)
            else:
                sfw_init_args = dict(mic_pos=mic_pos, fs=fs, N=N, lam=lam)
                s = TimeDomainSFW(y=measurements, mic_pos=mic_pos, fs=fs, N=N, lam=lam)

        # maximum reachable distance
        max_norm = c * N / param_dict["fs"] + 0.5
        rmax = meta_param_dict.get("rmax", max_norm)

        grid, sph_grid, n_sph = create_grid_spherical(meta_param_dict["rmin"], rmax, meta_param_dict["dr"],
                                                      meta_param_dict["dphi"], meta_param_dict["dphi"])

        # file name without extension
        file_ind = os.path.splitext((os.path.split(path)[-1]))[0]
        curr_save_path = os.path.join(save_path, file_ind)

        if curr_save_path.endswith("_param"):
            res_path = curr_save_path[:-6] + "_res.json"
            rir_path = curr_save_path[:-6] + "_rir.json"
            out_path = curr_save_path[:-6] + ".out"
        else:
            print("invalid path")
            exit(1)

        normalization = meta_param_dict["normalization"]
        min_norm = meta_param_dict["min_norm"]
        spherical_search = meta_param_dict.get("spherical_search", 0)
        stdout = sys.stdout

        added_noise = np.zeros_like(s.y)
        if psnr is not None:  # add noise (unsupported for frequential domain)
            std = np.max(np.abs(s.y))*10**(-psnr/20)
            added_noise = added_noise + rand_gen.normal(0, std, s.y.shape)

        if psnr_corr is not None:  # add a sound source emitting blank noise somewhere in the room
            noise_pos = param_dict.get("noise_pos")
            if noise_pos is None:
                print("The position of the noise source must be specified")

            noise_rir, N_noise, _, _, _, _ = simulate_rir(fs=fs, room_dim=room_dim, src_pos=noise_pos,
                                                          mic_array=mic_pos, origin=origin, max_order=max_order,
                                                          absorptions=absorptions, cutoff=cutoff)
            assert N == N_noise, "the length of the RIR at the noise position should be the same as the original RIR"

            # noise source emitting on at least N time samples
            noise = rand_gen.normal(0, 1, size=N)
            M = mic_pos.shape[0]
            noise_rir = noise_rir.reshape(M, N_noise)  # reshape to convolve on every microphone
            for i in range(M):
                noise_rir[i] = np.convolve(noise_rir[i], noise)[:N]

            w = noise_rir.flatten()*10**(-psnr_corr/20)*np.max(np.abs(s.y))
            added_noise = added_noise + w

        if psnr_corr is not None or psnr is not None:  # if there is noise : update the signal
            new_y = s.y + added_noise
            measurements = new_y.copy()  # update the measurements to save the target RIR
            s.__init__(y=new_y, **sfw_init_args)

        # apply a transformation to the room coordinates (done after creating the observations)
        rot_walls = meta_param_dict.get("rotation_walls")  # overwrite the room rotation
        if rot_walls is None:  # using the rotation specific to the room
            rot_walls = param_dict.get("rotation_walls")
        rot_walls = Rotation.from_euler("xyz", rot_walls, degrees=True)
        inv_rot_walls = rot_walls.inv()
        s.mic_pos = mic_pos @ rot_walls.as_matrix()

        sys.stdout = open(out_path, 'w')  # redirecting stdout to capture the prints
        a, x = s.reconstruct(grid=grid, niter=meta_param_dict["max_iter"], min_norm=min_norm, max_norm=max_norm,
                             max_ampl=200, algo_start_cb=algo_start_cb,
                             freeze_step=freeze_step, resliding_step=resliding_step,
                             normalization=normalization, spike_merging=False, spherical_search=spherical_search,
                             use_hard_stop=True, verbose=True, search_method="rough", early_stopping=True, plot=False)

        # reversing the coordinate change
        x = x @ inv_rot_walls.as_matrix()
        s.mic_pos = s.mic_pos @ inv_rot_walls.as_matrix()

        if domain != "frequential":  # extend the RIR to the maximum length
            s.NN = compute_time_sample(s.global_N, s.fs)

        reconstr_rir = s.gamma(a, x)

        ind, dist = compare_arrays(x, src)
        print("source matching and distances : \n", ind)
        print("distances and source orders : \n", dist, orders[ind])
        max_dist_global = np.max(dist)

        dist_dic = dict()
        dist_dic["distances"], dist_dic["matching"] = dist, ind

        if algo_start_cb is not None:
            if algo_start_cb.get("n_cut") is not None:
                dist_dic["cut_ind"] = s.cut_ind
        # save the position of the microphones
        dist_dic["mic_array"] = s.mic_pos

        if domain == "frequential":
            measurements = [np.real(measurements).tolist(), np.imag(measurements).tolist()]
            reconstr_rir = [np.real(reconstr_rir).tolist(), np.imag(reconstr_rir).tolist()]
        save_results(res_path, rir_path, image_pos=src, ampl=ampl, orders=orders, reconstr_pos=x, reconstr_ampl=a,
                     rir=measurements, reconstr_rir=reconstr_rir, N=N, **dist_dic)

        inda, indb, dist = unique_matches(x, src, ampl=a)
        mean_dist = np.mean(dist[dist < tol_recov])
        sorted_ampl_reconstr = a[inda]
        sorted_ampl_exact = ampl[indb]
        correlation_ampl = correlation(sorted_ampl_exact, sorted_ampl_reconstr)
        # number of distinct recovered sources
        nb_recov = (dist < tol_recov).sum()

        res = dict(exp_id=curr_save_path, nb_found=len(a), nb_recov=nb_recov, corr_ampl=correlation_ampl,
                   min_dist=np.min(dist), max_dist_global=max_dist_global, max_dist=np.max(dist), mean_dist=mean_dist)
        # re-read the CSV in case it has been modified by another process
        df_res = pd.read_csv(df_path)
        df_res = df_res.append(res, ignore_index=True)
        df_res.to_csv(df_path, index=False)

        sys.stdout.close()
        sys.stdout = stdout

        if plot:
            plt.plot(measurements, label="real rir")
            plt.plot(reconstr_rir, '--', label="reconstructed rir")
            plt.legend()
            plt.show()
            plot_room(mic_pos, src, ampl, x)

        if curr_save_path.endswith("_param"):
            curr_save_path = curr_save_path[:-6] + "_res.json"
    print("total execution time : {} s".format(time.time() - tstart))
