from src.sfw import (TimeDomainSFW, FrequencyDomainSFW, EpsilonTimeDomainSFW, compute_time_sample, TimeDomainSFWNorm1,
                     TimeDomainSFWNorm2, FrequencyDomainSFWNorm1, DeconvolutionSFW)
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from src.simulation.utils import (create_grid_spherical, c, compare_arrays, save_results,
                                  json_to_dict, correlation, unique_matches, create_grid_spherical_multiple)
from src.tools.visualization import plot_room
from src.simulation.simulate_pra import simulate_rir, load_antenna
import os
import getopt
import sys
import pandas as pd
import time

tol_recov = 2e-2  # tolerance threshold for quick evaluation

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
    slide_opt = meta_param_dict.get("slide_opt", None)

    # choose to cut the rir
    cutoff = meta_param_dict.get("cutoff", -1)

    # global parameters for the set of experiments
    lam, ideal = meta_param_dict["lambda"], meta_param_dict["ideal"]
    fs = meta_param_dict["fs"]  # sampling frequency
    fc = meta_param_dict.get("fc", fs)  # cutoff frequency, defaults to fs

    ms = meta_param_dict.get("mic_size")  # array size factor
    mic_path = meta_param_dict.get("mic_path")  # path to the microphone positions
    if mic_path is None:
        mic_path = 'data/eigenmike32_cartesian.csv'

    max_order = meta_param_dict["max_order"]  # maximum order of reflections used

    # amplitude threshold for deleting the spikes at each iteration
    deletion_tol = meta_param_dict.get("deletion_tol", 0.05)

    # amplitude threshold for deleting the spikes at the end
    end_tol = meta_param_dict.get("end_tol", 0.05)

    # grid search method
    grid_method = meta_param_dict.get("grid_search", "naive")

    # optimization parameters
    opt_param = meta_param_dict.get("opt_param", None)

    # number of microphones considered during the spherical search
    nmic = meta_param_dict.get("nmic", None)

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
            mic_pos = load_antenna(mic_size=ms, file_path=mic_path)
        else:  # use default values
            mic_pos = load_antenna(mic_size=1., file_path=mic_path)

        rot = meta_param_dict.get("rotation_mic")
        if rot is not None:  # overwrite the default rotation
            print("overwriting default microphone rotation")
            rot_mic = Rotation.from_euler("xyz", rot, degrees=True)
        else:
            mic_rot = param_dict.get("rotation_mic")
            if mic_rot is None:
                print("no rotation applied to the antenna")
                mic_rot = [0, 0, 0.]
            rot_mic = Rotation.from_euler("xyz", mic_rot, degrees=True)

        # rotate the microphones
        original_mic_pos = mic_pos.copy()
        mic_pos = rot_mic.apply(mic_pos)

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
        (measurements, N, src, ampl, mic_pos, orders,
         full_src, full_ampl) = simulate_rir(fs=fs, room_dim=room_dim, src_pos=src_pos, mic_array=mic_pos,
                                             origin=origin, max_order=max_order, absorptions=absorptions,
                                             cutoff=cutoff, return_full=True)

        domain = meta_param_dict.get("domain")
        if domain is None:
            print("No domain provided, considering the time domain by default")
            domain = "time"
        elif domain == "frequency":
            domain = "frequential"

        sf_types = {"time": [TimeDomainSFW, TimeDomainSFWNorm1, TimeDomainSFWNorm2],
                    "deconvolution": [DeconvolutionSFW], "frequential": [FrequencyDomainSFW, FrequencyDomainSFWNorm1],
                    "time_epsilon": [EpsilonTimeDomainSFW]}

        sfw_init_args = dict(mic_pos=mic_pos, fs=fs, fc=fc, N=N, lam=lam,
                             deletion_tol=deletion_tol, end_tol=end_tol)

        if domain == "frequential":
            sfw_init_args["freq_range"] = meta_param_dict.get("freq_range")
        elif domain == "time_epsilon":
            eps = meta_param_dict.get("eps")
            if eps is None:
                print("epsilon must be provided")
                exit(1)
            sfw_init_args["eps"] = eps
        elif domain == "time":
            pass
        elif domain == "deconvolution":
            sfw_init_args["freq_range"] = meta_param_dict.get("freq_range")
            sfw_init_args["source_pos"] = src[orders == 0]
        else:
            sfw_init_args = {}
            print("invalid domain type")  # should not be reached
            exit(1)

        normalization = meta_param_dict["normalization"]  # normalization used (0 for default)

        if ideal:  # exact theoretical observations
            s = sf_types[domain][normalization](y=(full_ampl, full_src), **sfw_init_args)
            if domain == "frequential":
                measurements = s.time_sfw.y
            elif domain == "deconvolution":
                measurements = s.freq_sfw.time_sfw.y
            else:
                measurements = s.y
        else:  # recreation using pyroom acoustics. The parameters are only taken from the room parameters file
            s = sf_types[domain][normalization](y=measurements, **sfw_init_args)

        # maximum reachable distance
        max_norm = c * N / meta_param_dict["fs"] + 0.5
        rmax = meta_param_dict.get("rmax", max_norm)
        multiple_spheres = meta_param_dict.get("multiple_spheres", 0)
        scale_dphi = meta_param_dict.get("scale_dphi", False)
        dr = meta_param_dict.get("dr", 0.1)
        if scale_dphi:
            grid = meta_param_dict["dphi"]
        elif multiple_spheres > 0:
            grid, sph_grid = create_grid_spherical_multiple(rmin=meta_param_dict["rmin"],
                                                            nspheres=multiple_spheres, dr=dr,
                                                            dphi=meta_param_dict["dphi"],
                                                            dtheta=meta_param_dict["dphi"])

        else:
            grid, sph_grid, n_sph = create_grid_spherical(rmin=meta_param_dict["rmin"], rmax=rmax, dr=dr,
                                                          dphi=meta_param_dict["dphi"], dtheta=meta_param_dict["dphi"])

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

        min_norm = meta_param_dict["min_norm"]
        spherical_search = meta_param_dict.get("spherical_search", 0)
        stdout = sys.stdout

        added_noise = np.zeros_like(measurements)
        if psnr is not None:  # add noise (unsupported for frequential domain)
            std = np.max(np.abs(measurements))*10**(-psnr/20)
            added_noise = added_noise + rand_gen.normal(0, std, measurements.shape)

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
            measurements = measurements + added_noise  # update the measurements to save the target RIR
            s.__init__(y=measurements, **sfw_init_args)

        # reverse rotation on microphones before reconstruction
        s.update_mic_pos(original_mic_pos)
        if domain == "deconvolution":
            s.source_pos = rot_mic.apply(s.source_pos, inverse=True)

        save_var = meta_param_dict.get("save_path")
        if save_var is not None:
            save_freq = meta_param_dict.get("save_freq")
            if save_freq is None:
                save_freq = 8
            save_var = (save_freq, save_var + "{}.csv".format(exp_ind))

        sys.stdout = open(out_path, 'w')  # redirecting stdout to capture the prints
        a, x = s.reconstruct(grid=grid, niter=meta_param_dict["max_iter"], min_norm=min_norm, max_norm=max_norm,
                             max_ampl=200, algo_start_cb=algo_start_cb,
                             slide_opt=slide_opt, spike_merging=False,
                             spherical_search=spherical_search, use_hard_stop=True, verbose=True,
                             search_method=grid_method, opt_param=opt_param, nmic=nmic,
                             early_stopping=True, plot=False, saving_param=save_var)

        if meta_param_dict.get("reverse_coordinates", False):  # reversing the coordinate change
            x = rot_mic.apply(x)
            s.update_mic_pos(rot_mic.apply(s.mic_pos))

        # extracting the results
        if domain == "frequential":  # extend the RIR to the maximum length
            s.time_sfw.NN = compute_time_sample(s.time_sfw.global_N, s.fs)
            reconstr_rir = s.time_sfw.gamma(a, x)
        elif domain == "deconvolution":
            s.freq_sfw.time_sfw.NN = compute_time_sample(s.freq_sfw.time_sfw.global_N, s.fs)
            reconstr_rir = s.freq_sfw.time_sfw.gamma(a, x)
        else:
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
                dist_dic["cut_ind"] = s.get_cut_ind()

        # save the position of the microphones
        dist_dic["mic_array"] = s.mic_pos

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
