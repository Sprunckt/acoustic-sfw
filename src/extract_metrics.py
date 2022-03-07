"""
Get the relevant metrics for a set of experiments
Meant to be used as a script in command line.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.utils import json_to_dict, correlation, unique_matches, compare_arrays
import pandas as pd
import os
import sys
import getopt


def count_results(directory):
    files = os.listdir(directory)
    k = 0
    for f in files:
        if "_res.json" in f:
            k += 1
    return k


def extract_subdirectories(pth):
    subdir = []
    if not os.path.exists(os.path.join(pth, 'results.csv')):
        ndir = 0
        for it in os.listdir(pth):
            full_path = os.path.join(pth, it)
            if os.path.isdir(full_path):
                ndir += 1
                subdir += extract_subdirectories(full_path)
        if ndir == 0:
            print("skipping {}, no subdirectory found".format(pth))
    else:
        subdir.append(pth)

    return subdir


def get_metrics(paths, save_path=None, n_plot=0, show=False, method="amplitude", delimiter=np.inf, tol=5e-2, n_src=0,
                unique=True):
    path_list = []
    for path in paths:
        path_list += extract_subdirectories(path)

    df = pd.DataFrame(columns=["exp_id", "TP", "FN", "FP", "precision", "recall"])
    for path in path_list:  # loop over every directory
        n_res = count_results(path)

        exp_df = pd.DataFrame(columns=["exp_id", "nb_found", "nb_recov",
                                       "mean_tp_dist", "mean_recov_dist",
                                       "ampl_corr", "ampl_rel_error"])
        thresholds = np.logspace(np.log10(1e-4), np.log10(0.3), n_plot)
        tp, fp, fn = 0, 0, 0
        global_tp, global_fp = np.zeros_like(thresholds), np.zeros_like(thresholds)
        global_fn = np.zeros_like(thresholds)

        for i in range(n_res):  # loop over every experiment
            complete_path = os.path.join(path, "exp_{}_res.json".format(i))

            try:
                res_dict = json_to_dict(complete_path)
            except FileNotFoundError:
                print(complete_path + " not found, continuing \n")
                continue

            real_sources, predicted_sources = res_dict["image_pos"], res_dict["reconstr_pos"]
            real_ampl, reconstr_ampl = res_dict["ampl"], res_dict["reconstr_ampl"]
            orders = res_dict["orders"]

            # only consider n_src sources, or every source if n_src = 0
            if method == "amplitude":  # sort according to amplitudes
                sort_ind = np.argsort(real_ampl)
                real_sources, real_ampl = real_sources[sort_ind[-n_src:], :], real_ampl[sort_ind[-n_src:]]
                orders = orders[sort_ind[-n_src:]]

            elif method == "distance":  # sort according to the distances
                mic_array = res_dict["mic_array"]
                # distances between sources and microphones
                real_dist = np.sqrt(np.sum((real_sources[np.newaxis, :, :]
                                            - mic_array[:, np.newaxis, :]) ** 2, axis=2))
                # find the sources that are at a distance at most 'delimiter' of at least one microphone
                remaining_src_ind = np.any(real_dist < delimiter, axis=0)
                # compute the minimal distance to the array
                real_dist = np.min(real_dist[:, remaining_src_ind], axis=0)
                sort_ind = np.argsort(real_dist)  # sort according to distance

                if n_src > 0:
                    sort_ind = sort_ind[:n_src]

                real_sources = real_sources[remaining_src_ind][sort_ind, :]
                real_ampl = real_ampl[remaining_src_ind][sort_ind]
                orders = orders[remaining_src_ind][sort_ind]

            elif method == "order":  # extract according to a given source order
                source_index = orders == n_src
                real_sources, real_ampl = real_sources[source_index, :], real_ampl[source_index]
            elif method == "orderinf":  # extract all the sources of an order inferior to the given order
                source_index = orders <= n_src
                real_sources, real_ampl = real_sources[source_index, :], real_ampl[source_index]
            else:
                print("method option not recognized. \n"
                      "method should be in ['amplitude', 'order', 'distance']")
                exit(1)
            nb_found, nb_needed = len(reconstr_ampl), len(real_ampl)

            if unique:
                # unique matches, looking only at the True positives at minimum distance
                inda, indb, dist = unique_matches(predicted_sources, real_sources, ampl=None)
            else:
                indb, dist = compare_arrays(predicted_sources, real_sources)
                inda = np.arange(len(reconstr_ampl))

            ind_tol = dist < tol
            mean_tp_dist = dist[ind_tol].mean()  # mean distance across recovered sources
            sorted_ampl_reconstr = reconstr_ampl[inda][ind_tol]
            sorted_ampl_exact = real_ampl[indb][ind_tol]

            # correlation and relative error for the amplitudes of the best (max ampl) recovered sources
            correlation_ampl = correlation(sorted_ampl_exact, sorted_ampl_reconstr)
            relative_error = np.mean(np.abs(sorted_ampl_exact - sorted_ampl_reconstr) / sorted_ampl_exact)
            # number of distinct recovered sources
            ctp = (dist < tol).sum()

            tp += ctp  # positions correctly guessed
            fp += nb_found - ctp  # positions incorrectly guessed
            fn += nb_needed - ctp  # sources not found (false negatives)

            # current true positives computed at every threshold for plotting
            complete_ctp = (dist < thresholds[:, np.newaxis]).sum(axis=-1)
            global_tp += complete_ctp  # positions correctly guessed
            global_fp += nb_found - complete_ctp  # positions incorrectly guessed
            global_fn += nb_needed - complete_ctp  # sources not found (false negatives)

            # mean distance to real sources for the best recovered sources
            mean_recov_dist = dist[dist < tol].mean()
            entry = dict(exp_id=i, nb_found=nb_found, nb_recov=ctp,
                         mean_tp_dist=mean_tp_dist, mean_recov_dist=mean_recov_dist,
                         ampl_corr=correlation_ampl, ampl_rel_error=relative_error)
            exp_df = exp_df.append(entry, ignore_index=True)

        if n_plot > 0:
            plt.plot(thresholds, global_tp / (global_tp + global_fp), '-.', label='precision')
            plt.plot(thresholds, global_tp / (global_tp + global_fn), '-.', label='recall')
            plt.xlabel('tolerance threshold')
            plt.ylabel('recovery ratio')
            plt.xscale('log')
            plt.legend()
            if save_path is not None:
                plt.savefig(os.path.join(save_path, '{}_recall_curve.pdf'.format(os.path.split(path)[-1])))
            if show:
                plt.show()
            else:
                plt.clf()
        exp_df.to_csv(os.path.join(path, "metrics_{}.csv".format(tol)), index=False)

        entry = dict(exp_id=path, TP=tp, FN=fn, FP=fp,
                     precision=tp/(tp+fp), recall=tp/(tp+fn))

        df = df.append(entry, ignore_index=True)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        df.to_csv(os.path.join(save_path, 'metrics_{}.csv'.format(tol)), index=False)
    return df


if __name__ == "__main__":

    try:
        paths_d = sys.argv[1].split(",")
        opts, args = getopt.getopt(sys.argv[2:], 'su', ['tol=', 'save_path=', 'delimiter=', 'n_plot=', 'method=',
                                                       'n_src='])

    except getopt.GetoptError:
        print("extract_metrics.py path [--tol=] [--save_path=] [--n_src=] [--n_plot=] [--method=] [-is] \n"
              "path : path to a directory or several paths separated by commas \n"
              "--tol= : absolute tolerance \n"
              "--method= : method used to identify which true sources to consider. Available methods : \n"
              "   -amplitude : rank by decreasing amplitudes \n"
              "   -distance : rank by increasing distance \n"
              "   -order : extract all the image sources of a given order. The order can be specified with the n_src"
              "argument. \n"
              "   -orderinf : extract all the image sources for which the order is inferior to the order specified by "
              "the n_src argument.\n"
              "--delimiter= : delimiting threshold for the distance method : only consider the true sources that are at"
              "a distance inferior or equal to 'delimiter'. Same behavior for the predicted sources, with a maximal"
              "distance of 'delimiter' + tol.\n"
              "--n_src : number of sources considered for the reconstruction. If not specified, all the theoretical"
              "sources recoverable for the given method and delimiter are considered. If method == order, n_src refers"
              "to the image source order considered.\n"
              "--n_plot= : number of data points for a recall/precision curve in function "
              "of the tolerance threshold \n"
              "-u : allow to count the True positives multiple times for a given source. If two reconstructed sources"
              "are close to a same image source, both are counted as True positives. By default, one True positive per"
              "image source is possible. This option only makes sense for computing the precision."
              "-s : show the plots after saving")
        sys.exit(1)

    save_path_d = "."
    n_plot_d, show_d, method_d, delimiter_d = 0, False, "distance", np.inf
    tol_d, n_src_d, unique_d = 1e-2, 0, True
    for opt, arg in opts:
        if opt == '--save_path':
            save_path_d = arg
        elif opt == '--tol':
            tol_d = float(arg)
        elif opt == '--n_src':
            n_src_d = int(arg)
        elif opt == '--n_plot':
            n_plot_d = int(arg)
        elif opt == '--method':
            method_d = arg
        elif opt == '--delimiter':
            delimiter_d = float(arg)
        elif opt == '-s':
            show_d = True
        elif opt == '-u':
            unique_d = False

    get_metrics(paths_d, save_path_d, n_plot_d, show_d, method_d, delimiter_d, tol_d, n_src_d, unique=unique_d)
