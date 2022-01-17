"""
Get the relevant metrics for a set of experiments
Meant to be used as a script in command line.
"""
import numpy as np
from src.simulation.utils import json_to_dict, correlation, compare_arrays, unique_matches
import pandas as pd
import os
import sys
import getopt


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


if __name__ == "__main__":

    try:
        paths = sys.argv[1].split(",")
        opts, args = getopt.getopt(sys.argv[2:], 'p', ['tol=', 'save_path='])

    except getopt.GetoptError:
        print("extract_metrics.py path [--tol=] \n"
              "path : path to a directory or several paths separated by commas \n"
              "--tol= : absolute tolerance \n")
        sys.exit(1)

    save_path = "."
    plot = False
    tol = 1e-2
    for opt, arg in opts:
        if opt == '--save_path':
            save_path = arg
        elif opt == '--tol':
            tol = float(arg)

    path_list = []
    for path in paths:
        path_list += extract_subdirectories(path)

    df = pd.DataFrame(columns=["exp_id", "TP", "FN", "FP", "precision", "recall"])
    for path in path_list:  # loop over every directory
        df_res = pd.read_csv(os.path.join(path, "results.csv"))
        n_res = len(df_res)

        exp_df = pd.DataFrame(columns=["exp_id", "nb_found", "nb_recov",
                                       "mean_tp_dist", "mean_recov_dist",  "break_dist",
                                       "ampl_corr", "ampl_rel_error"])
        tp, fp, fn = 0, 0, 0
        for i in range(n_res):  # loop over every experiment
            complete_path = os.path.join(path, "exp_{}_res.json".format(i))
            res_dict = json_to_dict(complete_path)
            real_sources, predicted_sources = res_dict["image_pos"], res_dict["reconstr_pos"]
            real_ampl, reconstr_ampl = res_dict["ampl"], res_dict["reconstr_ampl"]

            nb_found, nb_needed = len(reconstr_ampl), len(real_ampl)

            # unique matches, looking only at the True positives with maximum amplitude
            inda, indb, dist = unique_matches(predicted_sources, real_sources, ampl=reconstr_ampl)
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

            # mean distance to real sources for the best recovered sources
            mean_recov_dist = dist[dist < tol].mean()

            if ctp == nb_needed:  # all sources are retrieved
                break_dist = np.max(dist)
            else:  # look at the minimal distance for the false negatives
                break_dist = np.min(dist[dist >= tol])
            entry = dict(exp_id=i, nb_found=nb_found, nb_recov=ctp,
                         mean_tp_dist=mean_tp_dist, mean_recov_dist=mean_recov_dist,  break_dist=break_dist,
                         ampl_corr=correlation_ampl, ampl_rel_error=relative_error)
            exp_df = exp_df.append(entry, ignore_index=True)

        exp_df.to_csv(os.path.join(path, "metrics_{}.csv".format(tol)))

        entry = dict(exp_id=path, TP=tp, FN=fn, FP=fp,
                     precision=tp/(tp+fp), recall=tp/(tp+fn))

        df = df.append(entry, ignore_index=True)

    df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)
