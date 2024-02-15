"""
Get the relevant metrics for a set of experiments
Can be used as a script in command line.
"""
import numpy as np
from src.simulation.utils import (json_to_dict, correlation, unique_matches, compare_arrays)
from src.tools.geometry_reconstruction import great_circle_distance, radial_distance
import pandas as pd
import os
import sys
import getopt
from scipy.spatial.transform import Rotation


def count_results(directory):
    files = os.listdir(directory)
    k, max_ind = 0, 0
    for f in files:
        fsplit = f.split("_")
        if "res.json" == fsplit[-1]:
            k += 1
            max_ind = np.maximum(max_ind, int(fsplit[-2]))
    return k, max_ind


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


def get_metrics(paths, save_path=None, method="amplitude", delimiter=np.inf, tol=None, n_src=0,
                max_norm=None, unique=True, distance="cartesian", reconstr_ampl_threshold=None, metrics_path=None,
                individual_path=None,   # save distances to sources individually, only works for a single cartesian tol
                return_paths=False, params_path=None, subset=None):

    assert distance in ['spherical', 'cartesian'], "invalid distance type"

    if tol is None:  # setting default tolerance depending on the distance type
        if distance == 'cartesian':
            tol = [5e-2]
        elif distance == 'spherical':
            tol = [[1e-2, 12]]
    else:
        if type(tol) == str:
            if distance == 'cartesian':
                tol = [float(tol)]
            else:
                tol = [[float(a) for a in tol.split(',')]]

    if distance == 'spherical':
        assert len(tol[0]) == 2, 'tol should be a list containing the radial and angular thresholds'
    ntol = len(tol)

    path_list = []
    for path in paths:
        path_list += extract_subdirectories(path)

    global_dfs = [pd.DataFrame(columns=["exp_id", "TP", "FN", "FP", "precision", "recall"]) for _ in range(ntol)]
    for path in path_list:  # loop over every directory
        n_res, max_ind = count_results(path)
        col = ["exp_id", "nb_src", "nb_found", "nb_recov", "recall", "precision", "ampl_corr", "ampl_rel_error",
               "mean_recov_dist", "mean_recov_radial_dist", "mean_recov_angular_dist", "mean_radial_dist",
               "mean_angular_dist"]
        types = [int, int, int, int, float, float, float, float, float, float, float, float, float]

        exp_dfs = [pd.DataFrame(columns=col).astype({col[i]: types[i] for i in range(len(col))})
                   for _ in range(ntol)]  # one DF per tolerance threshold given

        tp, fp, fn, global_error = [0]*ntol, [0]*ntol, [0]*ntol, [0]*ntol
        individual_dist = []
        if subset is None:
            indrange = range(max_ind + 1)
        else:
            indrange = subset

        for i in indrange:  # loop over every experiment
            complete_path = os.path.join(path, "exp_{}_res.json".format(i))

            try:
                res_dict = json_to_dict(complete_path)
            except FileNotFoundError:
                print(complete_path + " not found, continuing \n")
                continue

            real_sources, predicted_sources = res_dict["image_pos"], res_dict["reconstr_pos"]
            real_ampl, reconstr_ampl = res_dict["ampl"], res_dict["reconstr_ampl"]
            orders = res_dict["orders"]

            if params_path is not None:  # apply inverse rotation to the reconstructed sources
                params_dict = json_to_dict(os.path.join(params_path, "exp_{}_param.json".format(i)))
                rotation = params_dict["rotation_mic"]
                rotation = Rotation.from_euler('xyz', rotation, degrees=True)
                predicted_sources = rotation.apply(predicted_sources)

            if reconstr_ampl_threshold is not None:
                ind_considered = reconstr_ampl >= reconstr_ampl_threshold
                reconstr_ampl = reconstr_ampl[ind_considered]
                predicted_sources = predicted_sources[ind_considered]

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
                orders = orders[source_index]
            else:
                print("method option not recognized. \n"
                      "method should be in ['amplitude', 'order', 'distance']")
                exit(1)

            if max_norm is not None:
                norm_ind = np.linalg.norm(real_sources, axis=-1) <= max_norm
                real_sources, real_ampl = real_sources[norm_ind], real_ampl[norm_ind]
            nb_found, nb_needed = len(reconstr_ampl), len(real_ampl)

            if len(real_sources) == 0:
                print("empty array, continuing")
                continue

            if unique:
                # unique matches, looking only at the True positives at minimum distance
                inda, indb, dist = unique_matches(predicted_sources, real_sources, ampl=None)
            else:
                indb, dist = compare_arrays(predicted_sources, real_sources)
                inda = np.arange(len(reconstr_ampl))

            # radial distance
            dist_rad = radial_distance(predicted_sources[inda], real_sources[indb])
            # circle distance in degrees
            dist_circle = great_circle_distance(predicted_sources[inda], real_sources[indb], rad=False)

            for k, sub_tol in enumerate(tol):
                if distance == "cartesian":
                    ind_tol = dist < sub_tol
                elif distance == "spherical":
                    ind_tol = (dist_rad < sub_tol[0]) & (dist_circle < sub_tol[1])
                else:
                    ind_tol = None

                sorted_ampl_reconstr = reconstr_ampl[inda][ind_tol]
                sorted_ampl_exact = real_ampl[indb][ind_tol]

                # correlation and relative error for the amplitudes of the best (max ampl) recovered sources
                correlation_ampl = correlation(sorted_ampl_exact, sorted_ampl_reconstr)
                relative_error = np.mean(np.abs(sorted_ampl_exact - sorted_ampl_reconstr) / sorted_ampl_exact)
                # number of distinct recovered sources
                ctp = ind_tol.sum()
                nb_real = len(real_sources)  # expected number of sources

                tp[k] += ctp  # positions correctly guessed
                fp[k] += nb_found - ctp  # positions incorrectly guessed
                fn[k] += nb_needed - ctp  # sources not found (false negatives)

                entry = dict(exp_id=i, nb_src=nb_real, nb_found=nb_found, nb_recov=ctp, recall=ctp/nb_real,
                             precision=ctp/nb_found, ampl_corr=correlation_ampl, ampl_rel_error=relative_error)
                if ctp == 0:
                    print("no source found exp ", i)
                # mean cartesian distance to real sources for the best recovered sources
                entry["mean_recov_dist"] = dist[ind_tol].mean()
                global_error[k] += dist[ind_tol].sum()

                # mean distance to real sources for all recovered sources
                if not unique:
                    entry["mean_dist"] = dist.mean()
                else:
                    _, all_dist = compare_arrays(predicted_sources, real_sources)
                    entry["mean_dist"] = all_dist.mean()

                # spherical distances
                entry["mean_recov_radial_dist"] = dist_rad[ind_tol].mean()
                entry["mean_recov_angular_dist"] = dist_circle[ind_tol].mean()

                if not unique:
                    entry["mean_radial_dist"] = dist_rad.mean()
                    entry["mean_angular_dist"] = dist_circle.mean()
                else:
                    all_ind, _ = compare_arrays(predicted_sources, real_sources)
                    all_dist_rad = radial_distance(predicted_sources, real_sources[all_ind])
                    all_dist_circle = great_circle_distance(predicted_sources, real_sources[all_ind], rad=False)
                    entry["mean_radial_dist"] = all_dist_rad.mean()
                    entry["mean_angular_dist"] = all_dist_circle.mean()

                exp_dfs[k] = exp_dfs[k].append(entry, ignore_index=True)
            if individual_path is not None:
                individual_dist.append(dist[ind_tol])

        if individual_path is not None:  # save as .csv
            individual_dist = np.concatenate(individual_dist)
            np.savetxt(os.path.join(individual_path, "{}_distances_{}.csv".format(os.path.split(path)[-1], n_src)),
                       individual_dist, delimiter=",")
        if metrics_path is not None:
            for k in range(ntol):
                exp_dfs[k].to_csv(os.path.join(metrics_path, "{}_metrics_{}_{}.csv".format(os.path.split(path)[-1],
                                                                                           n_src, tol[k])),
                                  index=False)

                entry = dict(exp_id=path, TP=tp[k], FN=fn[k], FP=fp[k], mean_recov_dist=global_error[k]/tp[k],
                             precision=tp[k]/(tp[k]+fp[k]), recall=tp[k]/(tp[k]+fn[k]))

                global_dfs[k] = global_dfs[k].append(entry, ignore_index=True)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for k in range(ntol):
            global_dfs[k].to_csv(os.path.join(save_path, 'global_metrics_{}_{}.csv'.format(n_src, tol[k])), index=False)

    if return_paths:
        return global_dfs, path_list
    else:
        return global_dfs


def get_slices(df, col, slice_edges):
    nslices = len(slice_edges) - 1
    slices_indices = [[] for _ in range(nslices)]

    for i in range(nslices):
        slices_indices[i] = df.exp_id.loc[(slice_edges[i] <= df[col]) & (df[col] <= slice_edges[i+1])]

    return slices_indices


def get_metrics_per_slice(df, slices, slice_edges):
    """Return a DataFrame where every metric is computed per slice (each row is a slice)"""
    nslices = len(slices)
    row_id = ["{}-{}".format(slice_edges[i], slice_edges[i+1]) for i in range(nslices)]
    mean_df = pd.DataFrame(columns=df.columns)
    mean_df = mean_df.astype(df.dtypes)
    mean_df.rename(columns={'exp_id': 'range', 'ampl_corr': 'mean_ampl_corr'}, inplace=True)
    if 'mean_recov_dist' in df.columns:
        distance = 'cartesian'
    elif 'mean_recov_radial_dist' in df.columns:
        distance = 'spherical'
    else:
        distance = None
        print("Error : missing mean distance column")
        exit(1)

    for i in range(nslices):
        sub_df = df.loc[df.exp_id.isin(slices[i])]
        nb_src, nb_found, nb_recov = sub_df.nb_src.sum(), sub_df.nb_found.sum(), sub_df.nb_recov.sum()
        precision, recall = nb_recov / nb_found, nb_recov / nb_src
        entry = dict(nb_src=nb_src, nb_found=nb_found, nb_recov=nb_recov)

        # multiply by nb_recov to get slice mean
        entry['mean_recov_dist'] = (sub_df.nb_recov * sub_df.mean_recov_dist).sum() / nb_recov
        entry['mean_dist'] = (sub_df.nb_found * sub_df.mean_dist).sum() / nb_found
        entry['mean_recov_radial_dist'] = (sub_df.nb_recov * sub_df.mean_recov_radial_dist).sum() / nb_recov
        entry['mean_recov_angular_dist'] = (sub_df.nb_recov * sub_df.mean_recov_angular_dist).sum() / nb_recov
        entry['mean_angular_dist'] = (sub_df.nb_found * sub_df.mean_angular_dist).sum() / nb_found
        entry['mean_radial_dist'] = (sub_df.nb_found * sub_df.mean_radial_dist).sum() / nb_found

        entry['mean_ampl_corr'] = sub_df.ampl_corr.mean()
        entry['ampl_rel_error'] = (sub_df.nb_recov * sub_df.ampl_rel_error).sum() / nb_recov
        entry['range'] = row_id[i]
        entry['recall'], entry['precision'] = recall, precision
        mean_df = mean_df.append(entry, ignore_index=True)

    return mean_df


def aggregate_results(path, exp_names, tol, slices, edges, metric='cartesian'):
    ntol, nexp = len(tol), len(exp_names)

    aggregated_df = pd.DataFrame()
    for i in range(ntol):
        for j in range(nexp):
            metrics_path = os.path.join(path, "{}_metrics_{}.csv".format(exp_names[j], str(tol[i])))
            try:
                df_tmp = pd.read_csv(metrics_path)
            except FileNotFoundError:
                print("Tolerance {} for experiment {} not found : path \n {} \n "
                      "not found, stopping".format(tol[i], exp_names[j], metrics_path))
                exit(1)

            sliced_df = get_metrics_per_slice(df_tmp, slices, edges)
            sliced_df["tol"] = float(tol[i]) if metric == 'cartesian' else float(tol[i][1])
            sliced_df["exp_name"] = exp_names[j]
            aggregated_df = aggregated_df.append(sliced_df, ignore_index=True)
    return aggregated_df


def compute_heatmap(aggregated_df, xcol, ycol, ccol):
    xval = pd.unique(aggregated_df[xcol])
    yval = pd.unique(aggregated_df[ycol])
    heat_map = []
    for x in xval:
        res_x = aggregated_df.loc[aggregated_df[xcol] == x][ccol]

        heat_map.append(res_x)

    return xval, yval, np.array(heat_map).T


if __name__ == "__main__":

    try:
        paths_d = sys.argv[1].split(",")
        opts, args = getopt.getopt(sys.argv[2:], 'su', ['tol=', 'save_path=', 'delimiter=', 'method=',
                                                        'n_src=', 'metrics_path=', 'ampl_thresh=', 'distance='])

    except getopt.GetoptError:
        print("extract_metrics.py path [--tol=] [--save_path=] [--n_src=] [--method=] [--metrics_path]"
              "[--delimiter] [--ampl_thresh=] [--distance=] [-su] \n"
              "path : path to a directory or several paths separated by commas \n"
              "--tol= : absolute tolerance \n"
              "--method= : method used to identify which true sources to consider. Available methods : \n"
              "   -amplitude : rank by decreasing amplitudes \n"
              "   -distance : rank by increasing distance \n"
              "   -order : extract all the image sources of a given order. The order can be specified with the n_src"
              "argument. \n"
              "   -orderinf : extract all the image sources for which the order is inferior to the order specified by "
              "the n_src argument.\n"
              "--metrics_path= : custom path to save the csv containing the computed metrics per experiment (by default"
              "saves to the experiment folder \n"
              "--delimiter= : delimiting threshold for the distance method : only consider the true sources that are at"
              "a distance inferior or equal to 'delimiter'. Same behavior for the predicted sources, with a maximal"
              "distance of 'delimiter' + tol.\n"
              "--n_src : number of sources considered for the reconstruction. If not specified, all the theoretical"
              "sources recoverable for the given method and delimiter are considered. If method == order, n_src refers"
              "to the image source order considered.\n"
              "--ampl_thresh= : ignores all reconstructed sources for which the amplitude is below the given "
              "threshold\n"
              "-u : allow to count the True positives multiple times for a given source. If two reconstructed sources"
              "are close to a same image source, both are counted as True positives. By default, one True positive per"
              "image source is possible. This option only makes sense for computing the precision."
              "-s : show the plots after saving")
        sys.exit(1)

    save_path_d, metrics_path_d = ".", None
    method_d, delimiter_d = "distance", np.inf
    tol_d, n_src_d, unique_d = [1e-2], 0, True
    ampl_thresh_d, dist_d = None, 'cartesian'
    for opt, arg in opts:
        if opt == '--save_path':
            save_path_d = arg
        if opt == '--metrics_path':
            metrics_path_d = arg
        elif opt == '--tol':
            tol_d = arg
        elif opt == '--n_src':
            n_src_d = int(arg)
        elif opt == '--method':
            method_d = arg
        elif opt == '--delimiter':
            delimiter_d = float(arg)
        elif opt == '--ampl_thresh':
            ampl_thresh_d = float(arg)
        elif opt == '--distance':
            dist_d = arg
        elif opt == '-s':
            show_d = True
        elif opt == '-u':
            unique_d = False

    get_metrics(paths_d, save_path=save_path_d, method=method_d, delimiter=delimiter_d,
                tol=tol_d, n_src=n_src_d, unique=unique_d, reconstr_ampl_threshold=ampl_thresh_d,
                metrics_path=metrics_path_d, distance=dist_d)
