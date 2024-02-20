import numpy as np
from scipy.spatial.transform import Rotation
from src.simulation.utils import (create_grid_spherical, c, json_to_dict, create_grid_spherical_multiple)
from src.tools.certificates import gamma_op, etav, etav_der2, pV
from src.simulation.simulate_pra import load_antenna
import multiprocessing as mp
from functools import partial
import os
import getopt
import sys
import time

# default directory where the results should be saved (overwritten by command line arguments)
save_path = "experiments/certif/16_3"

# path to the parameter json files used for the simulation
exp_paths = "roomdb_art_recons_val"

# path to reconstruction results
res_path = "experiments/art_recons/add_sim/16_3"


if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['path=', 'exp=', 'exp_path=', 'res_path='])

    except getopt.GetoptError:
        print("evaluate_certificates.py [--path=] [--exp=] [--exp_path]\n"
              "--path= : directory where the results will be saved \n"
              "--exp_path= : directory containing the experiment parameters \n"
              "--res_path= : directory containing the reconstruction results \n"
              "--exp= : two integers, separated by a comma, that give the range of experience IDs that will be"
              "considered in the folder. Ex : 4,9 will apply SFW to the experiences 4 to 8.")
        sys.exit(1)

    ind_start, ind_end = 45, 46
    for opt, arg in opts:
        if opt == '--path':
            save_path = arg
        elif opt == '--exp':
            ind_start, ind_end = [int(t) for t in arg.split(',')]
        elif opt == '--exp_path':
            exp_paths = arg
        elif opt == '--res_path':
            res_path = arg

    paths = [os.path.join(exp_paths, "exp_{}_param.json".format(i)) for i in range(ind_start, ind_end)]
    res_paths = [os.path.join(res_path, "exp_{}_res.json".format(i)) for i in range(ind_start, ind_end)]
    exp_ids = [i for i in range(ind_start, ind_end)]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # certificate evaluation parameters
    certif_param_path = os.path.join(save_path, "parameters.json")
    certif_param_dict = json_to_dict(certif_param_path)

    # path to a json file containing additional parameters used for all simulations (eg grid spacing)
    meta_param_path = os.path.join(res_path, "parameters.json")
    meta_param_dict = json_to_dict(meta_param_path)

    # global parameters for the set of experiments
    lam = meta_param_dict["lambda"]
    fs = meta_param_dict["fs"]  # sampling frequency

    ms = meta_param_dict.get("mic_size")  # array size factor
    mic_path = meta_param_dict.get("mic_path", "data/eigenmike32_cartesian.csv")  # path to the microphone positions

    # type of grid used for evaluating |etav| : "full" or "centered"
    grid_type = certif_param_dict.get("grid_type", "full")

    # grid spacing, either (dr, dtheta) for "centered"  or (dx, dr, dtheta, rmin, rmax) if grid_type is "full"
    grid_spacing = certif_param_dict.get("grid_spacing", [0.05, 0.01, 2.])

    # extent overflow for the grid
    overflow = certif_param_dict.get("overflow", 2*c/fs)
    print("grid type : {}, grid spacing : {}, overflow : {}".format(grid_type, grid_spacing, overflow))
    tstart = time.time()
    for exp_ind, path in enumerate(paths):
        print("Evaluating certificates in exp " + os.path.split(path)[-1])
        # parameters specific to the current room experience
        param_dict = json_to_dict(path)
        res_dict = json_to_dict(res_paths[exp_ind])

        # exact locations of the sources
        image_pos = res_dict["image_pos"]
        # apply rotation to the sources to have the same reference frame as the microphones
        rot = param_dict.get("rotation_walls")
        if rot is not None:
            rotation = Rotation.from_euler("xyz", rot, degrees=True)
            image_pos = image_pos @ rotation.as_matrix()

        N = res_dict["N"]
        dict_path = os.path.join(save_path, "certif_{}.json".format(exp_ids[exp_ind]))
        dict_res = dict(exp_id=exp_ids[exp_ind])

        if ms is not None:  # overwrite the microphone positions
            mic_pos = load_antenna(mic_size=ms, file_path=mic_path)
        else:  # use default values
            mic_pos = load_antenna(mic_size=1., file_path=mic_path)

        full_grid = None
        # create the evaluation grid for etav
        if grid_type == "centered":  # create spherical grids centered around microphones
            # the radius of each sphere depends on the times of arrival of the sources
            r = np.sqrt(np.sum((image_pos[np.newaxis, :] - mic_pos[:, np.newaxis])**2, axis=2))  # shape (M, K)
            nspheres = int(np.ceil(overflow/grid_spacing[0]))
            sph_grid, _ = create_grid_spherical_multiple(rmin=1., nspheres=nspheres, dr=grid_spacing[0],
                                                         dphi=grid_spacing[1], dtheta=grid_spacing[1])
            full_grid = []
            for i, mic in enumerate(mic_pos):
                r_curr = r[i]
                r_curr = r_curr[r_curr > 0.75]  # delete points that are too close to the microphone
                full_grid.append(mic[np.newaxis, :] + np.reshape(sph_grid[:, np.newaxis, :] *
                                                                 r_curr[np.newaxis, :, np.newaxis], [-1, 3]))
            full_grid = np.concatenate(full_grid, axis=0)

        elif grid_type == "full":  # create a full grid : cartesian grid with spherical refinement around image pos
            # rough cartesian grid
            valmin, valmax = np.min(image_pos, axis=0) - overflow, np.max(image_pos, axis=0) + overflow
            full_grid = np.meshgrid(np.arange(valmin[0], valmax[0], grid_spacing[0]),
                                    np.arange(valmin[1], valmax[1], grid_spacing[0]),
                                    np.arange(valmin[2], valmax[2], grid_spacing[0]))

            full_grid = np.stack([full_grid[0].flatten(), full_grid[1].flatten(), full_grid[2].flatten()], axis=1)
            # spherical refinement around image_pos
            sph_grid, _ = create_grid_spherical(rmin=grid_spacing[3], rmax=grid_spacing[4], dr=grid_spacing[1],
                                                dphi=grid_spacing[2], dtheta=grid_spacing[2])

            full_grid = np.concatenate([full_grid, np.zeros([len(sph_grid)*len(image_pos)])], axis=0)

        print("Number of grid nodes: ", len(full_grid))
        # compute the rank of the gamma matrix
        print("Computing gamma matrix rank")
        gamma_mat = gamma_op(image_pos, N, mic_pos, fs)
        rank = np.linalg.matrix_rank(gamma_mat)
        dict_res["gamma_rank"] = rank

        # compute the hessian of eta_v at each spike position
        print("Computing precertificate")
        pvec = pV(image_pos, N, mic_pos, fs)
        print("Computing eta_v hessian")
        hess_det = np.zeros(len(image_pos))
        hess = etav_der2(image_pos, pvec, N, mic_pos, fs)
        for i in range(len(image_pos)):
            hess_det[i] = np.abs(np.linalg.det(hess[i, :, :]))

        dict_res["d2eta_val"] = hess_det

        # compute the maximum value of |eta_v| in parallel using multiprocessing.Pool
        print("Computing max |eta_v|")
        def max_eta_worker(i, batch_size):
            print("batch ", i)
            return np.max(np.abs(etav(full_grid[i*batch_size:(i+1)*batch_size], pvec, N, mic_pos, fs)))

        pool = mp.Pool(2)
        # split the grid into batches, split the batches among the workers
        batch_size = 5000
        n_batches = int(np.ceil(len(full_grid)/batch_size))
        print("Number of batches: ", n_batches)
        fun = partial(max_eta_worker, batch_size=batch_size)
        max_eta = np.max(pool.map(fun, range(n_batches)))
        dict_res["max_eta"] = max_eta

        # save the results
        json_to_dict(dict_path, dict_res)


    print("total execution time : {} s".format(time.time() - tstart))
