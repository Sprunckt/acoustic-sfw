import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
import pandas as pd
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from scipy.signal import decimate
from sklearn.linear_model import Lasso, LinearRegression
from src.simulation.utils import (disp_measure, c, cut_vec_rir, create_grid_spherical)
import multiprocessing
import time
from abc import ABC, abstractmethod
import os

stop_tol = 1e-6
merge_tol = 0.02

ampl_freeze_threshold = 0.05
dist_freeze_threshold = 0.01


def flat_to_multi_ind(ind, N):
    return ind // N, ind % N


def compute_time_sample(N, fs):
    """
    Compute the full discretization of the time interval
    """
    return np.arange(N) / fs


def sliding_window_norm(a, win_length):
    """
    Compute the maximum of a sliding mean on the energy.
    """
    energy = np.abs(a)**2
    sliding_mean = np.convolve(energy, np.full(win_length, 1./win_length), 'same')
    ind = np.argmax(sliding_mean)
    return ind, sliding_mean[ind]


class SFW(ABC):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], fs: float,
                 lam: float = 1e-2, N: int = 0, fc: float = None, deletion_tol: float = 0.05):
        self.fs, self.lam = fs, lam
        self.fc = fs if fc is None else fc
        self.d = 3
        self.N = N
        if type(y) == np.ndarray:
            self.y = y
        else:
            self.y = self.gamma(y[0], y[1])

        self.ak, self.xk = np.zeros(1), np.zeros((1, self.d))  # initializing with the null measure
        self.xkp = np.zeros((1, self.d))  # temporary storage for spike locations

        self.res = self.y.copy()  # residue
        self.res_norm = np.linalg.norm(self.res[:self.N])  # full RIR norm (including all microphones)
        self.nk = 0
        self.it = 0

        # used to put boundaries on positions and amplitudes
        self.max_norm, self.max_ampl = 0., 0.

        self.opt_options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08,
                            'maxiter': None, 'disp': False, 'return_all': False, 'finite_diff_rel_step': None}
        self.eta_jac, self.slide_jac = None, None
        self.timer = None

        # spike history for freezing spikes
        self.old_ak, self.old_xk = np.zeros(0), np.zeros((0, self.d))
        self.spike_hist_counter = np.zeros(0, dtype=int)
        self.active_spikes, self.n_active = np.zeros(0, dtype=bool), 0
        self.freeze_step = 0
        self.slide_control = 0  # control over the sliding step : 0 -> every iteration, 1 -> only at the end, 2 -> none
        self.y_freeze = self.y

        # manage saves
        self.save, self.save_path = False, None
        self.saving_freq = 0
        self.save_list = []

        self.ncores = len(os.sched_getaffinity(0))

        self.deletion_tol = deletion_tol

    @abstractmethod
    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Linear operator (M(R^d) -> R^J)
        Args:
            -a (array) : flat array containing the amplitudes of the spikes
            -x (array) : positions of the spikes, shape (len(a), d)
        Return: array containing the evaluation of gamma on the measure (shape (J,))
        """
        pass

    @abstractmethod
    def etak(self, x: np.ndarray) -> float:
        """Objective function for the new spike location optimization ; the normalization by lambda is only taken in
        account for the stopping criterion and not during optimization"""
        pass

    @abstractmethod
    def _LASSO_step(self) -> Tuple[Lasso, np.ndarray]:
        """
        Performs the lasso step and returns a tuple (lasso_fitter, ak) (the sklearn LASSO object and the ak
        coefficients)
        """
        pass

    def _get_amplitude_fitter(self):
        """
        Regression model used for amplitude optimization (LASSO if lambda > 0).
        """
        if self.lam > 0:
            return Lasso(positive=True, alpha=self.lam)
        else:
            return LinearRegression(positive=True)

    def _obj_lasso(self, a):
        """
        Objective function for the optimization on spike amplitudes.
        """

        return 0.5 * np.sum((self.gamma(a, self.xkp) - self.y) ** 2) + self.lam * np.sum(np.abs(a))

    def _slide_step(self, ini, verbose=False):
        tstart = time.time()

        # determining if it is a partial or full sliding step
        n_bounds = len(ini) // (self.d + 1)

        if n_bounds == self.nk:
            args = (self.y, self.nk)
        elif n_bounds == self.n_active:
            args = (self.y_freeze, self.n_active)
        else:
            args = None
            print("invalid bounds vector length for LBFGS-B")
            exit(1)

        # bounds for the amplitudes and positions
        bounds = [(0., self.max_ampl)] * n_bounds + [(-self.max_norm, self.max_norm)] * n_bounds * self.d

        opti_res = minimize_parallel(self._obj_slide, ini, jac=self.slide_jac, bounds=bounds,
                                     args=args, parallel={'max_workers': self.ncores})
        mk, nit_slide, val_fin, tend = opti_res.x, opti_res.nit, opti_res.fun, time.time()
        if verbose:
            print("Initial/final values : {} {} \n".format(self._obj_slide(ini, y=args[0], n_spikes=args[1]), val_fin))
            print("Optimization converged in {} iterations, exec time : {} s".format(nit_slide,
                                                                                     tend - tstart))
            if not opti_res.success:
                print("Last optimization failed, reason : {}".format(opti_res.message))

        return mk, nit_slide, val_fin, opti_res

    def _obj_slide(self, var, y, n_spikes):
        """
        Objective function for the sliding step (optimization on locations and amplitude).
        """
        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        return 0.5 * np.sum((self.gamma(a, x) - y) ** 2) + self.lam * np.sum(np.abs(a))

    def _optigrid(self, x, success=False):
        """
        Wrapper for the spike selection optimization step
        """
        opt_res = minimize(self.etak, x, jac=self.eta_jac, method="BFGS", options=self.opt_options)
        if success:
            return [*opt_res.x, opt_res.fun, opt_res.nit, opt_res.success, opt_res.message]
        else:
            return [*opt_res.x, opt_res.fun, opt_res.nit]

    def _stop(self, verbose=True):
        """
        Called at algorithm definitive stop, handles ultimate sliding step/step saving
        """
        if self.slide_control == 1:  # sliding once before the end
            print("Last sliding step before stopping")
            ini = np.concatenate([self.ak, self.xk.flatten()])
            ini_val = self._obj_slide(ini, y=self.y, n_spikes=self.nk)
            mk, nit_slide, val_fin, opti_res = self._slide_step(ini, verbose=verbose)
            if val_fin < ini_val:
                self.ak, self.xk = mk[:self.nk], mk[self.nk:].reshape([-1, self.d])
                if self.save:
                    self.save_list.append([self.it+1, time.time() - self.timer, self.ak, self.xk])
            else:
                if verbose:
                    print("Energy increased, ignoring the sliding step")

        if self.save:
            df = pd.DataFrame(self.save_list, columns=["iter", "t", "ak", "xk"])
            df.to_csv(self.save_path, index=False)

        if verbose:
            print("total exec time : {} s".format(time.time() - self.timer))

        return self.ak, self.xk

    def merge_spikes(self):
        ak_tmp, xk_tmp = self.ak.copy(), self.xk.copy()
        k, tmp_nk, tot_merged = 0, self.nk, 0

        while k < tmp_nk:
            null_ind = np.zeros(tmp_nk, dtype=bool)
            for m in range(k + 1, tmp_nk):
                curr_x, curr_ax = xk_tmp[k], ak_tmp[k]
                curr_y, curr_ay = xk_tmp[m], ak_tmp[m]
                if np.linalg.norm(curr_x - curr_y) < merge_tol:
                    tot_merged += 1
                    null_ind[m] = True
                    # merged spike position (weighted mean of the spikes)
                    xk_tmp[k] = ((np.abs(curr_x) * np.abs(curr_ax) + np.abs(curr_y) * np.abs(curr_ay))
                                 / (np.abs(curr_ax) + np.abs(curr_ay)))
                    ak_tmp[k] += curr_ay

            # deleting the merged spikes
            ak_tmp = ak_tmp[~null_ind]
            xk_tmp = xk_tmp[~null_ind]
            tmp_nk = (~null_ind).sum()  # number of remaining spikes
            k += 1
        self.ak, self.xk = ak_tmp.copy(), xk_tmp.copy()
        self.nk = len(self.ak)

        return tot_merged

    @abstractmethod
    def _grid_initialization_function(self, search_method, grid, nmic, verbose, **parameters):
        """Functions that configures a grid for the initialization of the step finding step."""
        pass

    @abstractmethod
    def _get_normalized_fun(self):
        """Return a tuple of callables containing the jacobians for the eta optimization and sliding steps. """
        pass

    def compute_residue(self):
        """Compute and update the residue."""
        gk = self.gamma(self.ak, self.xk)
        self.res = self.y - gk

        return self.res

    def _algorithm_start_callback(self, **args):
        """Called at algorithm start"""
        pass

    def _iteration_start_callback(self, **args):
        """Called at the beginning of each iteration"""
        pass

    def _append_history(self, a, x):
        """Add a new active spike"""
        self.old_xk = np.concatenate([self.old_xk, x], axis=0)
        self.old_ak = np.concatenate([self.old_ak, a])
        self.spike_hist_counter = np.append(self.spike_hist_counter, -1)  # incremented instantly at history update
        self.active_spikes = np.append(self.active_spikes, True)

    def _update_history(self, ind_to_delete):
        """Update the active spikes based on the given null amplitude indices. Checks for the spike that should be
        frozen based on the distance they moved since their last checkup."""
        if np.any(ind_to_delete):  # only keep the non-zero spikes
            self.old_xk = self.old_xk[~ind_to_delete]
            self.old_ak = self.old_ak[~ind_to_delete]
            self.spike_hist_counter = self.spike_hist_counter[~ind_to_delete]
            self.active_spikes = self.active_spikes[~ind_to_delete]

        self.spike_hist_counter[self.active_spikes] += 1
        # find the spikes for which a refresh is necessary
        maxit_reached = self.spike_hist_counter >= self.freeze_step
        self.spike_hist_counter[maxit_reached] = 0  # reset the counter

        # check if these spikes' positions and amplitudes have sufficiently moved
        ampl_dist = np.abs(self.ak[maxit_reached] - self.old_ak[maxit_reached])
        spike_dist = np.linalg.norm(self.xk[maxit_reached] - self.old_xk[maxit_reached], axis=1)
        freeze = np.zeros_like(maxit_reached, dtype=bool)
        freeze[maxit_reached] = (ampl_dist < ampl_freeze_threshold) & (spike_dist < dist_freeze_threshold)

        ind_freeze = maxit_reached & freeze
        if np.any(ind_freeze):
            self.active_spikes[ind_freeze] = False  # freeze the unmoving spikes
            # update the semi-residual
            self.y_freeze = self.y_freeze - self.gamma(self.ak[ind_freeze], self.xk[ind_freeze])
        ind_active_mod = maxit_reached & ~freeze  # indices that are not frozen and need a refresh
        # update the active spikes
        self.old_ak[ind_active_mod] = self.ak[ind_active_mod]
        self.old_xk[ind_active_mod] = self.xk[ind_active_mod]
        self.n_active = self.active_spikes.sum()

    def _on_stop(self, reason=None, verbose=False):
        """Method called when a stopping criterion is met, should return True if the algorithm has to stop on the spot,
        False if it is allowed to go on through (for example if the RIR is not extended to its full length)."""
        return True

    def _it_end_cb(self):
        """
        Iteration end callback.
        """
        # update the residue
        self.compute_residue()

        if self.save and self.it % self.saving_freq == 0:
            self.save_list.append([self.it, time.time() - self.timer,
                                   self.ak.copy(), self.xk.copy()])

    def reconstruct(self, grid=None, niter=7, min_norm=-np.inf, max_norm=np.inf, max_ampl=np.inf,
                    search_method="rough", spike_merging=False, spherical_search=0,
                    use_hard_stop=True, verbose=True, early_stopping=False,
                    plot=False, algo_start_cb=None, it_start_cb=None, nmic=None,
                    slide_opt=None, saving_param=None, opt_param=None) -> (np.ndarray, np.ndarray):
        """
        Apply the SFW algorithm to reconstruct the measure based on the measurements self.y.

        Args:
            -grid (array or float): parameter used to specify the grid search. It is passed to the grid initialization
            function. For the time domain case : if spherical_search == 0, should be the array of the grid.
        If spherical_search == 1, if it is an array it is assumed to be a spherical grid of radius 1 that is scaled
        accordingly during the algorithm. If it is a float, it is assumed to be the angular step for a spherical grid
        of radius 1, that is scaled depending on the radius r by 1/log(r) to keep a good density of nodes.
            -niter (int): maximal number of iterations
            -max_ampl (float): upper bound on spikes amplitudes
            -min_norm (float): minimal norm allowed for the position found at the end of the grid search
            -max_norm (float): used as bounds for the coordinates of the spike locations in each direction
            -use_hard_stop (bool): if True, add max|etak| <= 1 as a stopping condition
            -early_stopping (bool): if True, stop at the end of an iteration if the last spike found has zero amplitude
            -search_method (str): grid search methods for the spike position search. If "rough" : perform a coarse
        optimization on each point of the grid before refining on the best position. If "full" : perform a fine
        optimization on each grid point (costly). If "naive" : find the best value on the grid and use it as
        initialization (quicker but less precise with coarser grids). If "decimation": decimates the signal several
        times and apply the rough method on the decimated, using the best values at each decimation level as
        initialization for the next level until the complete signal is reached. The number of decimations and kept
        points at each level can be specified in a dict "decimation_args" that should be contained in the start_cb dict
        and can have keys: "nb_decimation", "discard_factor".
            -opt_param (dict): parameters for the spike finding optimization.
        Keys: * 'roughgtol', 'gtol' tolerance for the optimization algorithm (roughgtol is for the 'rough' pre-search)
              * 'roughmaxiter': maximum number of iterations for the rough pre-search.
            -spherical_search (int): if equal to 1 : assume that the given grid is spherical. The maximum energy spike
        of the residual is used to find the distance from a microphone to an image source, and applying a grid search
        on the corresponding sphere. The grid is parametrized by the grid argument. The argument nmic can be used to
        increase the number of considered microphone during the initialization.
            -nmic (int): number of microphone considered for the spherical search (ie one sphere per microphone)
            -algo_start_cb (dict): dictionary containing the arguments passed to the algorithm start callback
            it_start_cb (dict): dictionary containing the arguments passed to the iteration start callback
            -slide_opt (dict) : dictionary containing the options for the sliding step. If None : perform a full sliding
        at each step. Else : should contain a key "method", the associated value being in ["slide_once", "no_slide",
        "freeze"]. Behavior : * "slide_once" : skip the sliding step and perform a single sliding at the end
                * "no_slide" : completely skip the sliding step
                * "freeze" : the additional key "freeze_step" should be added. Check each spike every
        "freeze_step" iterations. If the spike has not moved sufficiently since the last check, the spike is frozen and
        is not allowed to slide in the next iterations.
        Additional option : the key "resliding_step" allows for an additional periodic sliding step to be applied on
        every spike (including the frozen ones).
            -saving_param (tuple): if not None, should have the signature (int, str). Save the measure every
        saving_param[0] iteration to a csv file containing : iter (iteration number), t (execution time at the end of
        the iteration), ak (array of amplitudes), xk (array of locations). The file is saved to the saving_param[1]
        path.
         Return:
            (ak, xk) where :
            -ak is a flat array of shape (K,) containing the amplitudes of the recovered measure
            -xk is a (K, d) shaped array containing the locations of the K spikes composing the measure
        """
        if saving_param is not None:
            self.saving_freq = saving_param[0]
            self.save_path = saving_param[1]
            self.save = True
            if not os.path.exists(os.path.dirname(self.save_path)):
                print("{} does not exists".format(self.save_path))
                exit(1)

        default_maxiter, default_gtol = 100, 1e-5
        if opt_param is not None:
            rough_gtol, gtol = opt_param.get("roughtol", self.opt_options["gtol"]),  opt_param.get("gtol", default_gtol)
            rough_maxiter = opt_param.get("roughmaxiter", default_maxiter)
        else:
            gtol, rough_gtol = default_gtol, self.opt_options["gtol"]
            rough_maxiter = default_maxiter
        if nmic is None:
            nmic = 1

        self.timer = time.time()
        self.eta_jac, self.slide_jac = self._get_normalized_fun()

        self.max_norm = max_norm
        self.max_ampl = max_ampl

        if self.lam == 0.:
            use_hard_stop = False
            if verbose:
                print("lambda=0, disabling eta stopping criterion")

        # checking the sliding options
        if slide_opt is None:
            resliding_step = 0
        else:
            slide_method = slide_opt.get("method")
            resliding_step = slide_opt.get("resliding_step", 0)
            assert resliding_step >= 0, "resliding_step should be 0 or strictly positive"

            assert slide_method is not None, "The key 'method' should be specified in slide_opt"
            if slide_method == 'slide_once':
                self.slide_control = 1
            elif slide_method == 'freeze':
                self.freeze_step = slide_opt.get("freeze_step", 0)
                assert self.freeze_step >= 0, "freeze_step should be 0 or strictly positive"
            elif slide_method == 'no_slide':
                self.slide_control = 2
            else:
                print("method '{}' not recognized".format(slide_method))

        if verbose:
            print("Executing on {} cores".format(self.ncores))

        reslide_counter = 0

        search_grid = grid.copy()  # making a copy to be sure the original grid is not modified afterwards
        assert search_grid is not None, "a grid must be specified for the initial grid search"

        if algo_start_cb is None:
            algo_start_cb = {}

        if it_start_cb is None:
            it_start_cb = {}

        it_start_cb["verbose"] = verbose
        algo_start_cb["verbose"] = verbose

        if search_method == 'decimation':
            assert isinstance(self, TimeDomainSFW), "decimation only supported in standard time domain"
            algo_start_cb["activate_decimation"] = True

        start_cb_ret = self._algorithm_start_callback(**algo_start_cb)

        while self.it < niter:
            self.it += 1
            if verbose:
                print("iteration {}, residual norm : {}".format(self.it, self.res_norm))

            self._iteration_start_callback(**it_start_cb)
            # find argmax etak to get the new spike position (step 3)
            if verbose:
                print("Optimizing for spike position -----")

            tstart = time.time()

            if spherical_search == 1:  # take argmax on the complete rir and search on the corresponding sphere
                if search_method == "decimation":
                    search_grid = self._grid_initialization_function(search_method, grid, nmic, dec_obj=start_cb_ret,
                                                                     verbose=verbose)
                else:
                    search_grid = self._grid_initialization_function(search_method, grid, nmic, verbose=verbose)

            if verbose:
                print("Starting a grid search to minimize etak, method : ", search_method)

            if search_method == "rough":
                # perform a low precision search at each grid point
                self.opt_options["gtol"] = rough_gtol
                self.opt_options["maxiter"] = rough_maxiter

                # spreading the loop over multiple processors
                p = multiprocessing.Pool(self.ncores)
                gr_opt = np.array(p.map(self._optigrid, search_grid))
                p.close()

                # searching for the best result over the grid at a minimal distance from the origin
                norms = np.linalg.norm(gr_opt[:, :self.d], axis=1)
                kept_ind = gr_opt[norms > min_norm]

                if len(kept_ind) == 0:  # it means the spikes found are all inside the ball of radius min_norm
                    if verbose:
                        print("Cannot find a spike outside the minimal norm ball")
                    if self._on_stop(verbose=verbose):
                        return self._stop(verbose=verbose)
                    else:
                        self._it_end_cb()
                        continue
                else:
                    best_ind = np.argmin(kept_ind[:, self.d])
                    best_x = kept_ind[best_ind, :self.d]
                    nit = kept_ind[best_ind, self.d+1]

                # perform a finer optimization using the position found as initialization
                self.opt_options["gtol"] = gtol
                self.opt_options["maxiter"] = None

                gr_opt = self._optigrid(best_x, success=True)
                gr_opt[self.d+1] += nit  # increment by the previous number of iterations

            elif search_method == 'naive':  # naive method : searching for argmin on grid and using as initialization
                p = multiprocessing.Pool(self.ncores)
                mapping = p.map(self.etak, search_grid)
                p.close()

                ind_max = np.argmin(mapping)

                self.opt_options["gtol"] = gtol
                gr_opt = self._optigrid(search_grid[ind_max], success=True)

            else:  # decimation method
                success, res = start_cb_ret.grid_search(search_grid, min_norm=min_norm, rough_gtol=rough_gtol,
                                                        rough_maxiter=rough_maxiter, verbose=verbose)

                if not success:  # case where no spikes were found outside of the minimal norm ball
                    if self._on_stop(verbose=verbose):
                        return self._stop(verbose=verbose)
                    else:
                        self._it_end_cb()
                        continue

                # perform a finer optimization using the position found as initialization
                self.opt_options["gtol"] = gtol
                self.opt_options["maxiter"] = None

                gr_opt = self._optigrid(res, success=True)

            x_new, best_val, nit = np.reshape(gr_opt[:self.d], [1, self.d]), gr_opt[self.d], gr_opt[self.d+1]

            if self.lam > 0.:
                etaval = np.abs(best_val) / self.lam
            else:
                etaval = np.abs(best_val)

            if verbose:
                print("exec time for grid optimization : ", time.time() - tstart)
            success = gr_opt[self.d+2]
            if not success:
                print("etak optimization failed, reason : {}".format(gr_opt[self.d+3]))

            if verbose:
                print("Optimization converged in {} iterations".format(nit))
                print("New position : {} \n eta value : {}".format(x_new, etaval))

            if use_hard_stop and etaval <= 1 + stop_tol:
                if verbose:
                    print("Stopping criterion met : etak(x_new)={} < 1".format(etaval))
                if self._on_stop(reason='etak_stop', verbose=verbose):
                    return self._stop(verbose=verbose)
                else:
                    self._it_end_cb()
                    continue

            # solve LASSO to adjust the amplitudes according to the new spike (step 7)
            if self.nk > 0:
                self.xkp = np.concatenate([self.xk, x_new], axis=0)
            else:
                self.xkp = np.array([x_new]).reshape(1, self.d)
            self.nk += 1  # increment the number of spikes
            self.n_active += 1  # increment the number of active spikes
            if verbose:
                print("Optimizing for spike amplitudes --------")

            tstart = time.time()

            lasso_fitter, ak_new = self._LASSO_step()

            if verbose:
                if self.lam > 0:
                    print("LASSO solved in {} iterations, exec time : {} s".format(lasso_fitter.n_iter_,
                                                                                   time.time() - tstart))
                print("New amplitudes : {} \n".format(ak_new))

            if self.slide_control > 0:
                nit_slide, decreased_energy = 0, False
            else:
                # sliding step : optimize to adjust both the positions and amplitudes
                tmp_active = np.append(self.active_spikes, [True])
                ini = np.concatenate([ak_new[tmp_active],
                                      self.xkp[tmp_active].flatten()])

                ini_val = self._obj_slide(ini, y=self.y_freeze, n_spikes=self.n_active)

                if verbose:
                    print("Sliding step --------")

                mk, nit_slide, val_fin, opti_res = self._slide_step(ini, verbose=verbose)

                decreased_energy = val_fin < ini_val

                # use the new measure if the sliding step decreased the energy, else keep the old values
                if decreased_energy:
                    ak_new[tmp_active] = mk[:self.n_active]
                    self.xkp[tmp_active] = mk[self.n_active:].reshape([-1, self.d])
                else:
                    if verbose:
                        print("Energy increased, ignoring this step")

            self.ak, self.xk = ak_new, self.xkp

            if resliding_step:
                reslide_counter += 1
                if reslide_counter >= resliding_step:
                    ini = np.concatenate([self.ak, self.xk.flatten()])
                    ini_val = self._obj_slide(ini, y=self.y, n_spikes=self.nk)
                    if verbose:
                        print("Periodic sliding step on every spike ---")
                    mk, nit_slide, val_fin, opti_res = self._slide_step(ini, verbose=verbose)

                    if val_fin < ini_val:  # check if the energy has decreased
                        self.ak, self.xk = mk[:self.nk], mk[self.nk:].reshape([-1, self.d])
                        reslide_counter = 0
                    else:
                        if verbose:
                            print("Energy increased, sliding not applied - retrying next iteration")

            # deleting null amplitude spikes
            ind_null = np.asarray(np.abs(self.ak) < self.deletion_tol)
            self.ak = self.ak[~ind_null]
            self.xk = self.xk[~ind_null, :]
            self.nk = len(self.ak)

            if self.freeze_step and self.nk > 0:  # update the frozen spikes history
                self._append_history(self.ak[self.nk-1:], self.xk[self.nk-1].reshape(1, 3))
                self._update_history(ind_null)  # if the new spike is null it is instantly deleted. todo: clean this
                if verbose:
                    print("active spikes : \n", np.where(self.active_spikes)[0])
            else:  # all the spikes are active
                self.active_spikes = np.ones(self.nk, dtype=bool)
                self.n_active = self.nk

            if self.nk == 0:  # no non-zero spikes
                self.ak, self.xk, self.nk = np.zeros(1), np.zeros((1, self.d)), 0
                if self._on_stop(reason='All spikes null', verbose=verbose):
                    print("Error : all spikes are null, stopping")
                    return self._stop(verbose=verbose)
                else:
                    self._it_end_cb()
                    continue

            # last spike is null and minor changes from the previous iteration at the sliding step
            elif (early_stopping and ind_null[-1]
                  and (nit_slide == 1 or not decreased_energy)):

                if self._on_stop(reason='last_null', verbose=verbose):  # stopping
                    print("Last spike has null amplitude, stopping")
                    return self._stop(verbose=verbose)
                else:  # pulling through (after RIR extension for instance)
                    self._it_end_cb()
                    continue

            if spike_merging:
                if verbose:
                    print("merging spikes --")
                tot_merged = self.merge_spikes()
                if verbose:
                    print("{} spikes merged".format(tot_merged))

            if verbose:
                print("New measure :")
                disp_measure(self.ak, self.xk)

            # save if necessary and update residue
            self._it_end_cb()

            if plot:
                gk = self.gamma(self.ak, self.xk)  # current rir
                if self.y.dtype == float:
                    plt.plot(self.y, label="measures")
                    plt.plot(gk, '--', label="current value")
                    plt.plot(self.res, label="residue")

                else:
                    fig, ax = plt.subplots(2)
                    ax[0].set_title("real part"), ax[1].set_title("imaginary part")
                    ax[0].plot(np.real(self.y)), ax[1].plot(np.imag(self.y), label="measures")
                    ax[0].plot(np.real(gk), '--'), ax[1].plot(np.imag(gk), '--', label="current")
                    ax[0].plot(np.real(self.res), '--'), ax[1].plot(np.imag(self.res), '--', label="residue")
                plt.legend()
                plt.show()
        print("Maximum number of iterations {} reached".format(niter))
        return self._stop(verbose=verbose)


class TimeDomainSFW(SFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], mic_pos: np.ndarray,
                 fs: float, N: int, lam: float = 1e-2, fc=None, deletion_tol=5e-2):
        """
        Args:
            -y (ndarray or tuple(ndarray, ndarray)) : measurements (shape (J,) = (N*M,)) or tuple (a,x) containing the
        amplitudes and positions of the measure that has to be reconstructed. In that last case the ideal observations
        are computed.
            -mic_pos (ndarray) : positions of the microphones (shape (M,d))
            -fs (float) : sampling frequency
            -N (int) : number of time samples
            -lam (float) : penalization parameter of the BLASSO.
        """

        # number of time samples of the full RIR
        self.global_N = N
        # number of time samples used in practice (might change during execution)
        self.NN = compute_time_sample(N, fs)  # discretized time interval

        self.mic_pos, self.M = mic_pos, len(mic_pos)

        self.d = mic_pos.shape[1]
        assert 1 < self.d < 4, "Invalid dimension d"

        self.J = self.M * N

        # getting attributes and methods from parent class
        super().__init__(y=y, fs=fs, lam=lam, N=N, fc=fc, deletion_tol=deletion_tol)
        # full rir
        self.global_y = self.y

        # antenna diameter
        self.antenna_diameter = np.sqrt(np.max(np.sum((self.mic_pos[np.newaxis, :, :] -
                                                       self.mic_pos[:, np.newaxis, :])**2, axis=-1)))

        # variables used to extend the time interval progressively
        mc_rir = self.y.reshape(self.M, self.N)  # multi-channel RIR
        # cumulated energy, mean on the microphones
        self.cumulated_energy = np.mean(np.sqrt(np.cumsum(mc_rir ** 2, axis=1)), axis=0)
        self.cut_ind = None  # indexes for cutting the RIR
        self.n_cut = 0
        self.swap_counter = -1  # number of full iterations since last extension
        self.swap_factor = 0.  # growth criterion on the residual norm to extend the RIR
        self.swap_frequency = 0  # number of iterations before extending the RIR
        self.res_norm = self.cumulated_energy[-1]  # mean total energy of the multichannel RIR
        self.old_norm = self.res_norm
        self.current_ind = 0  # index of the last cutting threshold used
        assert self.y.size == self.J, "invalid measurements length, {} != {}".format(self.y.size, self.J)

    def _algorithm_start_callback(self, verbose=False, n_cut=0, int_start=None,
                                  swap_frequency=10, swap_factor=0.5, method='energy', min_dist=2.,
                                  decimation_args=None, activate_decimation=False):
        """Parametrize the segmentation of the time interval.
        Args:-n_cut (int) : number of cutting thresholds (number of subintervals -1). Set to 0 to use the full RIR.
        10 is a good default value for 60ms.
        -method (str): 'energy': cut according to cumulative square sum
                       'time': cut linearly the RIR, starting from the time needed to propagate through the distance
        'min_dist' (ie min_dist/343).
        -int_start (float) : starting value for the choice of the starting threshold in the energy method (maximum value
         is the norm of the RIR), default : norm/3
        -swap_frequency (int) : maximum number of full iterations before extending to the next threshold"""

        if n_cut > 0:

            if method == 'energy':
                cut_ind = []
                int_end = self.cumulated_energy[-1]  # vector norm of the first microphone RIR
                if int_start is None:
                    int_start = int_end / 3.

                cut_values = np.linspace(int_start, int_end, n_cut, endpoint=False)

                overshoot = int(self.fs * self.antenna_diameter / c)  # worst minimal time to go through the antenna
                cut_ind.append(np.argmax(self.cumulated_energy > cut_values[0]) + overshoot)

                for i in range(1, n_cut):
                    # find the smallest time sample for which an energy threshold is reached, overshoot by a few samples
                    new_ind = np.argmax(self.cumulated_energy > cut_values[i]) + overshoot
                    if new_ind - cut_ind[-1] > 50:  # only add the next index if there is a minimal time separation
                        cut_ind.append(new_ind)
                cut_ind.append(self.global_N - 1)  # last index of the complete first microphone RIR
                self.cut_ind = np.array(cut_ind)

            elif method == 'time':
                # time sample corresponding to sound propagation over min_dist
                ind_start = int(np.rint(self.fs*min_dist / c))
                ind_end = self.global_N - 1
                self.cut_ind = np.linspace(ind_start, ind_end, n_cut+2, endpoint=True, dtype=int)[1:]

            else:

                print("Segmenting method '{}' not recognized".format(method))
                exit(1)

            if verbose:
                print("Cutting the RIR according to {} thresholds".format(len(self.cut_ind) - 1))
                print("Cutting samples : ", self.cut_ind)

            # update N, NN according to the segmentation
            self.N = self.cut_ind[0]

            self.NN = compute_time_sample(self.N, self.fs)

            self.y = cut_vec_rir(self.global_y, self.M, self.global_N, self.N)
            self.y_freeze = self.y.copy()
            self.swap_frequency = swap_frequency
            self.swap_factor = swap_factor
            self.compute_residue()
            self.old_norm = self.res_norm
            self.n_cut = len(self.cut_ind)
        else:
            self.cut_ind = np.empty(0)
        self.current_ind = 0

        if activate_decimation:
            if decimation_args is None:
                decimation_args = {"nb_decimation": 2}
            dec = Decimation(self, **decimation_args)
            return dec

    def _extend_rir(self, reason, verbose=False):
        if self.current_ind < self.n_cut - 1:  # check that the last threshold has not been reached
            self.old_norm = self.res_norm
            self.current_ind += 1  # update the slice index
            # update the norm : current residual + the norm of the appended RIR segment
            app_norm = (self.cumulated_energy[self.cut_ind[self.current_ind]] -
                        self.cumulated_energy[self.cut_ind[self.current_ind - 1]])
            self.old_norm += app_norm

            # update RIR length
            self.N = self.cut_ind[self.current_ind]
            self.NN = compute_time_sample(self.N, self.fs)
            self.y = cut_vec_rir(self.global_y, self.M, self.global_N, self.N)
            if self.freeze_step:
                # extend the semi-residual
                frozen_ind = ~self.active_spikes
                self.y_freeze = self.y - self.gamma(self.ak[frozen_ind],
                                                    self.xk[frozen_ind, :])
            else:
                self.y_freeze = self.y
            self.compute_residue()

            if verbose:
                print("Extending RIR from {}s to {}s after {} it".format(self.cut_ind[self.current_ind - 1] / self.fs,
                                                                         self.cut_ind[self.current_ind] / self.fs,
                                                                         self.swap_counter))
                print("Reason : {}".format(reason))

            self.swap_counter = 0
            return True
        else:
            return False

    def _on_stop(self, reason=None, verbose=False):
        """Should return False if the RIR can be extended (meaning the algorithm is allowed to pull through), True
        otherwise."""
        should_stop = not self._extend_rir(reason="stopping criterion met", verbose=verbose)
        return should_stop

    def _iteration_start_callback(self, verbose=False):
        self.swap_counter += 1
        curr_norm, thresh = self.res_norm, self.swap_factor*self.old_norm
        threshold_reached = curr_norm < thresh  # norm criterion
        max_iter_reached = self.swap_counter > self.swap_frequency
        if threshold_reached:
            can_extend = self._extend_rir("norm criterion reached: {} < {}".format(curr_norm, thresh), verbose=verbose)
        elif max_iter_reached:
            can_extend = self._extend_rir("max iteration reached", verbose=verbose)
        else:
            can_extend = False

        return can_extend

    def sinc_filt(self, t):
        """
        The filter applied for each measurement.
        """
        return np.sinc(t * self.fc)

    def sinc_der(self, t):
        """
        Derivate of the filter
        """
        w = np.pi * self.fc
        return (t * np.cos(w * t) - np.sin(w * t) / w) / (t ** 2 + np.finfo(float).eps)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum( M, K, N, axis=1)
        return np.sum(self.sinc_filt(self.NN[np.newaxis, :] - dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / dist[:, :, np.newaxis]
                      * a[np.newaxis, :, np.newaxis], axis=1).reshape(-1)

    def compute_residue(self):
        gk = self.gamma(self.ak, self.xk)
        self.res = self.y - gk
        self.res_norm = np.mean(np.linalg.norm(self.res.reshape(self.M, self.N), axis=1), axis=0)  # mean residual norm

        return self.res

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # shape (M, N, K) -> (J, K)
        gamma_mat = np.reshape(self.sinc_filt(self.NN[np.newaxis, :, np.newaxis] - dist[:, np.newaxis, :] / c)
                               / 4 / np.pi / dist[:, np.newaxis, :], newshape=(self.M * self.N, -1))

        lasso_fitter = self._get_amplitude_fitter()
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * self.y)
        return lasso_fitter, lasso_fitter.coef_.flatten()

    def etak(self, x: np.ndarray) -> float:
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] - dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis]).flatten()

        return -np.sum(self.res * gammaj)

    def _jac_etak(self, x):
        diff = x[np.newaxis, :] - self.mic_pos[:, :]  # difference, shape (M, 3)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        int_term = self.NN[np.newaxis, :] - dist[:, np.newaxis] / c  # shape (M, N)

        # sum shape (M,  N) into shape (M,), derivate without the xk_i - xm_i factor
        tens = np.sum(((- self.sinc_filt(int_term) / dist[:, np.newaxis] - self.sinc_der(int_term) / c)
                       / dist[:, np.newaxis] ** 2 / 4 / np.pi) * self.res.reshape(self.M, self.N), axis=1)

        # shape (M,3) into (3,)
        jac = (np.sum(tens[:, np.newaxis] * diff, axis=0).flatten())

        return -jac

    def _jac_slide_obj(self, var, y, n_spikes):
        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        diff = x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]  # difference, shape (M, K, 3)
        # distances from the diracs contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        jac = np.zeros(n_spikes * 4)

        int_term = self.NN[np.newaxis, np.newaxis, :] - dist[:, :, np.newaxis] / c
        # shape (M, K, N) = gamma_j(x_k)
        gamma_tens = self.sinc_filt(int_term) / 4 / np.pi / dist[:, :, np.newaxis]

        # sum_k ak.gamma_j(x_k) - y_j : sum(M, K, N, axis=1) - y = -residue
        residue = (np.einsum("mkn,k->mn", gamma_tens, a, optimize='greedy').reshape(-1) - y)
        # derivates in ak : multiply the residue by gammaj(x_k) and sum on j (meaning m and N)
        jac[:n_spikes] = (np.einsum("mn,mkn->k", residue.reshape(self.M, self.N), gamma_tens, optimize='greedy')
                          + self.lam * np.sign(a))

        # shape (M, K, N), derivative without the xk_i - xm_i factor
        gamma_tens = ((- gamma_tens - self.sinc_der(int_term) / 4 / np.pi / c)
                      / dist[:, :, np.newaxis] ** 2 * residue.reshape(self.M, self.N)[:, np.newaxis, :])

        # original shape (M,K,3,N)
        jac[n_spikes:] = (np.einsum("k,mkn,mki->ki", a, gamma_tens, diff, optimize='greedy').reshape(-1))

        return jac

    def _grid_initialization_function(self, search_method, grid, nmic, verbose, **params):
        if search_method == 'decimation':
            # update residues and vector lengths
            params["dec_obj"].decimate_residue()
            # intialization using the smallest sampling frequency
            return params["dec_obj"].sub_sfw[-1]._grid_initialization_function(search_method="standard", grid=grid,
                                                                               nmic=nmic, verbose=verbose, **params)
        else:
            val, ind_max = np.zeros(self.M), np.zeros(self.M, dtype=int)
            for m in range(self.M):
                ind_max[m], val[m] = sliding_window_norm(self.res[m * self.N: (m + 1) * self.N], 3)

            ind_best = np.argsort(val)[-nmic:]
            r = ind_max[ind_best] * c / self.fs  # radiuses, shape (nkept,)
            search_grid = np.reshape(r[:, np.newaxis, np.newaxis] * grid[np.newaxis, :, :]
                                     + self.mic_pos[ind_best][:, np.newaxis, :], [-1, self.d])

            if verbose:
                print("searching around mic {} at radius {}, {} grid points".format(ind_best, r, len(search_grid)))
            return search_grid

    def _get_normalized_fun(self):
        normalized_eta_jac = self._jac_etak
        slide_jac = self._jac_slide_obj
        return normalized_eta_jac, slide_jac

    def get_cut_ind(self):
        return self.cut_ind

    def update_mic_pos(self, new_pos):
        self.mic_pos = new_pos


class TimeDomainSFWNorm1(TimeDomainSFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], mic_pos: np.ndarray,
                 fs: float, N: int, lam: float = 1e-2, fc=None, deletion_tol=1e-2):
        """Adds a normalization factor 1/(sqrt(sum(1/dist(x, xm)**2)."""
        y = TimeDomainSFW(y, mic_pos, fs, N, lam, fc).y  # necessary if y if a tuple of exact amplitudes and images
        super().__init__(y=y, mic_pos=mic_pos, fs=fs, N=N, lam=lam, fc=fc, deletion_tol=deletion_tol)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] - dist[:, :, np.newaxis] / c)  # (M, K, N)
                  / 4 / np.pi / dist[:, :, np.newaxis] * a[np.newaxis, :, np.newaxis])

        # sum(M, K, N, axis=1) / (K,)
        return np.sum(gammaj / np.linalg.norm(1/dist, axis=0)[np.newaxis, :, np.newaxis], axis=1).flatten()

    def etak(self, x: np.ndarray) -> float:
        """Normalization of etak by 1/norm_2([dist(x, xm)]_m)"""
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] - dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis] / np.linalg.norm(1 / dist)).flatten()

        return -np.sum(self.res * gammaj)

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        gammaj = (self.sinc_filt(self.NN[np.newaxis, :, np.newaxis] - dist[:, np.newaxis, :] / c)
                  / 4 / np.pi / dist[:, np.newaxis, :] / np.linalg.norm(1 / dist, axis=0)[np.newaxis, np.newaxis, :])

        # shape (M, N, K) -> (J, K)
        gamma_mat = np.reshape(gammaj, newshape=(self.M * self.N, -1))

        lasso_fitter = self._get_amplitude_fitter()
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * self.y)
        return lasso_fitter, lasso_fitter.coef_.flatten()

    def _get_normalized_fun(self):
        normalized_eta_jac = "3-point"
        slide_jac = None
        return normalized_eta_jac, slide_jac


class TimeDomainSFWNorm2(TimeDomainSFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], mic_pos: np.ndarray,
                 fs: float, N: int, lam: float = 1e-2, fc=None, deletion_tol=5e-3):
        """Adds a normalization factor 1/(norm_2(gamma(psi)))."""
        y = TimeDomainSFW(y, mic_pos, fs, N, lam, fc).y
        super().__init__(y=y, mic_pos=mic_pos, fs=fs, N=N, lam=lam, fc=fc, deletion_tol=deletion_tol)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # (M, K, N)
        gammaj = self.sinc_filt(self.NN[np.newaxis, :] - dist[:, :, np.newaxis] / c) / dist[:, :, np.newaxis]

        # sum(M, K, N, axis=1) / (K,)
        return np.sum(gammaj * a[np.newaxis, :, np.newaxis]
                      / (np.linalg.norm(gammaj, axis=(0, 2))[np.newaxis, :, np.newaxis]+1e-32), axis=1).flatten()

    def etak(self, x: np.ndarray) -> float:
        """Normalization of etak by 1/norm_2([dist(x, xm)]_m)"""
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] - dist[:, np.newaxis] / c) / dist[:, np.newaxis]).flatten()

        return -np.sum(self.res * gammaj) / (np.linalg.norm(gammaj)+1e-32)

    def _jac_etak(self, x):
        diff = x[:, np.newaxis] - self.mic_pos[:, :].T  # difference, shape (3, M)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=0))

        int_term = self.NN[np.newaxis, :] - dist[:, np.newaxis] / c  # shape (M, N)

        # shape (M, N) = gamma_j(x)
        gamma_tens = self.sinc_filt(int_term) / dist[:, np.newaxis]
        # gamma norm, float
        gamma_norm = np.linalg.norm(gamma_tens)

        # common factor in the derivative in x_k (shape (3, M, N))
        common_factor = ((gamma_tens + self.sinc_der(int_term) / c)[np.newaxis, :, :]
                         * diff[:, :, np.newaxis])

        # shape (3, M, N)
        first_term = (- common_factor / gamma_norm
                      / (dist ** 2)[np.newaxis, :, np.newaxis])
        return -np.sum((first_term
                       + np.sum(common_factor * gamma_tens[np.newaxis, :, :]  # final shape (3,)
                                / (dist ** 2)[np.newaxis, :, np.newaxis], axis=(1, 2))[:, np.newaxis, np.newaxis]
                       * gamma_tens[np.newaxis, :, :] / (gamma_norm ** 3))
                       * self.res.reshape(self.M, self.N)[np.newaxis, :, :], axis=(1, 2))

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        gammaj = (self.sinc_filt(self.NN[np.newaxis, :, np.newaxis] - dist[:, np.newaxis, :] / c)
                  / 4 / np.pi / dist[:, np.newaxis, :])

        # shape (M, N, K) -> (J, K)
        gamma_mat = np.reshape(gammaj / np.linalg.norm(gammaj, axis=(0, 1)), newshape=(self.M * self.N, -1))

        lasso_fitter = self._get_amplitude_fitter()
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * self.y)
        return lasso_fitter, lasso_fitter.coef_.flatten()

    def _jac_slide_obj(self, var, y, n_spikes):
        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        diff = x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]  # difference, shape (M, K, 3)
        # distances from the diracs contained in x to every microphone, shape (M, K), K=len(x)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        jac = np.zeros(n_spikes * 4)

        int_term = self.NN[np.newaxis, np.newaxis, :] - dist[:, :, np.newaxis] / c
        # shape (M, K, N) = gamma_j(x_k)
        gamma_tens = self.sinc_filt(int_term) / dist[:, :, np.newaxis]

        # gamma norm, shape (K,)
        gamma_norm = np.sqrt(np.sum(gamma_tens**2, axis=(0, 2)))

        # sum_k ak.gamma_j(x_k) - y_j : sum(K, M, N, axis=1) - y = -residue
        residue = (np.einsum("mkn,k->mn", gamma_tens, a / gamma_norm, optimize='greedy').reshape(-1) - y)

        # derivative in ak : multiply the residue by gammaj(x_k) and sum on j (meaning m and n)
        jac[:n_spikes] = (np.einsum("mn,mkn->k", residue.reshape(self.M, self.N), gamma_tens, optimize='greedy')
                          / gamma_norm + self.lam*np.sign(a))

        # common factor in the derivative in x_k (shape (M, K, N, 3))
        common_factor = ((gamma_tens + self.sinc_der(int_term) / c)[:, :, :, np.newaxis]
                         * diff[:, :, np.newaxis, :])

        # shape (M, K, N, 3), derivative in x_k
        first_term = (- common_factor / gamma_norm[np.newaxis, :, np.newaxis, np.newaxis]
                      / (dist**2)[:, :, np.newaxis, np.newaxis])  # shape (K, M, N, 3)
        gamma_der = (first_term +
                     np.einsum("mkni,mkn->ki", common_factor, gamma_tens / (dist**2)[:, :, np.newaxis],
                               optimize='greedy')[np.newaxis, :, np.newaxis, :]  # final shape (K,3)
                     * gamma_tens[:, :, :, np.newaxis] / (gamma_norm**3)[np.newaxis, :, np.newaxis, np.newaxis])
        jac[n_spikes:] = (np.einsum("mkni,k,mn->ki", gamma_der, a, residue.reshape(self.M, self.N),
                                    optimize='greedy').reshape(-1))
        return jac

    def _get_normalized_fun(self):
        normalized_eta_jac = self._jac_etak
        slide_jac = self._jac_slide_obj
        return normalized_eta_jac, slide_jac


class EpsilonTimeDomainSFW(TimeDomainSFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], mic_pos: np.ndarray,
                 fs: float, N: int, lam: float = 1e-2, eps=1e-3, fc=None):
        """
        Args:
            -y (ndarray or tuple(ndarray, ndarray)) : measurements (shape (J,) = (N*M,)) or tuple (a,x) containing the
        amplitudes and positions of the measure that has to be reconstructed. In that last case the ideal observations
        are computed.
            -mic_pos (ndarray) : positions of the microphones (shape (M,d))
            -fs (float) : sampling frequency
            -N (int) : number of time samples
            -lam (float) : penalization parameter of the BLASSO.
        """
        self.eps = eps

        super().__init__(y=y, fs=fs, lam=lam, N=N, mic_pos=mic_pos, fc=fc)  # getting attributes and methods from parent class
        temp_sfw = TimeDomainSFW(y=y, fs=fs, lam=lam, N=N, mic_pos=mic_pos)
        self.y = temp_sfw.y.copy()  # overwriting measures initialization

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum( M, K, N, axis=1)
        return np.sum(self.sinc_filt(self.NN[np.newaxis, :] - dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / np.sqrt(dist[:, :, np.newaxis] ** 2 + self.eps)
                      * a[np.newaxis, :, np.newaxis], axis=1).flatten()

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # shape (M, N, K) -> (J, K)
        gamma_mat = np.reshape(self.sinc_filt(self.NN[np.newaxis, :, np.newaxis] - dist[:, np.newaxis, :] / c)
                               / 4 / np.pi / np.sqrt(dist[:, np.newaxis, :] ** 2 + self.eps), newshape=(self.J, -1))

        lasso_fitter = self._get_amplitude_fitter()
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * self.y)
        return lasso_fitter, lasso_fitter.coef_.flatten()

    def etak(self, x: np.ndarray) -> float:
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] - dist[:, np.newaxis] / c)
                  / 4 / np.pi / np.sqrt(dist[:, np.newaxis] ** 2 + self.eps)).flatten()

        return -np.abs(np.sum(self.res * gammaj))

    def _get_normalized_fun(self):
        return "3-point", "3-point"


class FrequencyDomainSFW(SFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], N: int,
                 mic_pos: np.ndarray, fs: float, lam: float = 1e-2, freq_range=None, fc=None, deletion_tol=5e-2):
        """
        Args:
            -y (ndarray or tuple(ndarray, ndarray)) : measurements (shape (J,) = (N*M,)) or tuple (a,x) containing the
        amplitudes and positions of the measure that has to be reconstructed. In that last case the ideal observations
        are computed.
            -mic_pos (ndarray): positions of the microphones (shape (M,d))
            -fs (float): sampling frequency
            -N (int): number of time samples
            -lam (float): penalization parameter of the BLASSO.
            -fc  (float): cutting frequency for the time response.
            - obs_freq (tuple): tuple of two floats delimiting the range of observed frequencies. Only the frequencies
            in the range [obs_freq[0], obs_freq[1]] will be considered.
        """

        # creating a time sfw object to compute the residual on the RIR (used for grid initialization)
        self.time_sfw = TimeDomainSFW(y=y, mic_pos=mic_pos, fs=fs, N=N, fc=fc)
        self.mic_pos, self.M = mic_pos, len(mic_pos)

        self.d = mic_pos.shape[1]
        assert 1 < self.d < 4, "Invalid dimension d"

        self.obs_freq = (0., fs) if freq_range is None else freq_range
        self.fs = fs

        self._update_rir()

        # getting attributes and methods from parent class
        super().__init__(y=self.y, fs=fs, lam=lam, N=N, fc=fc, deletion_tol=deletion_tol)

    def _update_freq(self):
        """
        Update the frequency array depending on the length of the time response stored in time_sfw.
        """
        # array of observed frequencies, normalized by 2pi to reduce computations in Fourier transform
        self.freq_array = np.fft.rfftfreq(self.time_sfw.N, d=1. / self.fs)
        self.freq_mask = (self.freq_array >= self.obs_freq[0]) & (self.freq_array < self.obs_freq[1])
        self.freq_array = self.freq_array[self.freq_mask] * 2 * np.pi
        self.Nfreq = len(self.freq_array)
        self.J = self.M * self.Nfreq

    def _update_rir(self):
        """
        Update the frequency response depending on the current length of the time response stored in time_sfw.
        """
        self._update_freq()

        # compute the FFT of the rir
        self.y = np.fft.rfft(self.time_sfw.y.reshape(self.M, self.time_sfw.N),
                             axis=-1)[:, self.freq_mask].flatten()

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum(M, K, N_freq, axis=1), no 1/fs factor in order to account for the DFT approximation
        return np.sum(np.exp(-1j * self.freq_array[np.newaxis, np.newaxis, :] * dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / dist[:, :, np.newaxis]
                      * a[np.newaxis, :, np.newaxis], axis=1).flatten()

    def etak(self, x: np.ndarray) -> float:
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, Nfreq) to (M*Nfreq,)
        gammaj = (np.exp(-1j * self.freq_array[np.newaxis, :] * dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis]).flatten()

        return -np.sum(np.real(self.res * np.conj(gammaj)))

    def compute_residue(self):
        gk = self.gamma(self.ak, self.xk)
        self.res = self.y - gk
        self.res_norm = np.mean(np.linalg.norm(self.res.reshape(self.M, -1), axis=1), axis=0)  # mean residual norm

        return self.res

    def _algorithm_start_callback(self, **args):
        """Cut the RIR in time"""
        self.time_sfw._algorithm_start_callback(**args)
        self._update_rir()
        self.compute_residue()

    def _iteration_start_callback(self, verbose=False):
        self.time_sfw.ak, self.time_sfw.xk = self.ak[:], self.xk[:]

        # set the views to be immutable (controls that the time_sfw object does not modify the spikes)
        self.time_sfw.ak.setflags(write=False)
        self.time_sfw.xk.setflags(write=False)

        self.time_sfw.compute_residue()

        # extend the time RIR if necessary
        is_extended = self.time_sfw._iteration_start_callback(verbose)

        if is_extended:  # in that case, update the frequency RIR
            self._update_rir()
            self.compute_residue()

    def _extend_rir(self, reason, verbose=False):
        can_extend = self.time_sfw._extend_rir(reason=reason, verbose=verbose)
        self._update_rir()
        self.compute_residue()

        return can_extend

    def _on_stop(self, reason=None, verbose=False):
        """Stop if the time RIR cannot be extended further"""
        can_extend = self._extend_rir(reason=reason, verbose=verbose)

        return not can_extend

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # shape (M, N_freq, K) -> (J, K)
        gamma_mat_cpx = np.reshape(np.exp(-1j * self.freq_array[np.newaxis, :, np.newaxis] * dist[:, np.newaxis, :] / c)
                                   / 4 / np.pi / dist[:, np.newaxis, :],
                                   newshape=(self.J, -1))

        gamma_mat = np.concatenate([np.real(gamma_mat_cpx),
                                    np.imag(gamma_mat_cpx)], axis=0)
        lasso_fitter = self._get_amplitude_fitter()
        target = np.concatenate([np.real(self.y), np.imag(self.y)]).reshape(-1, 1)
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * target)

        return lasso_fitter, lasso_fitter.coef_.flatten()

    def _grid_initialization_function(self, search_method, grid, nmic, verbose, **params):
        return self.time_sfw._grid_initialization_function(search_method, grid, nmic, verbose, **params)

    def _get_normalized_fun(self):
        normalized_eta_jac = self._jac_etak
        slide_jac = self._jac_slide_obj
        return normalized_eta_jac, slide_jac

    def _obj_slide(self, var, y, n_spikes):
        """
        Objective function for the sliding step, adapted for complex values
        """

        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        return 0.5 * np.sum(np.abs(self.gamma(a, x) - y) ** 2) + self.lam * np.sum(np.abs(a))

    def _jac_slide_obj(self, var, y, n_spikes):
        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        diff = x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]  # difference, shape (M, K, 3)
        # distances from the diracs contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        jac = np.zeros(n_spikes * 4)

        # shape (M, K, Nfreq)
        gamma_tens = (np.exp(-1j * self.freq_array[np.newaxis, np.newaxis, :] * dist[:, :, np.newaxis]/c)
                      / dist[:, :, np.newaxis] / 4 / np.pi)

        # gamma(a,x) - y = -res, shape (M,Nfreq)
        residue_c = np.conj(np.einsum("mkn,k->mn", gamma_tens, a, optimize='greedy') - y.reshape(self.M, self.Nfreq))

        # derivatives in a
        jac[:n_spikes] = self.lam + np.sum(np.real(residue_c[:, np.newaxis, :]*gamma_tens), axis=(0, 2))

        # real part in the gradient, shape (M, K, Nfreq)
        real_part = -np.real((1/dist[:, :, np.newaxis] + 1j*self.freq_array[np.newaxis, np.newaxis, :] / c)*gamma_tens
                             * residue_c[:, np.newaxis, :])

        # derivatives in x, shape (M, K, 3, N_freq) -> (K, 3) flattened
        jac[n_spikes:] = np.einsum("mkn,k,mkin->ki", real_part, a,
                                   diff[:, :, :, np.newaxis]  # (M, K, 3) -> (M, K, 3, N_freq)
                                   / dist[:, :, np.newaxis, np.newaxis], optimize='greedy').reshape(-1)

        # sum(M, K, N_freq, axis=1)
        return jac

    def _jac_etak(self, x):
        diff = x[np.newaxis, :] - self.mic_pos[:, :]  # difference, shape (M, 3)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        # conjugate gradient of gamma_m without the (r-rm) factor, shape (M,Nfreq)
        exp_conj = ((1/dist[:, np.newaxis] - 1j*self.freq_array[np.newaxis, :]/c)
                    * (np.exp(1j*self.freq_array[np.newaxis, :]*dist[:, np.newaxis]/c)
                    / 4 / np.pi / dist[:, np.newaxis]**2) * self.res.reshape([self.M, self.Nfreq]))

        return np.sum(np.real(exp_conj[:, :, np.newaxis])*diff[:, np.newaxis, :], axis=(0, 1)).reshape(-1)

    def get_cut_ind(self):
        return self.time_sfw.cut_ind

    def update_mic_pos(self, new_pos):
        self.mic_pos = new_pos
        self.time_sfw.update_mic_pos(new_pos)


class FrequencyDomainSFWNorm1(FrequencyDomainSFW):
    """Adds a normalization factor 1/(sqrt(sum(1/dist(x, xm)**2)."""

    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], N: int,
                 mic_pos: np.ndarray, fs: float, lam: float = 1e-2, freq_range=None, fc=None, deletion_tol=5e-2):
        y = TimeDomainSFW(y, N=N, mic_pos=mic_pos, fs=fs, lam=lam, fc=fc, deletion_tol=deletion_tol).y
        super().__init__(y=y, N=N, mic_pos=mic_pos, fs=fs, lam=lam,
                         fc=fc, deletion_tol=deletion_tol, freq_range=freq_range)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum(M, K, N_freq, axis=1)
        return np.sum(np.exp(-1j * self.freq_array[np.newaxis, np.newaxis, :] * dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / dist[:, :, np.newaxis]
                      / np.linalg.norm(1 / dist, axis=0)[np.newaxis, :, np.newaxis]
                      * a[np.newaxis, :, np.newaxis], axis=1).flatten()

    def etak(self, x: np.ndarray) -> float:
        """Normalization of etak by 1/norm_2([dist(x, xm)]_m)"""
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (np.exp(-1j * self.freq_array[np.newaxis, :] * dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis] / np.linalg.norm(1 / dist)).flatten()

        return -np.abs(np.sum(self.res * np.conj(gammaj)))

    def _get_normalized_fun(self):
        normalized_eta_jac = "3-point"
        slide_jac = None
        return normalized_eta_jac, slide_jac

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # shape (M, N_freq, K) -> (J, K)
        gamma_mat_cpx = np.reshape(np.exp(
            -1j * self.freq_array[np.newaxis, :, np.newaxis] * dist[:, np.newaxis, :] / c)
                                   / np.linalg.norm(1 / dist, axis=0)[np.newaxis, np.newaxis, :]
                                   / 4 / np.pi / dist[:, np.newaxis, :],
                                   newshape=(self.J, -1))

        gamma_mat = np.concatenate([np.real(gamma_mat_cpx),
                                    np.imag(gamma_mat_cpx)], axis=0)
        lasso_fitter = self._get_amplitude_fitter()
        target = np.concatenate([np.real(self.y), np.imag(self.y)]).reshape(-1, 1)
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * target)

        return lasso_fitter, lasso_fitter.coef_.flatten()


class DeconvolutionSFW(SFW):
    """Blind spikes deconvolution. Does not work correctly with current initialization, and time segmentation is not
    supported."""
    def __init__(self, source_pos, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], N: int,
                 mic_pos: np.ndarray, fs: float, lam: float = 1e-2, freq_range=None, fc=None, deletion_tol=5e-2):
        self.mic_pos, self.M = mic_pos, len(mic_pos)

        self.extract_ind = np.triu_indices(self.M, 1)
        self.d = mic_pos.shape[1]
        self.source_pos = source_pos.copy()
        self.freq_sfw = FrequencyDomainSFW(y=y, N=N, mic_pos=mic_pos, fs=fs, fc=fc, freq_range=freq_range,
                                           deletion_tol=deletion_tol)
        self.Nfreq = self.freq_sfw.Nfreq

        self.y = -self.gamma(np.array([1.]), self.source_pos.reshape([1, 3]))

        super().__init__(y=self.y, fs=fs, lam=lam, N=self.freq_sfw.Nfreq, fc=fc, deletion_tol=deletion_tol)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        freq_response = self.freq_sfw.gamma(a, x).reshape([self.M, self.Nfreq])
        # matrix containing the pairwise Hadamard products, shape (M, M, Nfreq)
        mat = self.freq_sfw.y.reshape([self.M, self.Nfreq])[:, np.newaxis, :]*freq_response[np.newaxis, :, :]
        mat = mat[self.extract_ind] - np.transpose(mat, (1, 0, 2))[self.extract_ind]  # extract the upper coefficients

        return mat.reshape(-1)

    def etak(self, x: np.ndarray) -> float:
        return - np.sum(np.real(self.gamma(np.array([1.]), x.reshape([1, 3]))
                                * np.conj(self.res)))

    def _LASSO_step(self):
        gamma_mat_cpx = np.array(
            [self.gamma(np.ones(1), self.xkp[i, :].reshape([-1, self.d])) for i in range(self.nk)]).T
        gamma_mat = np.concatenate([np.real(gamma_mat_cpx),
                                    np.imag(gamma_mat_cpx)], axis=0)
        lasso_fitter = Lasso(alpha=self.lam, positive=True)
        target_cpx = self.y
        target = np.concatenate([np.real(target_cpx), np.imag(target_cpx)]).reshape(-1, 1)
        scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * target)

        return lasso_fitter, lasso_fitter.coef_.flatten()

    def _get_normalized_fun(self):
        return None, None

    def _obj_slide(self, var, y, n_spikes):
        """
        Objective function for the sliding step, adapted for complex values
        """
        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        return 0.5 * np.sum(np.abs(self.gamma(a, x) - y) ** 2) + self.lam * np.sum(np.abs(a))

    def _algorithm_start_callback(self, **args):
        self.freq_sfw._algorithm_start_callback(**args)

    def _iteration_start_callback(self, verbose=False):
        # append the source position to get the correct initialization, use copies of ak, xk
        self.freq_sfw.ak = np.concatenate([self.ak[:], np.array([1.])])
        self.freq_sfw.xk = np.concatenate([self.xk[:], self.source_pos])

        self.freq_sfw._iteration_start_callback(verbose=verbose)

    def _grid_initialization_function(self, search_method, grid, nmic, verbose, **params):
        return self.freq_sfw._grid_initialization_function(search_method, grid, nmic, verbose, **params)

    def get_cut_ind(self):
        return self.freq_sfw.get_cut_ind()

    def update_mic_pos(self, new_pos):
        self.mic_pos = new_pos
        self.freq_sfw.update_mic_pos(new_pos)


class Decimation:
    def __init__(self, original_sfw, nb_decimation, discard_factor=0.5):
        self.original_sfw, self.nb_decimation = original_sfw, nb_decimation
        self.discard_factor = discard_factor
        self.sub_sfw = []
        previous_sf, y = original_sfw, original_sfw.y
        for i in range(nb_decimation):  # create decimated sfw objects
            new_fs = previous_sf.fs / 2.

            # the signal y is not decimated and will not be used (the residual will be decimated later)
            decimated_sf = TimeDomainSFW(y=y, N=self.original_sfw.N, mic_pos=original_sfw.mic_pos, fs=new_fs,
                                         lam=original_sfw.lam, deletion_tol=original_sfw.deletion_tol, fc=new_fs)
            self.sub_sfw.append(decimated_sf)
            previous_sf = decimated_sf

    def decimate_residue(self):
        for i in range(self.nb_decimation):
            self.sub_sfw[i].res = decimate(self.original_sfw.res.reshape([self.original_sfw.M, -1]),
                                           q=2**(i+1), axis=-1,ftype='fir')
            self.sub_sfw[i].N = self.sub_sfw[i].res.shape[1]  # update to decimated length
            self.sub_sfw[i].res = self.sub_sfw[i].res.reshape(-1)
            self.sub_sfw[i].NN = compute_time_sample(self.sub_sfw[i].N, self.sub_sfw[i].fs)

    def grid_search(self, grid, min_norm,  rough_gtol, rough_maxiter, verbose=False):
        p = multiprocessing.Pool(self.original_sfw.ncores)
        nkept = len(grid)
        # perform original search at lowest sampling frequency first
        for i in range(self.nb_decimation-1, -1, -1):
            # perform a low precision search at each grid point
            self.sub_sfw[i].opt_options["gtol"] = rough_gtol
            self.sub_sfw[i].opt_options["maxiter"] = rough_maxiter

            gr_opt = np.array(p.map(self.sub_sfw[i]._optigrid, grid))

            # searching for the best result over the grid at a minimal distance from the origin
            norms = np.linalg.norm(gr_opt[:, :self.original_sfw.d], axis=1)

            kept_points = gr_opt[:, :self.original_sfw.d+1][norms > min_norm]

            if len(kept_points) == 0:  # it means the spikes found are all inside the ball of radius min_norm
                if verbose:
                    print("Cannot find a spike outside the minimal norm ball")
                if i > 0:  # actual sampling frequency not found, pass
                    continue
                else:
                    return False, None
            else:
                nkept = np.maximum(int(self.discard_factor*nkept), 1)  # keep at least one grid point
                best_ind = np.argsort(kept_points[:, self.original_sfw.d])[:nkept]

                if i == 0:
                    grid = kept_points[best_ind[0], :self.original_sfw.d]
                else:
                    grid = kept_points[best_ind, :self.original_sfw.d]

        p.close()

        return True, grid
