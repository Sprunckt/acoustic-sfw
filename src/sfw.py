import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from sklearn.linear_model import Lasso
from src.simulation.utils import (disp_measure, c, cut_vec_rir, create_grid_spherical)
import multiprocessing
import time
from abc import ABC, abstractmethod
import os

stop_tol = 1e-6
deletion_tol = 0.05
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
    energy = a**2
    sliding_mean = np.convolve(energy, np.full(win_length, 1./win_length), 'same')
    ind = np.argmax(sliding_mean)
    return ind, sliding_mean[ind]


class SFW(ABC):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], fs: float, lam: float = 1e-2):
        self.fs, self.lam = fs, lam
        self.d = 3

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

        self.opt_options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08,
                            'maxiter': None, 'disp': False, 'return_all': False, 'finite_diff_rel_step': None}
        self.eta, self.eta_jac = None, None
        self.timer = None

        # spike history for freezing spikes
        self.old_ak, self.old_xk = np.zeros(0), np.zeros((0, self.d))
        self.spike_hist_counter = np.zeros(0, dtype=int)
        self.active_spikes, self.n_active = np.zeros(0, dtype=bool), 0
        self.freeze_step = 0
        self.y_freeze = self.y

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

    def _obj_lasso(self, a):
        """
        Objective function for the optimization on spike amplitudes.
        """

        return 0.5 * np.sum((self.gamma(a, self.xkp) - self.y) ** 2) + self.lam * np.sum(np.abs(a))

    def _obj_slide(self, var, y, n_spikes):
        """
        Objective function for the sliding step (optimization on locations and amplitude).
        """
        a, x = var[:n_spikes], var[n_spikes:].reshape(-1, self.d)
        return 0.5 * np.sum((self.gamma(a, x) - y) ** 2) + self.lam * np.sum(np.abs(a))

    def _optigrid(self, x):
        return minimize(self.eta, x, jac=self.eta_jac, method="BFGS", options=self.opt_options)

    def _stop(self, verbose=True):
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
    def _grid_initialization_function(self, parameters, verbose):
        pass

    @abstractmethod
    def _get_normalized_fun(self, normalization):
        """Return a tuple of callables containing the normalized eta function and the corresponding jacobians for the
        eta optimization and sliding steps. normalization=0 should return the default functions."""
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

    def _on_stop(self, verbose=False):
        """Method called when a stopping criterion is met, should return True if the algorithm has to stop on the spot,
        False if it is allowed to go on through (for example if the RIR is not extended to its full length)."""
        return True

    def reconstruct(self, grid=None, niter=7, min_norm=-np.inf, max_norm=np.inf, max_ampl=np.inf, normalization=0,
                    search_method="rough", spike_merging=False, spherical_search=0,
                    use_hard_stop=True, verbose=True, early_stopping=False,
                    plot=False, algo_start_cb=None, it_start_cb=None,
                    freeze_step=0, resliding_step=0) -> (np.ndarray, np.ndarray):
        """
        Apply the SFW algorithm to reconstruct the measure based on the measurements self.y.

        Args:
            -grid (array or float) : parameter used to specify the grid search. It is passed to the grid initialization
            function. For the time domain case : if spherical_grid == 0, should be the array of the grid.
        If spherical_grid == 1, if it is an array it is assumed to be a spherical grid of radius 1 that is scaled
        accordingly during the algorithm. If it is a float, it is assumed to be the angular step for a spherical grid
        of radius 1, that is scaled depending on the radius r by 1/log(r) to keep a good density of nodes.
            -niter (int) : maximal number of iterations
            -max_ampl (float) : upper bound on spikes amplitudes
            -normalization (int) : the normalization to add to gamma for he spike localization step. 0 means no
        normalization, 1 adds a factor 1/(sqrt(sum(1/dist(x, xm)**2))
            -min_norm (float) : minimal norm allowed for the position found at the end of the grid search
            -max_norm (float) : used as bounds for the coordinates of the spike locations in each direction
            -use_hard_stop (bool) : if True, add max|etak| <= 1 as a stopping condition
            -early_stopping (bool) : if True, stop at the end of an iteration if the last spike found has zero amplitude
            -search_method (str) : grid search methods for the spike position search. If "rough" : perform a coarse
        optimization on each point of the grid before refining on the best position. If "full" : perform a fine
        optimization on each grid point (costly). If "naive" : find the best value on the grid and use it as
        initialization (fastest but less precize).
            -spherical_search (int) : if equal to 1 : assume that the given grid is spherical. The maximum energy spike
        of the residual is used to find the distance from a microphone to an image source, and applying a grid search
        on the corresponding sphere. The grid is parametrized by the grid argument.
            - algo_start_cb (dict) : dictionary containing the arguments passed to the algorithm start callback
            it_start_cb (dict) : dictionary containing the arguments passed to the iteration start callback
            -freeze_step (int) : if strictly positive : check each spike every 'freeze_step' iterations. If the spike
        has not moved sufficiently since the last check, the spike is frozen and is not allowed to slide in the next
        iterations. Speeds up the execution when the number of iterations becomes important, but lessens the accuracy.
            -resliding_step (int) : if strictly positive : apply a periodic sliding step on all the spikes (including
        the frozen ones).
         Return:
            (ak, xk) where :
            -ak is a flat array of shape (K,) containing the amplitudes of the recovered measure
            -xk is a (K, d) shaped array containing the locations of the K spikes composing the measure
        """
        self.timer = time.time()
        self.eta, self.eta_jac, slide_jac = self._get_normalized_fun(normalization)

        xmin, xmax = -max_norm, max_norm
        amin, amax = 0, max_ampl
        self.nk = 0
        self.freeze_step = freeze_step

        ncores = len(os.sched_getaffinity(0))
        if verbose:
            print("Executing on {} cores".format(ncores))
            
        reslide_counter = 0

        search_grid = grid
        assert search_grid is not None, "a grid must be specified for the initial grid search"

        if algo_start_cb is None:
            algo_start_cb = {}

        if it_start_cb is None:
            it_start_cb = {}

        it_start_cb["verbose"] = verbose
        algo_start_cb["verbose"] = verbose
        self._algorithm_start_callback(**algo_start_cb)
 
        for i in range(niter):
            self.it += 1
            if verbose:
                print("iteration {}, residual norm : {}".format(self.it, self.res_norm))

            self._iteration_start_callback(**it_start_cb)
            # find argmax etak to get the new spike position (step 3)
            if verbose:
                print("Optimizing for spike position -----")
            # grid search : one optimization per grid point (the grid has the shape (Ngrid, d))
            tstart = time.time()

            if spherical_search == 1:  # take argmax on the complete rir and search on the corresponding sphere
                search_grid = self._grid_initialization_function(grid, verbose=verbose)

            if verbose:
                print("Starting a grid search to minimize etak, method : ", search_method)

            rough_search = search_method == "rough"
            if rough_search or search_method == "full":

                if search_method == "rough":  # perform a low precision search
                    self.opt_options["gtol"] = np.minimum(1e-1, 1e-6/self.lam)

                # spreading the loop over multiple processors
                p = multiprocessing.Pool(ncores)
                gr_opt = p.map(self._optigrid, search_grid)
                p.close()

                # searching for the best result over the grid
                curr_min, curr_opti_res = np.inf, None
                for el in gr_opt:
                    if el.fun < curr_min and np.linalg.norm(el.x) > min_norm:
                        curr_min = el.fun
                        curr_opti_res = el

                if curr_opti_res is None:  # it means the spikes found are all inside the ball of radius min_norm
                    if verbose:
                        print("Cannot find a spike outside the minimal norm ball")
                    if self._on_stop():
                        return self._stop(verbose=verbose)
                    else:
                        continue

                if rough_search:  # perform a finer optimization using the position found as initialization
                    nit = curr_opti_res.nit
                    self.opt_options["gtol"] = 1e-6
                    opti_res = self._optigrid(curr_opti_res.x)
                    nit += opti_res.nit
                else:
                    opti_res = curr_opti_res
                    nit = opti_res.nit

                del gr_opt

            else:
                mapping = np.apply_along_axis(self.eta, 1, search_grid)
                ind_max = np.argmin(mapping)
                self.opt_options["gtol"] = 1e-6
                opti_res = self._optigrid(search_grid[ind_max])
                nit = opti_res.nit

            etaval = np.abs(opti_res.fun)
            x_new = opti_res.x.reshape([1, self.d])

            if verbose:
                print("exec time for grid optimization : ", time.time() - tstart)

            if not opti_res.success:
                print("etak optimization failed, reason : {}".format(opti_res.message))

            if verbose:
                print("Optimization converged in {} iterations".format(nit))
                print("New position : {} \n eta value : {}".format(x_new, etaval))

            if use_hard_stop and etaval <= 1 + stop_tol:
                if verbose:
                    print("Stopping criterion met : etak(x_new)={} < 1".format(etaval))
                if self._on_stop():
                    return self._stop(verbose=verbose)
                else:
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
                print("LASSO solved in {} iterations, exec time : {} s".format(lasso_fitter.n_iter_,
                                                                               time.time() - tstart))
                print("New amplitudes : {} \n".format(ak_new))

            # sliding step : optimize to adjust both the positions and amplitudes

            # bounds for the amplitudes and positions
            bounds = [(amin, amax)] * self.n_active + [(xmin, xmax)] * self.n_active * self.d

            tmp_active = np.append(self.active_spikes, [True])
            ini = np.concatenate([ak_new[tmp_active],
                                  self.xkp[tmp_active].flatten()])

            ini_val = self._obj_slide(ini, y=self.y_freeze, n_spikes=self.n_active)

            if verbose:
                print("Sliding step --------")
                print("initial value : {}".format(ini_val))

            tstart = time.time()

            opti_res = minimize_parallel(self._obj_slide, ini, jac=slide_jac, bounds=bounds,
                                         args=(self.y_freeze, self.n_active), parallel={'max_workers': ncores})
            mk, nit_slide, val_fin = opti_res.x, opti_res.nit, opti_res.fun
            decreased_energy = val_fin < ini_val

            # use the new measure if the sliding step decreased the energy, else keep the old values
            if decreased_energy:
                ak_new[tmp_active], self.xkp[tmp_active] = mk[:self.n_active], mk[self.n_active:].reshape([-1, self.d])
            else:
                if verbose:
                    print("Energy increased, ignoring this step")

            self.ak, self.xk = ak_new, self.xkp

            if not opti_res.success:
                print("Last optimization failed, reason : {}".format(opti_res.message))

            if verbose:
                print("Optimization converged in {} iterations, exec time : {} s".format(nit_slide,
                                                                                         time.time() - tstart))
                print("Objective value : {}".format(val_fin))

            if resliding_step:
                reslide_counter += 1
                if reslide_counter >= resliding_step:
                    bounds = [(amin, amax)] * self.nk + [(xmin, xmax)] * self.nk * self.d
                    ini = np.concatenate([self.ak, self.xk.flatten()])
                    ini_val = self._obj_slide(ini, y=self.y, n_spikes=self.nk)
                    if verbose:
                        print("Periodic sliding step on every spike ---")
                    tstart = time.time()
                    opti_res = minimize_parallel(self._obj_slide, ini, jac=slide_jac,
                                                 bounds=bounds, args=(self.y, self.nk))
                    mk, nit_slide, val_fin = opti_res.x, opti_res.nit, opti_res.fun
                    tend = time.time()
                    if verbose:
                        print("Initial/final values : {} {} \n".format(ini_val, val_fin))
                        print("Optimization converged in {} iterations, exec time : {} s".format(nit_slide,
                                                                                                 tend - tstart))

                    if not opti_res.success:
                        print("Last optimization failed, reason : {}".format(opti_res.message))

                    if val_fin < ini_val:  # check if the energy has decreased
                        self.ak, self.xk = mk[:self.nk], mk[self.nk:].reshape([-1, self.d])
                        reslide_counter = 0
                    else:
                        if verbose:
                            print("Energy increased, sliding not applied - retrying next iteration")

            # deleting null amplitude spikes
            ind_null = np.asarray(np.abs(self.ak) < deletion_tol)
            self.ak = self.ak[~ind_null]
            self.xk = self.xk[~ind_null, :]
            self.nk = len(self.ak)

            if freeze_step and self.nk > 0:  # update the frozen spikes history
                self._append_history(self.ak[self.nk-1:], self.xk[self.nk-1].reshape(1, 3))
                self._update_history(ind_null)  # if the new spike is null it is instantly deleted. todo: clean this
                if verbose:
                    print("active spikes : \n", np.where(self.active_spikes)[0])
            else:  # all the spikes are active
                self.active_spikes = np.ones(self.nk, dtype=bool)
                self.n_active = self.nk

            if self.nk == 0:
                if self._on_stop():
                    print("Error : all spikes are null, stopping")
                    return self._stop(verbose=verbose)
                else:
                    continue
            # last spike is null and minor changes from the previous iteration at the sliding step
            elif (early_stopping and (ind_null.sum() == 1 and ind_null[-1])
                  and (nit_slide == 1 or not decreased_energy)):
                if self._on_stop():
                    print("Last spike has null amplitude, stopping")
                    return self._stop(verbose=verbose)
                else:
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

            # update the residue
            self.compute_residue()

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
                 fs: float, N: int, lam: float = 1e-2):
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
        self.N = N
        self.NN = compute_time_sample(self.N, fs)  # discretized time interval

        self.mic_pos, self.M = mic_pos, len(mic_pos)

        self.d = mic_pos.shape[1]
        assert 1 < self.d < 4, "Invalid dimension d"

        self.J = self.M * self.N

        super().__init__(y=y, fs=fs, lam=lam)  # getting attributes and methods from parent class
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
        self.res_norm = self.cumulated_energy[-1]  # norm of the first RIR
        self.old_norm = self.res_norm
        self.current_ind = 0  # index of the last cutting threshold used
        assert self.y.size == self.J, "invalid measurements length, {} != {}".format(self.y.size, self.J)

    def _algorithm_start_callback(self, verbose=False, n_cut=0, int_start=None,
                                  swap_frequency=10, swap_factor=0.5):
        """Parametrize the segmentation of the time interval.
        Args:-n_cut (int) : number of cutting thresholds (number of subintervals -1). Set to 0 to use the full RIR.
        5 is a good default value.
        -int_start (float) : starting value for the choice of the threshold (maximum value is the norm of the RIR),
        default : norm/3
        -swap_frequency (int) : maximum number of full iterations before extending to the next threshold"""
        int_end = self.cumulated_energy[-1]  # vector norm of the first microphone RIR
        if int_start is None:
            int_start = int_end / 3.

        cut_values = np.linspace(int_start, int_end, n_cut, endpoint=False)

        if n_cut > 0:
            overshoot = int(self.fs * self.antenna_diameter / c)  # worst minimal time to go through the antenna
            cut_ind = [np.argmax(self.cumulated_energy > cut_values[0]) + overshoot]

            for i in range(1, n_cut):
                # find the smallest time sample for which an energy threshold is reached (overshoot by a few samples)
                new_ind = np.argmax(self.cumulated_energy > cut_values[i]) + overshoot
                if new_ind - cut_ind[-1] > 50:  # only add the next index if there is a minimal time separation
                    cut_ind.append(new_ind)

            cut_ind.append(self.global_N - 1)  # last index of the complete first microphone RIR
            self.cut_ind = np.array(cut_ind)
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

    def _on_stop(self, verbose=False):
        """Should return False if the RIR can be extended (meaning the algorithm is allowed to pull through), True
        otherwise."""
        return not self._extend_rir(reason="stopping criterion met", verbose=verbose)

    def _iteration_start_callback(self, verbose=False):
        self.swap_counter += 1
        curr_norm, thresh = self.res_norm, self.swap_factor*self.old_norm
        threshold_reached = curr_norm < thresh  # norm criterion
        max_iter_reached = self.swap_counter > self.swap_frequency
        if threshold_reached:
            self._extend_rir("norm criterion reached: {} < {}".format(curr_norm, thresh), verbose=verbose)
        elif max_iter_reached:
            self._extend_rir("max iteration reached", verbose=verbose)

    def sinc_filt(self, t):
        """
        The filter applied for each measurement.
        """

        return np.sinc(t * self.fs)

    def sinc_der(self, t):
        """
        Derivate of the filter
        """
        w = np.pi * self.fs
        return (t * np.cos(w * t) - np.sin(w * t) / w) / (t ** 2 + np.finfo(float).eps)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum( M, K, N, axis=1)
        return np.sum(self.sinc_filt(self.NN[np.newaxis, :] - dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / dist[:, :, np.newaxis]
                      * a[np.newaxis, :, np.newaxis], axis=1).flatten()

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

        lasso_fitter = Lasso(alpha=self.lam, positive=True)
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

        return -np.abs(np.sum(self.res * gammaj)) / self.lam

    def etak_norm1(self, x: np.ndarray) -> float:
        """Normalization of etak by 1/norm_2([dist(x, xm)]_m)"""
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] - dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis] / np.linalg.norm(1 / dist)).flatten()

        return -np.abs(np.sum(self.res * gammaj)) / self.lam

    def _jac_etak(self, x):
        diff = x[np.newaxis, :] - self.mic_pos[:, :]  # difference, shape (M, 3)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        int_term = self.NN[np.newaxis, :] - dist[:, np.newaxis] / c  # shape (M, N)

        # sum shape (M,  N) into shape (M,), derivate without the xk_i - xm_i factor
        tens = np.sum(((- self.sinc_filt(int_term) / dist[:, np.newaxis] - self.sinc_der(int_term) / c)
                       / dist[:, np.newaxis] ** 2 / 4 / np.pi) * self.res.reshape(self.M, self.N), axis=1)

        # shape (M,3) into (M,)
        jac = (np.sum(tens[:, np.newaxis] * diff, axis=0).flatten())

        return -jac * np.sign(self.etak(x)) / self.lam

    def _jac_etak_norm1(self, x):
        diff = x[np.newaxis, :] - self.mic_pos[:, :]  # difference, shape (M, 3)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        int_term = self.NN[np.newaxis, :] - dist[:, np.newaxis] / c  # shape (M, N)

        norm2 = np.sum(1 / dist ** 2)
        norm = np.sqrt(norm2)  # float
        norm_der = - np.sum(diff / dist[:, np.newaxis] ** 4, axis=0) / norm  # shape (3,)

        # sum shape (M,  N) into shape (M,), derivative  of gammaj * N without the xk_i - xm_i factor
        der_gam = np.sum(((- self.sinc_filt(int_term) / dist[:, np.newaxis] - self.sinc_der(int_term) / c)
                          / dist[:, np.newaxis] ** 2 / 4 / np.pi / norm) * self.res.reshape(self.M, self.N), axis=1)
        der_gam = (np.sum(der_gam[:, np.newaxis] * diff, axis=0).flatten())  # shape (3,)

        # derivative of N * gammaj without the xk_i - xm_i factor
        der_N = (np.sum((- self.sinc_filt(int_term) / dist[:, np.newaxis] / 4 / np.pi).flatten() * self.res)
                 * norm_der / norm2)  # shape (3,)

        return -np.sign(self.etak_norm1(x)) * (der_gam + der_N) / self.lam

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
        residue = (np.sum(gamma_tens * a[np.newaxis, :, np.newaxis], axis=1).flatten() - y)
        # derivates in ak : multiply the residue by gammaj(x_k) and sum on j (meaning m and N)
        jac[:n_spikes] = (np.sum(residue.reshape(self.M, self.N)[:, np.newaxis, :] * gamma_tens, axis=(0, 2))
                          + self.lam * np.sign(a))

        # shape (M, K, N), derivate without the xk_i - xm_i factor
        gamma_tens = ((- gamma_tens - self.sinc_der(int_term) / 4 / np.pi / c)
                      / dist[:, :, np.newaxis] ** 2 * residue.reshape(self.M, self.N)[:, np.newaxis, :])

        # original shape (M,K,3,N)
        jac[n_spikes:] = (np.repeat(a, 3) * np.sum(gamma_tens[:, :, np.newaxis, :] * diff[:, :, :, np.newaxis],
                                                   axis=(0, 3)).flatten())
        return jac

    def _grid_initialization_function(self, parameter, verbose, **params):
        curr_max, n_max, m_max = -1, 0, 0
        for m in range(self.M):
            ind_tmp, max_tmp = sliding_window_norm(self.res[m*self.N: (m+1)*self.N], 3)
            if max_tmp > curr_max:
                curr_max = max_tmp
                n_max, m_max = ind_tmp, m

        r = n_max * c / self.fs
        if type(parameter) == np.ndarray:  # use the generated grid and scale it to the correct radius
            grid = r * parameter
        else:  # no pre-generated grid
            if r > 1:  # increase the number of nodes on the grid if the radius exceeds 1
                dtheta = parameter / np.log(r)
            else:
                dtheta = parameter
            grid, sph_grid, n_sph = create_grid_spherical(r, r, 1., dtheta=dtheta, dphi=dtheta)
        search_grid = grid + self.mic_pos[m_max][np.newaxis, :]
        if verbose:
            print("searching around mic {} at a radius {}, {} grid points".format(m_max, r, len(grid)))
        return search_grid

    def _get_normalized_fun(self, normalization):
        normalized_eta = [self.etak, self.etak_norm1][normalization]
        normalized_eta_jac = ["3-point", "3-point"][normalization]  # eta jacobian is broken
        slide_jac = self._jac_slide_obj
        return normalized_eta, normalized_eta_jac, slide_jac


class EpsilonTimeDomainSFW(TimeDomainSFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], mic_pos: np.ndarray,
                 fs: float, N: int, lam: float = 1e-2, eps=1e-3):
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

        super().__init__(y=y, fs=fs, lam=lam, N=N, mic_pos=mic_pos)  # getting attributes and methods from parent class
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

        lasso_fitter = Lasso(alpha=self.lam, positive=True)
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

        return -np.abs(np.sum(self.res * gammaj)) / self.lam

    def _get_normalized_fun(self, normalization):
        return self.etak, "3-point", "3-point"


class FrequencyDomainSFW(SFW):
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], N: int,
                 mic_pos: np.ndarray, fs: float, lam: float = 1e-2):
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

        # creating a time sfw object to compute the residual on the RIR (used for grid initialization)
        self.time_sfw = TimeDomainSFW(y=y, mic_pos=mic_pos, fs=fs, N=N)
        self.mic_pos, self.M = mic_pos, len(mic_pos)

        self.d = mic_pos.shape[1]
        assert 1 < self.d < 4, "Invalid dimension d"

        # array of observed frequencies
        self.freq_array = np.fft.rfftfreq(N, d=1. / fs) * 2 * np.pi

        self.N, self.N_freq = N, len(self.freq_array)

        self.J = self.M * self.N_freq

        # compute the FFT of the rir, divide by the normalization constant
        y_freq = np.fft.rfft(self.time_sfw.y.reshape(self.M, self.N),
                             axis=-1).flatten() / np.sqrt(2 * np.pi)

        super().__init__(y=y_freq, fs=fs, lam=lam)  # getting attributes and methods from parent class

    def sinc_hat(self, w):
        return 1. * (np.abs(w) <= self.fs * np.pi)  # no 1/fs factor to account for FT approximation with DFT

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum(M, K, N_freq, axis=1)
        return np.sum(self.sinc_hat(self.freq_array[np.newaxis, np.newaxis, :])
                      * np.exp(-1j * self.freq_array[np.newaxis, np.newaxis, :] * dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / np.sqrt(2 * np.pi) / dist[:, :, np.newaxis]
                      * a[np.newaxis, :, np.newaxis], axis=1).flatten()

    def etak(self, x: np.ndarray) -> float:
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, Nfreq) to (M*Nfreq,)
        gammaj = (self.sinc_hat(self.freq_array[np.newaxis, :])
                  * np.exp(-1j * self.freq_array[np.newaxis, :] * dist[:, np.newaxis] / c)
                  / 4 / np.pi / np.sqrt(2 * np.pi) / dist[:, np.newaxis]).flatten()

        return -np.abs(np.sum(self.res * np.conj(gammaj))) / self.lam

    def etak_norm1(self, x: np.ndarray) -> float:
        """Normalization of etak by 1/norm_2([dist(x, xm)]_m)"""
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_hat(self.freq_array[np.newaxis, :])
                  * np.exp(-1j * self.freq_array[np.newaxis, :] * dist[:, np.newaxis] / c)
                  / 4 / np.pi / np.sqrt(2 * np.pi) / dist[:, np.newaxis] / np.linalg.norm(1 / dist)).flatten()

        return -np.abs(np.sum(self.res * np.conj(gammaj))) / self.lam

    def _LASSO_step(self):
        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((self.xkp[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # shape (M, N_freq, K) -> (J, K)
        gamma_mat_cpx = np.reshape(self.sinc_hat(self.freq_array[np.newaxis, :, np.newaxis])
                                   * np.exp(
            -1j * self.freq_array[np.newaxis, :, np.newaxis] * dist[:, np.newaxis, :] / c)
                                   / 4 / np.pi / np.sqrt(2 * np.pi) / dist[:, np.newaxis, :],
                                   newshape=(self.J, -1))

        gamma_mat = np.concatenate([np.real(gamma_mat_cpx),
                                    np.imag(gamma_mat_cpx)], axis=0)
        lasso_fitter = Lasso(alpha=self.lam, positive=True)
        target = np.concatenate([np.real(self.y), np.imag(self.y)]).reshape(-1, 1)
        scale = np.sqrt(2 * len(gamma_mat))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * gamma_mat,
                         scale * target)

        return lasso_fitter, lasso_fitter.coef_.flatten()

    def _grid_initialization_function(self, parameter, verbose):
        # todo : use a sliding window norm as in time domain case
        # compute the time domain residue
        self.time_sfw.res = self.time_sfw.y - self.time_sfw.gamma(self.ak, self.xk)
        m_max, n_max = flat_to_multi_ind(np.argmax(self.time_sfw.res), self.N)
        r = n_max * c / self.fs

        if type(parameter) == np.ndarray:  # use the generated grid and scale it to the correct radius
            grid = r * parameter
        else:  # no pre-generated grid
            if r > 1:  # increase the number of nodes on the grid if the radius exceeds 1
                dtheta = parameter / np.log(r)
            else:
                dtheta = parameter
            grid, sph_grid, n_sph = create_grid_spherical(r, r, 1., dtheta=dtheta, dphi=dtheta)

        search_grid = r * grid + self.mic_pos[m_max][np.newaxis, :]
        if verbose:
            print("searching around mic {} at a radius {}".format(m_max, r))
        return search_grid

    def _get_normalized_fun(self, normalization):
        normalized_eta = [self.etak, self.etak_norm1][normalization]
        normalized_eta_jac = "3-point"
        slide_jac = "3-point"
        return normalized_eta, normalized_eta_jac, slide_jac

    def _obj_slide(self, var, y, n_spikes):
        """
        Objective function for the sliding step, adapted for complex values
        """

        a, x = var[:self.nk], var[self.nk:].reshape(-1, self.d)
        return 0.5 * np.sum(np.abs(self.gamma(a, x) - y) ** 2) + self.lam * np.sum(np.abs(a))
