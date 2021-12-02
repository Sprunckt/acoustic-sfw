import numpy as np
from typing import Union, Tuple
from scipy.optimize import minimize
from src.simulation.utils import (disp_measure, c)
import multiprocessing
import time

sfw_tol = 1e-6
merge_tol = 0.1


class SFW:
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
            -lam (float) : penalization parameter of the BLASSO
        """

        self.fs, self.N = fs, N
        self.mic_pos, self.M = mic_pos, len(mic_pos)
        self.J = self.M * self.N

        self.d = mic_pos.shape[1]
        assert 1 < self.d < 4, "Invalid dimension d"

        self.lam = lam
        self.NN = np.arange(self.N)

        self.ak, self.xk = np.zeros(1), np.zeros((1, self.d))  # initializing with the null measure
        self.xkp = np.zeros((1, self.d))  # temporary storage for spike locations

        if type(y) == np.ndarray:
            self.y = y
        else:
            self.y = self.gamma(y[0], y[1])
        assert self.y.size == self.J, "invalid measurements length"

        self.res = self.y.copy()  # residue
        self.nk = 0

        self.opt_options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08,
                            'maxiter': None, 'disp': False, 'return_all': False, 'finite_diff_rel_step': None}

    def sinc_filt(self, t):
        """
        The filter applied for each measurement.
        """

        return np.sinc(t * self.fs)

    def gamma(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Linear operator (M(R^d) -> R^J)
        Args:
            -a (array) : flat array containing the amplitudes of the spikes
            -x (array) : positions of the spikes, shape (len(a), d)
        Return: array containing the evaluation of gamma on the measure (shape (J,))
        """

        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # sum( M, K, N, axis=1)
        return np.sum(self.sinc_filt(self.NN[np.newaxis, :] / self.fs - dist[:, :, np.newaxis] / c)
                      / 4 / np.pi / dist[:, :, np.newaxis]
                      * a[np.newaxis, :, np.newaxis], axis=1).flatten()

    def etak(self, x: np.ndarray) -> float:
        """Objective function for the new spike location optimization"""

        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.sqrt(np.sum((x[np.newaxis, :] - self.mic_pos) ** 2, axis=1))

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] / self.fs - dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis]).flatten()

        return -np.abs(np.sum(self.res * gammaj)) / self.lam

    def _obj_lasso(self, a):
        """
        Objective function for the optimization on spike amplitudes.
        """

        return 0.5 * np.sum((self.gamma(a, self.xkp) - self.y) ** 2) + self.lam * np.sum(np.abs(a))

    def _obj_slide(self, var):
        """
        Objective function for the sliding step (optimization on locations and amplitude).
        """

        a, x = var[:self.nk], var[self.nk:].reshape(-1, self.d)
        return 0.5 * np.sum((self.gamma(a, x) - self.y) ** 2) + self.lam * np.sum(np.abs(a))

    def _optigrid(self, x):
        return minimize(self.etak, x, jac="3-point", method="BFGS", options=self.opt_options)

    def _stop(self):
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

    def reconstruct(self, grid=None, niter=7, min_norm=-np.inf, max_norm=np.inf, max_ampl=np.inf,
                    rough_search=False, spike_merging=False,
                    use_hard_stop=True, verbose=True, early_stopping=False) -> (np.ndarray, np.ndarray):
        """
        Apply the SFW algorithm to reconstruct the the measure based on the measurements self.y.

        Args:
            -grid (array) : grid used for the initial guess of the new spike
            -niter (int) : maximal number of iterations
            -min_norm (float) : minimal norm allowed for the position found at the end of the grid search
            -max_norm (float) : used as bounds for the coordinates of the spike locations in each direction
            -use_hard_stop (bool) : if True, add |etak| < 1 as a stopping condition
            -early_stopping (bool) : if True, stop at the end of an iteration if the last spike found has zero amplitude
         Return:
            (ak, xk) where :
            -ak is a flat array of shape (K,) containing the amplitudes of the recovered measure
            -xk is a (K, d) shaped array containing the locations of the K spikes composing the measure
        """

        xmin, xmax = -max_norm, max_norm
        amin, amax = 0, max_ampl
        self.nk = 0

        for i in range(niter):

            # find argmax etak to get the new spike position (step 3)
            if verbose:
                print("Optimizing for spike position -----")

            # grid search : one optimization per grid point (the grid has the shape (N, d))
            tstart = time.time()

            if verbose:
                print("Starting a grid search to minimize etak")

            if rough_search:  # perform a low precision search
                self.opt_options["gtol"] = 1e-2

            # spreading the loop over multiple processors
            p = multiprocessing.Pool(8)
            gr_opt = p.map(self._optigrid, grid)
            p.close()

            # searching for the best result over the grid
            curr_min, curr_opti_res = np.inf, None
            for el in gr_opt:
                if el.fun < curr_min and np.linalg.norm(el.x) > min_norm:
                    curr_min = el.fun
                    curr_opti_res = el
            if rough_search:  # perform a finer optimization using the position found as initialization
                nit = curr_opti_res.nit
                self.opt_options["gtol"] = 1e-6
                opti_res = self._optigrid(curr_opti_res.x)
                nit += opti_res.nit
            else:
                opti_res = curr_opti_res
                nit = opti_res.nit

            del gr_opt

            if verbose:
                print("exec time for grid optimization : ", time.time() - tstart)

            x_new = opti_res.x.reshape([1, self.d])
            etaval = np.abs(opti_res.fun)

            if not opti_res.success:
                print("etak optimization failed, reason : {}".format(opti_res.message))

            if verbose:
                print("Optimization converged in {} iterations".format(nit))
                print("New position : {} \n eta value : {}".format(x_new, etaval))

            if use_hard_stop and etaval <= 1 + sfw_tol:
                if verbose:
                    print("Stopping criterion met : etak(x_new)={} < 1".format(etaval))
                return self._stop()

            # solve LASSO to adjust the amplitudes according to the new spike (step 7)
            if self.nk > 0:
                self.xkp = np.concatenate([self.xk, x_new], axis=0)
                a_ini = np.append(self.ak, self.ak[-1])
            else:
                self.xkp = np.array([x_new]).reshape(1, self.d)
                a_ini = np.zeros(1)
            self.nk += 1

            if verbose:
                print("Optimizing for spike amplitudes --------")
                print("initial value : {}".format(self._obj_lasso(a_ini)))

            bounds = [(amin, amax)] * self.nk
            opti_res = minimize(self._obj_lasso, x0=a_ini, jac="3-point", method="L-BFGS-B", bounds=bounds)
            ak_new = opti_res.x
            if not opti_res.success:
                print("ak optimization failed, reason : {}".format(opti_res.message))
            elif verbose:
                print("Optimization converged in {} iterations".format(opti_res.nit))
                print("New amplitudes : {} \n objective value : {}".format(ak_new, opti_res.fun))

            # solve to adjust both the positions and amplitudes
            ini = np.concatenate([ak_new, self.xkp.flatten()])

            if verbose:
                print("Sliding step --------")
                print("initial value : {}".format(self._obj_slide(ini)))
            bounds = [(amin, amax)] * self.nk + [(xmin, xmax)] * self.nk * self.d
            opti_res = minimize(self._obj_slide, ini, jac="3-point", method="L-BFGS-B", bounds=bounds)
            mk = opti_res.x
            self.ak, self.xk = mk[:self.nk], mk[self.nk:].reshape([-1, self.d])

            if not opti_res.success:
                print("Last optimization failed, reason : {}".format(opti_res.message))

            if verbose:
                print("Optimization converged in {} iterations".format(opti_res.nit))
                print("Objective value : {}".format(opti_res.fun))

            # deleting null amplitude spikes
            ind_null = np.asarray(np.abs(self.ak) < sfw_tol)
            self.ak = self.ak[~ind_null]
            self.xk = self.xk[~ind_null, :]
            self.nk = len(self.ak)
            if self.nk == 0:
                print("Error : all spikes are null")
                return
            elif early_stopping and len(ind_null) == 1 and ind_null[0] == len(self.ak):  # last spike is null
                print("Last spike has null amplitude, stopping")
                return self._stop()

            if spike_merging:
                if verbose:
                    print("merging spikes --")

                tot_merged = self.merge_spikes()

                if verbose:
                    print("{} spikes merged".format(tot_merged))

            if verbose:
                print("New measure :")
                disp_measure(self.ak, self.xk)
            gk = self.gamma(self.ak, self.xk)
            self.res = self.y - gk
        print("Maximum number of iterations {} reached".format(niter))
        return self._stop()
