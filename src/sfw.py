import numpy as np
from typing import Union, Tuple
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from src.simulation.utils import (disp_measure, c)
import multiprocessing
import time

sfw_tol = 1e-6
merge_tol = 0.02


def flat_to_multi_ind(ind, N):
    return ind // N, ind % N


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
            -lam (float) : penalization parameter of the BLASSO.
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
        self.eta, self.eta_jac = None, None
        self.timer = None

    def sinc_filt(self, t):
        """
        The filter applied for each measurement.
        """

        return np.sinc(t * self.fs)

    def sinc_der(self, t):
        """
        Derivate of the filter
        """
        w = np.pi*self.fs
        return (t*np.cos(w*t) - np.sin(w*t)/w) / (t**2+np.finfo(float).eps)

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
        """Objective function for the new spike location optimization ; the normalization by lambda is only taken in
        account for the stopping criterion and not during optimization"""

        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] / self.fs - dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis]).flatten()

        return -np.abs(np.sum(self.res * gammaj)) / self.lam

    def etak_norm1(self, x: np.ndarray) -> float:
        """Normalization of etak by 1/norm_2([dist(x, xm)]_m)"""
        # distances from x (in R^3) to every microphone, shape (M,)
        dist = np.linalg.norm(x[np.newaxis, :] - self.mic_pos, axis=1)

        # shape (M, N) to (M*N,)
        gammaj = (self.sinc_filt(self.NN[np.newaxis, :] / self.fs - dist[:, np.newaxis] / c)
                  / 4 / np.pi / dist[:, np.newaxis] / np.linalg.norm(1/dist)).flatten()

        return -np.abs(np.sum(self.res * gammaj)) / self.lam

    def _jac_etak(self, x):
        diff = x[np.newaxis, :] - self.mic_pos[:, :]  # difference, shape (M, 3)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        int_term = self.NN[np.newaxis, :] / self.fs - dist[:, np.newaxis] / c  # shape (M, N)

        # sum shape (M,  N) into shape (M,), derivate without the xk_i - xm_i factor
        tens = np.sum(((- self.sinc_filt(int_term) / dist[:, np.newaxis] - self.sinc_der(int_term) / c)
                      / dist[:, np.newaxis]**2 / 4 / np.pi) * self.res.reshape(self.M, self.N), axis=1)

        # shape (M,3) into (M,)
        jac = (np.sum(tens[:, np.newaxis] * diff, axis=0).flatten())

        return -jac*np.sign(self.etak(x)) / self.lam

    def _jac_etak_norm1(self, x):
        diff = x[np.newaxis, :] - self.mic_pos[:, :]  # difference, shape (M, 3)
        # distances from in to every microphone, shape (M,)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        int_term = self.NN[np.newaxis, :] / self.fs - dist[:, np.newaxis] / c  # shape (M, N)

        norm2 = np.sum(1/dist**2)
        norm = np.sqrt(norm2)  # float
        norm_der = - np.sum(diff/dist[:, np.newaxis]**4, axis=0) / norm  # shape (3,)

        # sum shape (M,  N) into shape (M,), derivate  of gammaj * N without the xk_i - xm_i factor
        der_gam = np.sum(((- self.sinc_filt(int_term) / dist[:, np.newaxis] - self.sinc_der(int_term) / c)
                         / dist[:, np.newaxis]**2 / 4 / np.pi / norm) * self.res.reshape(self.M, self.N), axis=1)
        der_gam = (np.sum(der_gam[:, np.newaxis] * diff, axis=0).flatten())  # shape (3,)

        # derivate of N * gammaj without the xk_i - xm_i factor
        der_N = (np.sum((- self.sinc_filt(int_term) / dist[:, np.newaxis] / 4 / np.pi).flatten() * self.res)
                 * norm_der / norm2)  # shape (3,)

        return -np.sign(self.etak_norm1(x))*(der_gam + der_N) / self.lam

    def _jac_slide_obj(self, var):
        a, x = var[:self.nk], var[self.nk:].reshape(-1, self.d)
        diff = x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]  # difference, shape (M, K, 3)
        # distances from the diracs contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        jac = np.zeros(self.nk*4)

        int_term = self.NN[np.newaxis, np.newaxis, :] / self.fs - dist[:, :, np.newaxis] / c
        # shape (M, K, N) = gamma_j(x_k)
        gamma_tens = (self.sinc_filt(int_term) / 4 / np.pi / dist[:, :, np.newaxis])

        # sum_k ak.gamma_j(x_k) - y_j : sum(M, K, N, axis=1) - y = -residue
        residue = (np.sum(gamma_tens * a[np.newaxis, :, np.newaxis], axis=1).flatten() - self.y)
        # derivates in ak : multiply the residue by gammaj(x_k) and sum on j (meaning m and N)
        jac[:self.nk] = (np.sum(residue.reshape(self.M, self.N)[:, np.newaxis, :] * gamma_tens, axis=(0, 2))
                         + self.lam*np.sign(a))

        # shape (M, K, N), derivate without the xk_i - xm_i factor
        gamma_tens = ((- gamma_tens - self.sinc_der(int_term) / 4 / np.pi / c)
                      / dist[:, :, np.newaxis]**2 * residue.reshape(self.M, self.N)[:, np.newaxis, :])

        # original shape (M,K,3,N)
        jac[self.nk:] = (np.repeat(a, 3)
                         * np.sum(gamma_tens[:, :, np.newaxis, :] * diff[:, :, :, np.newaxis], axis=(0, 3)).flatten())

        return jac

    def _create_gamma_mat(self, x):
        """
        Linear operator (M(R^d) -> R^J)
        Args:
            -a (array) : flat array containing the amplitudes of the spikes
            -x (array) : positions of the spikes, shape (len(a), d)
        Return: array containing the evaluation of gamma on the measure (shape (J,))
        """

        # distances from the spikes contained in x to every microphone, shape (M,K), K=len(x)
        dist = np.sqrt(np.sum((x[np.newaxis, :, :] - self.mic_pos[:, np.newaxis, :]) ** 2, axis=2))

        # shape (M, N, K) -> (J, K)
        return np.reshape(self.sinc_filt(self.NN[np.newaxis, :, np.newaxis] / self.fs - dist[:, np.newaxis, :] / c)
                          / 4 / np.pi / dist[:, np.newaxis, :], newshape=(self.J, -1))

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
        return minimize(self.eta, x, jac="3-point", method="BFGS", options=self.opt_options)

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

    def reconstruct(self, grid=None, niter=7, min_norm=-np.inf, max_norm=np.inf, max_ampl=np.inf, normalization=0,
                    rough_search=False, spike_merging=False, spherical_search=0,
                    use_hard_stop=True, verbose=True, early_stopping=False) -> (np.ndarray, np.ndarray):
        """
        Apply the SFW algorithm to reconstruct the the measure based on the measurements self.y.

        Args:
            -grid (array) : grid used for the initial guess of the new spike
            -niter (int) : maximal number of iterations
            -max_ampl (float) : upper bound on spikes amplitudes
            -normalization (int) : the normalization to add to gamma for he spike localization step. 0 means no
        normalization, 1 adds a factor 1/(sqrt(sum(1/dist(x, xm)**2))
            -min_norm (float) : minimal norm allowed for the position found at the end of the grid search
            -max_norm (float) : used as bounds for the coordinates of the spike locations in each direction
            -use_hard_stop (bool) : if True, add |etak| < 1 as a stopping condition
            -early_stopping (bool) : if True, stop at the end of an iteration if the last spike found has zero amplitude

         Return:
            (ak, xk) where :
            -ak is a flat array of shape (K,) containing the amplitudes of the recovered measure
            -xk is a (K, d) shaped array containing the locations of the K spikes composing the measure
        """
        self.timer = time.time()
        normalizations, normalizations_jac = [self.etak, self.etak_norm1], [self._jac_etak, self._jac_etak_norm1]
        self.eta = normalizations[normalization]
        self.eta_jac = normalizations_jac[normalization]

        xmin, xmax = -max_norm, max_norm
        amin, amax = 0, max_ampl
        self.nk = 0

        search_grid = grid
        assert search_grid is not None, "a grid must be specified for the initial grid search"

        for i in range(niter):

            # find argmax etak to get the new spike position (step 3)
            if verbose:
                print("Optimizing for spike position -----")

            # grid search : one optimization per grid point (the grid has the shape (Ngrid, d))
            tstart = time.time()

            if spherical_search == 1:  # take argmax on the complete rir and search on the corresponding sphere
                m_max, n_max = flat_to_multi_ind(np.argmax(self.res), self.N)
                r = n_max * c / self.fs
                search_grid = r * grid + self.mic_pos[m_max][np.newaxis, :]

            if verbose:
                print("Starting a grid search to minimize etak")

            if rough_search:  # perform a low precision search
                self.opt_options["gtol"] = 1e-2

            # spreading the loop over multiple processors
            ncores = multiprocessing.cpu_count()
            p = multiprocessing.Pool(ncores)
            gr_opt = p.map(self._optigrid, search_grid)
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
                return self._stop(verbose=verbose)

            # solve LASSO to adjust the amplitudes according to the new spike (step 7)
            if self.nk > 0:
                self.xkp = np.concatenate([self.xk, x_new], axis=0)
            else:
                self.xkp = np.array([x_new]).reshape(1, self.d)
            self.nk += 1

            if verbose:
                print("Optimizing for spike amplitudes --------")

            tstart = time.time()

            lasso_fitter = Lasso(alpha=self.lam, positive=True)
            gamma_mat = self._create_gamma_mat(self.xkp)
            lasso_fitter.fit(np.sqrt(self.J) * gamma_mat,
                             np.sqrt(self.J) * self.y)
            ak_new = lasso_fitter.coef_.flatten()
            if verbose:
                print("LASSO solved in {} iterations, exec time : {} s".format(lasso_fitter.n_iter_,
                                                                               time.time() - tstart))
                print("New amplitudes : {} \n".format(ak_new))

            # solve to adjust both the positions and amplitudes
            ini = np.concatenate([ak_new, self.xkp.flatten()])

            if verbose:
                print("Sliding step --------")
                print("initial value : {}".format(self._obj_slide(ini)))

            tstart = time.time()
            bounds = [(amin, amax)] * self.nk + [(xmin, xmax)] * self.nk * self.d
            opti_res = minimize(self._obj_slide, ini, jac=self._jac_slide_obj, method="L-BFGS-B", bounds=bounds)
            mk = opti_res.x
            self.ak, self.xk = mk[:self.nk], mk[self.nk:].reshape([-1, self.d])

            if not opti_res.success:
                print("Last optimization failed, reason : {}".format(opti_res.message))

            if verbose:
                print("Optimization converged in {} iterations, exec time : {} s".format(opti_res.nit,
                                                                                         time.time() - tstart))
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
                return self._stop(verbose=verbose)

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
        return self._stop(verbose=verbose)
