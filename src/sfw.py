import numpy as np
from scipy.optimize import minimize
from utils import (disp_measure, tol, c)
import multiprocessing
import time


class SFW:
    def __init__(self, y: np.ndarray, mic_pos: np.ndarray,
                 fs: float, N: int, lam: float = 1e-2):
        """
        Args:
            -y (ndarray) : measurements (shape (J,) = (N*M,))
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

        self.y = y
        assert self.y.size == self.J, "invalid measurements length"

        self.lam = lam
        self.NN = np.arange(self.N)

        self.ak, self.xk = np.zeros(1), np.zeros((1, self.d))  # initializing with the null measure
        self.xkp = np.zeros((1, self.d))  # temporary storage for spike locations
        self.res = self.y.copy()  # residue
        self.nk = 0

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
        return minimize(self.etak, x, jac="3-point", method="BFGS")

    def _stop(self):
        return self.ak, self.xk

    def reconstruct(self, grid=None, niter=7,
                    use_hard_stop=True, verbose=True, early_stopping=False) -> (np.ndarray, np.ndarray):
        """
        Apply the SFW algorithm to reconstruct the the measure based on the measurements self.y.

        Args:
            -grid (array) : grid used for the initial guess of the new spike
            -niter (int) : maximal number of iterations
            -use_hard_stop (bool) : if True, add |etak| < 1 as a stopping condition
            -early_stopping (bool) : if True, stop at the end of an iteration if the last spike found has zero amplitude
         Return:
            (ak, xk) where :
            -ak is a flat array of shape (K,) containing the amplitudes of the recovered measure
            -xk is a (K, d) shaped array containing the locations of the K spikes composing the measure
        """

        xmin, xmax = -np.inf, np.inf
        amin, amax = 0, np.inf
        self.nk = 0

        for i in range(niter):

            # find argmax etak to get the new spike position (step 3)
            if verbose:
                print("Optimizing for spike position -----")

            # grid search : one optimization per grid point (the grid has the shape (N, d))
            if grid is not None:
                tstart = time.time()

                if verbose:
                    print("Starting a grid search to minimize etak")

                # spreading the loop over multiple processors
                p = multiprocessing.Pool(8)
                gr_opt = p.map(self._optigrid, grid)
                p.close()

                # searching for the best result over the grid
                curr_min, curr_opti_res = np.inf, None
                for el in gr_opt:
                    if el.fun < curr_min:
                        curr_min = el.fun
                        curr_opti_res = el

                del gr_opt
                opti_res = curr_opti_res

                if verbose:
                    print("exec time for grid optimization : ", time.time() - tstart)
            else:
                x_ini = np.zeros(self.d)
                print("initial guess : {} \n eta value : {}".format(str(x_ini), self.etak(x_ini)))
                opti_res = minimize(self.etak, x_ini, jac="3-point", method="BFGS")

            x_new = opti_res.x.reshape([1, self.d])
            etaval = np.abs(opti_res.fun)

            if not opti_res.success:
                print("etak optimization failed, reason : {}".format(opti_res.message))

            if verbose:
                print("Optimization converged in {} iterations".format(opti_res.nit))
                print("New position : {} \n eta value : {}".format(x_new, etaval))

            if use_hard_stop and etaval <= 1 + tol:
                if verbose:
                    print("Stopping criterion met : etak(x_new)={} < 1")
                return self._stop()

            # solve LASSO to adjust the amplitudes according to the new spike (step 7)
            if self.nk > 0:
                self.xkp = np.concatenate([self.xk, x_new], axis=0)
                a_ini = np.append(self.ak, 0)
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
            ind_null = np.asarray(np.abs(self.ak) < tol)
            self.ak = self.ak[~ind_null]
            self.xk = self.xk[~ind_null, :]
            self.nk = len(self.ak)
            if self.nk == 0:
                print("Error : all spikes are null")
                return
            elif early_stopping and len(ind_null) == 1 and ind_null[0] == len(self.ak):  # last spike is null
                print("Last spike has null amplitude, stopping")
                return self._stop()

            if verbose:
                print("New measure :")
                disp_measure(self.ak, self.xk)
            gk = self.gamma(self.ak, self.xk)
            self.res = self.y - gk
        print("Maximum number of iterations {} reached".format(niter))
        return self._stop()
