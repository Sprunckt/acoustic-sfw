import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Union, Tuple
from src.simulation.utils import c, disp_measure
from sklearn.linear_model import Lasso
import time

c = tf.cast(c, tf.float64)


class SFW:
    def __init__(self, y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], fs: float, mic_pos,
                 lam: float = 1e-2, N: int = 0, fc: float = None, deletion_tol: float = 0.05,
                 end_tol: float = 0.05):
        self.fs, self.lam = tf.cast(fs, tf.float64), tf.cast(lam, tf.float64)
        self.fc = fs if fc is None else tf.cast(fc, tf.float64)

        self.M, self.N = len(mic_pos), N
        self.NN = tf.range(0, self.N, dtype=tf.float64) / fs

        self.mic_pos = tf.Variable(mic_pos, dtype=tf.float64)

        if type(y) == np.ndarray:
            self.y = tf.Variable(y, dtype=tf.float64, shape=[self.M, self.N])
        else:
            print("wrong input type for y")
            exit(1)

        # initializing with the null measure
        self.ak, self.xk = tf.Variable(tf.zeros(1, dtype=tf.float64)),  tf.Variable(tf.zeros((1, 3), dtype=tf.float64))
        self.res = tf.identity(self.y)  # residue
        self.nk = 0
        self.it = 0
        self.timer = None

        self.deletion_tol, self.end_tol = deletion_tol, end_tol

    #@tf.function
    def kappa(self, t):
        return tf.experimental.numpy.sinc(self.fc*t)

    def etak(self, x):
        # shape M, Nx
        norms = tf.linalg.norm(tf.reshape(x, [-1, 3])[tf.newaxis, :, :] - self.mic_pos[:, tf.newaxis, :], axis=2)

        # shape (M, Nx, N) to (Nx,)
        return - tf.reduce_sum(self.res[:, tf.newaxis, :] *
                               self.kappa(self.NN[tf.newaxis, tf.newaxis, :] - norms[:, :, tf.newaxis] / c)
                               / 4 / np.pi / norms[:, :, tf.newaxis], axis=(0, 2))

    def etak_val_and_grad(self, x):
        return tfp.math.value_and_gradient(lambda u: tf.squeeze(self.etak(u)), x)

    #@tf.function
    def gamma(self, a, x):
        # shape (M, K)
        norms = tf.linalg.norm(tf.reshape(x, [-1, 3])[tf.newaxis, :, :] - self.mic_pos[:, tf.newaxis, :], axis=2)
        gammaj = (self.kappa(self.NN[tf.newaxis, tf.newaxis, :] - norms[:, :, tf.newaxis] / c)  # (M, K, N)
                  / 4 / np.pi / norms[:, :, tf.newaxis])
        return tf.reduce_sum(a[tf.newaxis, :, tf.newaxis] * gammaj, axis=1)

    def compute_residue(self):
        self.res = self.y - self.gamma(self.ak, self.xk)
        return self.res

    def init_grid_search(self, search_grid, nmic=8, verbose=False):
        sliding_mean = tf.reshape(tf.nn.conv1d(tf.reshape(tf.square(self.res), [self.M, self.N, 1]),
                                               tf.ones([3, 1, 1], dtype=tf.float64),
                                               stride=1, padding='SAME'), [self.M, self.N])

        bestind = np.argmax(sliding_mean, axis=1)
        bestval = sliding_mean.numpy()[np.arange(self.M), bestind]

        best_global = np.argsort(bestval)[-nmic:]
        dist_to_src = (bestind[best_global] / self.fs)*c  # toa * c

        full_grid = []
        for i in range(nmic):
            full_grid.append((search_grid * dist_to_src[i].numpy() +
                              self.mic_pos[best_global[i], :].numpy()[np.newaxis, :]).astype(np.float64))

        full_grid = tf.concat(full_grid, axis=0)

        if verbose:
            print("searching around mic {} at radius {}, {} grid points".format(best_global,
                                                                                dist_to_src, len(full_grid)))
        return full_grid

    def slide_obj_val_and_grad(self, x):
        return tfp.math.value_and_gradient(lambda u: tf.squeeze(self.slide_obj(u[:self.nk],
                                                                               tf.reshape(u[self.nk:], [-1, 3]))), x)

    #@tf.function
    def slide_obj(self, a, x):
        return 0.5*tf.reduce_sum(tf.square(self.y - self.gamma(tf.abs(a), x))) + self.lam*tf.reduce_sum(tf.abs(a))

    def reconstruct(self, niter, grid, verbose=False, nmic=8, search_method='naive', delete_tol=0.05):

        self.timer = time.time()
        for i in range(niter):
            # spike finding step
            tstart = time.time()
            search_grid = self.init_grid_search(grid, nmic, verbose=verbose)
            if search_method == 'naive':
                @tf.function
                def parall_map(gr):
                    return tf.argmin(tf.map_fn(self.etak, gr), output_type=tf.int32)

                indmin = parall_map(search_grid)
                ini = tf.reshape(search_grid[indmin[0]], [-1])
                print("Initial value:", self.etak(ini).numpy() / self.lam)
                optim_results = tfp.optimizer.bfgs_minimize(lambda x: self.etak_val_and_grad(x),
                                                            initial_position=ini, tolerance=1e-7)
            elif search_method == 'rough':
                optim_results = tfp.optimizer.bfgs_minimize(lambda x: self.etak_val_and_grad(x),
                                                            initial_position=search_grid, tolerance=1e-3)
                indmin = tf.argmin(self.etak(optim_results.position), output_type=tf.int32)
                optim_results = tfp.optimizer.bfgs_minimize(lambda x: self.etak_val_and_grad(x),
                                                            initial_position=optim_results.position[indmin],
                                                            tolerance=1e-8)

            tend = time.time()
            if verbose:
                print("Optimization success: ", optim_results.converged)
                print("etak value: ", self.etak(optim_results.position).numpy() / self.lam)
                print("exec time for grid optimization : ", tend - tstart)

            if self.nk > 0:
                self.xk = tf.concat([self.xk, tf.reshape(optim_results.position, [1, 3])], axis=0)
            else:
                self.xk = tf.reshape(optim_results.position, [1, 3])
            self.nk += 1

            # amplitudes optimization step
            gamma_mat = np.array(([np.array(self.gamma(tf.ones(1, dtype=tf.float64),
                                                       tf.reshape(self.xk[k, :], [1, 3]))).flatten()
                                   for k in range(self.nk)])).T

            lasso_fitter = Lasso(alpha=self.lam, positive=True)

            scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
            lasso_fitter.fit(scale * gamma_mat,
                             scale * self.y.numpy().flatten())
            lasso_fitter, lasso_fitter.coef_.flatten()
            self.ak = tf.Variable(lasso_fitter.coef_.flatten(), dtype=tf.float64)

            if verbose:
                print("New measure :")
                disp_measure(self.ak, self.xk)

            # delete null amplitudes
            ind_not_null = self.ak.numpy() >= delete_tol

            self.ak = tf.Variable(self.ak.numpy()[ind_not_null])
            self.xk = tf.Variable(self.xk.numpy()[ind_not_null, :])
            self.nk = ind_not_null.sum()

            self.compute_residue()
            if not ind_not_null[-1]:  # last spike is null
                break
        self.timer = time.time() - self.timer
        if verbose:
            print("Total execution time: ", self.timer)

        # sliding step before stopping
        tstart = time.time()
        ini = tf.concat([self.ak, tf.reshape(self.xk, [-1])], axis=0)
        print(ini.shape, self.slide_obj_val_and_grad(ini))
        optim_results = tfp.optimizer.bfgs_minimize(lambda x: self.slide_obj_val_and_grad(x),
                                                    initial_position=ini, tolerance=1e-7)
        self.ak = np.array(optim_results.position[:self.nk])
        self.xk = np.array(optim_results.position[self.nk:]).reshape([-1, 3])

        if verbose:
            print("Optimization success: ", optim_results.converged)
            print("Time for sliding step: {} s".format(time.time() - tstart))
            print("Initial value: {}, final value: {}".format(self.slide_obj(ini[:self.nk],
                                                                             tf.reshape(ini[self.nk:], [-1, 3])),
                                                              self.slide_obj(self.ak, self.xk)))
            print("Final measure:")
            disp_measure(self.ak, self.xk)

        return self.ak, self.xk

    def update_mic_pos(self, mic_pos):
        self.mic_pos = tf.Variable(mic_pos, dtype=tf.float64)


if __name__ == '__main__':
    sf = SFW(np.random.random([10, 56]), fs=16000, mic_pos=np.random.random([10, 3]), N=56)
    search_grid = tf.random.normal(shape=(1000, 3), dtype=tf.float64)
    sf.reconstruct(10, search_grid, verbose=True)