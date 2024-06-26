import numpy as np
import matplotlib.pyplot as plt
from src.simulation.utils import create_grid_spherical
import multiprocessing as mp
import os
import time
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from src.simulation.utils import unique_matches
import jax.numpy as jnp
import jaxopt
import jax
from jax.config import config
from functools import partial
config.update("jax_debug_nans", True)  # track nan occurences
config.update("jax_enable_x64", True)  # force float64 precision, disable for speedup

ncores = len(os.sched_getaffinity(0))


def spherical_to_cartesian(r, theta, phi):
    return np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])


def cartesian_to_spherical(p):
    p = p.reshape([-1, 3])
    r = np.sqrt(p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2)
    phi = np.arccos(p[:, 2] / r)  # elevation
    theta = np.arctan2(p[:, 1], p[:, 0])
    return np.stack([r, theta, phi], axis=1)


def spherical_distance_to_cartesian(ang_dist, radial_dist, radius, rad=True):
    if not rad:
        ang_dist = ang_dist*np.pi / 180
    r1, r2 = radius, radius + radial_dist
    return np.sqrt((r1 - r2*np.abs(np.cos(ang_dist)))**2 + r2**2*np.sin(ang_dist)**2)


def great_circle_distance(p1, p2, rad=True):
    tmp1, tmp2 = p1 / np.linalg.norm(p1, axis=-1)[:, np.newaxis], p2 / np.linalg.norm(p2, axis=-1)[:, np.newaxis]
    ang = np.arccos(np.sum(tmp1 * tmp2, axis=-1))
    if rad:
        return ang
    else:
        return ang * 180. / np.pi


def radial_distance(p1, p2, axis=-1):
    return np.abs(np.linalg.norm(p1, axis=axis)-np.linalg.norm(p2, axis=axis))


def degrees_to_rad(alpha):
    return np.pi * alpha / 180


def same_half_space(x, y, normal, origin, tol=1e-6):
    return (np.dot(x, normal) - origin)*(np.dot(y, normal) - origin) > tol


def generate_src_coordinates(d1, d2, lower_bound, upper_bound):
    """Generate the coordinates of the image sources in one direction, given the distances of the source to the walls
    and bounds (assume the origin is located at the source)."""
    coord_list = []
    curr_coord, parity, old_coord = 0, 0, 0
    while curr_coord > lower_bound:
        if parity == 0:
            curr_coord = old_coord - 2 * d1
        else:
            curr_coord = old_coord - 2 * d2
        old_coord = curr_coord
        if curr_coord > lower_bound:
            coord_list.append(curr_coord)
        parity = (parity + 1) % 2

    curr_coord, parity, old_coord = 0, 0, 0
    while curr_coord < upper_bound:
        if parity == 0:
            curr_coord = old_coord + 2 * d2
        else:
            curr_coord = old_coord + 2 * d1
        old_coord = curr_coord
        if curr_coord < upper_bound:
            coord_list.append(curr_coord)
        parity = (parity + 1) % 2

    return np.sort(coord_list)


def find_order1(image_pos, amplitudes, fusion=0., tol=0.1):
    """
    Simple algorithm to find the order 1 image sources from a cloud of image sources, by sorting the sources by distance
    and eliminating the points located in the half-spaces containing the order 1 sources found at the preceding
    iterations. It assumes the source and the order 1 image sources are present. Not robust to noise/false positives.
    """
    norms = np.linalg.norm(image_pos, axis=1)
    min_dist_ind = np.argmin(norms)
    source_pos = image_pos[min_dist_ind]  # find the source position as the closest source from the microphone
    if fusion > 0:
        is_close = np.linalg.norm(source_pos - image_pos, axis=1) < fusion
        ampl_src = np.sum(amplitudes[is_close])
        source_pos = np.sum(image_pos[is_close] * amplitudes[is_close, np.newaxis], axis=0) / ampl_src
        image_pos, amplitudes = image_pos[~is_close], amplitudes[~is_close]  # remove the merged sources
    else:
        ampl_src = amplitudes[min_dist_ind]
        image_pos = image_pos[np.arange(len(image_pos)) != min_dist_ind]  # image sources (removed the source)
        amplitudes = amplitudes[np.arange(len(image_pos)) != min_dist_ind]

    n_image_src = len(image_pos)
    order1, wall_pos = np.empty([6, 3], dtype=float), np.empty([6, 3], dtype=float)
    ampl_order1 = np.empty(6, dtype=float)
    dist = np.linalg.norm(source_pos[np.newaxis, :] - image_pos, axis=1)  # distances to source
    sorted_dist_ind = np.argsort(dist)

    closest_src = image_pos[sorted_dist_ind[0]]  # image source that is closest to the source (should be order 1)
    order1[0] = closest_src
    ampl_order1[0] = amplitudes[sorted_dist_ind[0]]
    wall_pos[0] = (closest_src + source_pos) / 2.
    normal_vect, origins = np.zeros((6, 3)),  np.zeros(6)
    normal_vect[0] = closest_src - source_pos
    origins[0] = np.dot(wall_pos[0], normal_vect[0])

    it_dist, nb_found = 1, 1
    old_found = 0
    while nb_found < 6:  # loop over the image sources until an order 1 is found
        found = True
        curr_source = image_pos[sorted_dist_ind[it_dist]]
        for k in range(nb_found):  # check if the point is in the same half-space as another order 1 source
            if same_half_space(order1[k], curr_source, normal_vect[k], origins[k], tol=tol):
                found = False
                break
        if found:
            if fusion > 0:  # merge sources at dist < fusion of curr_source, average by amplitudes
                is_close = np.linalg.norm(curr_source - image_pos, axis=1) < fusion
                ampl_order1[nb_found] = np.sum(amplitudes[is_close])
                order1[nb_found] = (np.sum(image_pos[is_close] * amplitudes[is_close, np.newaxis], axis=0) /
                                    ampl_order1[nb_found])
                image_pos, amplitudes = image_pos[~is_close], amplitudes[~is_close]  # remove the merged sources

                # recompute the distances in case of fusion
                dist = np.linalg.norm(source_pos[np.newaxis, :] - image_pos, axis=1)  # distances to source
                sorted_dist_ind, n_image_src = np.argsort(dist), len(dist)
                it_dist = 0
            else:
                order1[nb_found] = curr_source
                ampl_order1[nb_found] = amplitudes[sorted_dist_ind[it_dist]]
            wall_pos[nb_found] = (source_pos + order1[nb_found]) / 2.
            normal_vect[nb_found] = order1[nb_found] - source_pos
            origins[nb_found] = np.dot(wall_pos[nb_found], normal_vect[nb_found])
            nb_found += 1
            old_found = nb_found

        it_dist += 1
        if it_dist == n_image_src and nb_found < 6:  # if all the image sources have been checked and not enough
            tol *= 1.5  # order 1 sources have been found, increase the tolerance and start again
            it_dist = old_found
    return order1[:nb_found], wall_pos[:nb_found], source_pos, ampl_order1[:nb_found], ampl_src


def trim_clusters(cluster_centers, cluster_size, labels, threshold, trim_edges=True):
    if len(cluster_size) > 3:
        if threshold >= 1:
            ind_keep = cluster_size >= threshold
        elif 0 < threshold < 1:
            maxval = np.max(cluster_size)
            ind_keep = cluster_size >= threshold * maxval
        else:
            ind_keep = np.ones(len(cluster_size), dtype=bool)

        if not trim_edges:
            ind_inf, ind_sup = np.argmin(cluster_centers), np.argmax(cluster_centers)
            if cluster_size[ind_inf] > 0:
                ind_keep[ind_inf] = True
            if cluster_size[ind_sup] > 0:
                ind_keep[ind_sup] = True

        if ind_keep.sum() < 3:  # keep at least 3 clusters  (greatest 3)
            ind_keep = np.sort(np.argsort(cluster_size)[-3:])   # sort the indices of the 3 largest clusters
        else:
            ind_keep = np.where(ind_keep)[0]  # extract the indices of the clusters to keep

        cluster_centers, cluster_size = cluster_centers[ind_keep], cluster_size[ind_keep]

        # set the labels of removed clusters to -1
        labels[np.isin(labels, ind_keep, invert=True)] = -1
        labels, _ = reorganize_labels(labels)

    return cluster_centers, cluster_size, labels


def kmeans_clustering(data, niter=300, threshold=1., init=None):
    """Kmeans clustering of 1D data using sklearn's implementation. If init is an array of centroids: use the number
    of centroids as the number of clusters. Otherwise: choose the number of clusters as half of the data range + 1.
    If threshold >= 1, only return the clusters containing > threshold
    points. If 0 < threshold < 1, only return the clusters containing more than
    threshold * max(cluster_size) points. Otherwise: keep all the clusters."""

    data = data.reshape(-1, 1)

    nstart = int((np.max(data) - np.min(data))/2) + 1 if init is None else len(init)
    (init, n_init) = ('k-means++', 50) if init is None else (init.reshape(-1, 1), 1)#('k-means++', 50)
    ms = KMeans(n_clusters=nstart, max_iter=niter, random_state=42, init=init, n_init=n_init)
    ms.fit(data)
    labels, cluster_centers = ms.labels_, ms.cluster_centers_
    cluster_centers = delete_empty_clusters(cluster_centers, labels)
    labels, cluster_size = reorganize_labels(labels)

    cluster_centers, cluster_size, labels = trim_clusters(cluster_centers=cluster_centers,
                                                          cluster_size=cluster_size,
                                                          labels=labels,
                                                          threshold=threshold)
    if len(cluster_size) != len(cluster_centers):
        raise ValueError("Number of clusters and cluster sizes do not match. Something went wrong.")

    return cluster_centers.flatten(), cluster_size, labels


def delete_empty_clusters(clusters, labels):
    counts = np.zeros(len(clusters))
    unique_labels, counts_unique = np.unique(labels[labels >= 0], return_counts=True)
    counts[unique_labels] = counts_unique
    clusters = clusters[counts > 0]

    return clusters


def reorganize_labels(labels):
    """reorganize the labels to be between -1 and nclusters-1. Assume that there are as many labels as there are
    clusters (+1 if there are outliers). Has to be applied AFTER deleting empty clusters, otherwise it will wreck
    the indexation."""
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    old_labels = np.copy(labels)
    for i in range(len(unique_labels)):
        labels[old_labels == unique_labels[i]] = i

    return labels, counts


def sort_clusters(clusters, cluster_sizes, labels):
    """Sort the clusters and the labels according to the cluster sizes"""

    ind_sort = np.argsort(clusters.flatten())
    clusters = clusters[ind_sort].flatten()  # sort the clusters according to coordinates
    nclusters = len(clusters)
    cluster_sizes = cluster_sizes[ind_sort]  # cluster sizes
    non_out_mask = labels >= 0
    ind_sort_inv = np.zeros(nclusters, dtype=int)
    ind_sort_inv[ind_sort] = np.arange(nclusters)

    labels[non_out_mask] = ind_sort_inv[labels[non_out_mask]]  # sort labels accordingly

    return clusters, cluster_sizes, labels


def sample_sphere(n):
    """Monte-Carlo sampling of n points uniformly on the unit sphere"""
    p = np.random.normal(size=(n, 3))
    return cartesian_to_spherical(p)


class RotationFitter(ABC):
    """Abstract class for orientation fitting algorithms. Now only supports the kernel method (continuous histogram
    method was removed in commit 26a137d)."""
    def __init__(self, image_pos):
        self.image_pos = image_pos.copy()

        self.norms = np.linalg.norm(self.image_pos, axis=-1)
        self.n_src = len(self.image_pos)
        self.image_pos_transf = cartesian_to_spherical(self.image_pos)
        self.sin_pos, self.cos_pos = np.sin(self.image_pos_transf[:, 1:]), np.cos(self.image_pos_transf[:, 1:])
        self.cos_pos2, self.sin_pos2 = None, None  # will contain the cos and sin after projection

        self.costfun_grad, self.costfun_grad2 = None, None

    @abstractmethod
    def costfun(self, u, sig):
        pass

    @abstractmethod
    def costfun2(self, u, sig):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def compute_dot_prod(self, cos_u, sin_u):
        """Compute the dot product between u and the source positions."""
        return (self.cos_pos[:, 0] * self.sin_pos[:, 1] * cos_u[jnp.newaxis, 0] * sin_u[jnp.newaxis, 1] +
                self.sin_pos[:, 0] * self.sin_pos[:, 1] * sin_u[jnp.newaxis, 0] * sin_u[jnp.newaxis, 1] +
                self.cos_pos[:, 1] * cos_u[jnp.newaxis, 1]) * self.image_pos_transf[:, 0]

    def compute_dot_prod_polar(self, cos_u, sin_u):
        """Compute the dot product between u and the source positions."""
        return self.image_pos_transf[:, 0] * (cos_u*self.cos_pos2 + sin_u*self.sin_pos2)

    def fit(self, gridparam=2000, niter=10000, tol=1e-10, bvalues3d=None, bvalues2d=None, verbose=False, plot=False):
        """Reconstruct the distance of the source to each wall.
           args:-gridparam (union(int, float)): if int, number of points to sample randomly on the unit sphere.
           if float, the angular resolution of the grid in degrees.
                -niter (int): number of iterations
                -tol (float): tolerance for the stopping criterion
                -verbose (bool): print the cost function value at each iteration
                -bvalues3d, bvalues2d (array): list of decreasing bandwidth values for the 3d and 2d optimization
        """
        if isinstance(gridparam, int):
            grid = sample_sphere(gridparam)  # sample ngrid points randomly in the unit sphere

        elif isinstance(gridparam, float):
            grid, _, _ = create_grid_spherical(1., 1., 0.1, gridparam, gridparam, cartesian=False)
            grid = grid[:len(grid) // 2+1, :]  # only use half of the sphere to use the central symmetry
        else:
            raise ValueError("gridparam should be an int or a float.")

        if bvalues3d is None:
            if verbose:
                print("Using default 3d bandwidth value.")
            bvalues = [1.]
        if bvalues2d is None:
            if verbose:
                print("Using default 2d bandwidth value.")
            bvalues2d = [1.]

        tstart = time.time()

        tc1 = time.time()
        # grid search for the initial guess
        costval = np.apply_along_axis(self.costfun, 1, grid[:, 1:], bvalues3d[0])
        tc2 = time.time()

        if verbose:
            print("time for costval : ", tc2-tc1)
        bestind = np.argmin(costval)
        u0, initval = grid[bestind][1:], costval[bestind]
        for k in range(len(bvalues3d)):  # loop over the decreasing bandwidth values
            bandwidth = bvalues3d[k]
            initval = self.costfun(u0, bandwidth)
            minimizer = jaxopt.ScipyMinimize(fun=self.costfun, method='BFGS', maxiter=niter,
                                             options={'gtol': tol})
            res = minimizer.run(u0, bandwidth)
            u0 = res.params

            if verbose:
                print("Minimizing for bandwidth ", bandwidth)
                print("First optimization converged in {} iterations".format(res.state.iter_num))
                print("Original and final cost function values: {} and {}".format(initval, res.state.fun_val))

        if plot:
            cart_grid = spherical_to_cartesian(np.ones(len(grid)), grid[:, 1], grid[:, 2]).T

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(121, projection='3d')
            t = ax.scatter(cart_grid[:, 0], cart_grid[:, 1], cart_grid[:, 2], c=np.array(costval), s=10)
            plt.colorbar(t)

            ax = fig.add_subplot(122, projection='3d')
            costval = np.apply_along_axis(self.costfun, 1, grid[:, 1:], bandwidth)
            t = ax.scatter(cart_grid[:, 0], cart_grid[:, 1], cart_grid[:, 2], c=np.array(costval), s=10)
            plt.colorbar(t)

            plt.show()

        basis = [spherical_to_cartesian(1, res.params[0], res.params[1])]

        # compute coordinates in an arbitrary basis of the normal plane
        rand_gen = np.random.RandomState(0)  # set seed to 0 to get consistent results
        # uniform random vector in the plane
        vec2 = rand_gen.normal(0, 1, size=3)
        vec2 /= np.linalg.norm(vec2)
        vec2 -= np.dot(vec2, basis[0]) * basis[0]  # project on the plane
        vec3 = np.cross(basis[0], vec2)  # vec2, vec3 is an orthonormal basis of the plane with normal basis[0]

        # project the image sources on vec2, vec3
        self.image_pos_transf = np.stack([np.dot(self.image_pos, vec2), np.dot(self.image_pos, vec3)], axis=-1)

        # convert to polar coordinates
        self.image_pos_transf = np.concatenate([np.linalg.norm(self.image_pos_transf, axis=1)[:, np.newaxis],
                                                np.arctan2(self.image_pos_transf[:, 1],
                                                           self.image_pos_transf[:, 0])[:, np.newaxis]], axis=1)
        self.cos_pos2 = np.cos(self.image_pos_transf[:, 1])
        self.sin_pos2 = np.sin(self.image_pos_transf[:, 1])

        # grid search for the initial guess over the half circle
        grid = np.linspace(0, np.pi, 4000).reshape([-1, 1])

        tc1 = time.time()
        costval = np.apply_along_axis(self.costfun2, 1, grid, bvalues2d[0])
        tc2 = time.time()

        if verbose:
            print("Time for costfun2: ", tc2 - tc1)

        bestind = np.argmin(costval)
        u0 = grid[bestind]
        for k in range(len(bvalues2d)):  # loop over the decreasing bandwidth values
            bandwidth = bvalues2d[k]
            initval = self.costfun2(u0, bandwidth)

            minimizer = jaxopt.ScipyMinimize(fun=self.costfun2, method='BFGS', maxiter=niter,
                                             options={'gtol': tol})
            res = minimizer.run(u0, bandwidth)

            u0 = res.params
            if verbose:
                print("Minimizing for bandwidth ", bandwidth)
                print("Second optimization converged in {} iterations".format(res.state.iter_num))
                print("Original and final cost function values: {} and {}".format(initval, res.state.fun_val))

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].plot(costval)
            costval = np.apply_along_axis(self.costfun2, 1, grid, bandwidth)
            ax[1].plot(costval)
            plt.show()

        basis.append(np.cos(res.params)*vec2 + np.sin(res.params)*vec3)
        basis[1] /= np.linalg.norm(basis[1])

        # basis.append(rot.inv().apply(np.array([np.cos(res.x[0]), np.sin(res.x[0]), 0.])))
        basis.append(np.cross(basis[0], basis[1]))
        basis = np.stack(basis, axis=0)

        tend = time.time()

        if verbose:
            print("Total time: {}".format(tend - tstart))

        # clear unused jax memory, slows exec down but avoid memory leak if running in a loop
        jax.clear_backends()

        return basis


class KernelFitter(RotationFitter):
    def __init__(self, image_pos):
        super().__init__(image_pos)
        self.pairwise_distances = np.linalg.norm(
            self.image_pos[:, np.newaxis, :] - self.image_pos[np.newaxis, :, :], axis=-1)
        self.pairwise_distances[self.pairwise_distances == 0.] = 1.  # avoid division by 0

    @partial(jax.jit, static_argnums=(0,))
    def gaussian_kernel(self, x, sig):
        return -jnp.exp(- x ** 2 / 2 / sig ** 2)

    @partial(jax.jit, static_argnums=(0,))
    def costfun(self, u, sig):
        """Return the cost function value for a given u."""
        dot_prod = self.compute_dot_prod(jnp.cos(u), jnp.sin(u))

        return jnp.sum(self.gaussian_kernel(jnp.abs(dot_prod[:, jnp.newaxis] -
                                            dot_prod[jnp.newaxis, :]) / self.pairwise_distances, sig)) / self.n_src

    @partial(jax.jit, static_argnums=(0,))
    def costfun2(self, u, sig):
        """Return the cost function value for a given u (second step in polar coordinates)."""
        dot_prod = self.compute_dot_prod_polar(jnp.cos(u), jnp.sin(u))

        return jnp.sum(self.gaussian_kernel(jnp.abs(dot_prod[:, jnp.newaxis] -
                                            dot_prod[jnp.newaxis, :]) / self.pairwise_distances, sig)) / self.n_src


def merge_clusters(clusters, cluster_sizes, labels, min_cluster_sep, method="merge", verbose=False):
    clusters, cluster_sizes, labels = sort_clusters(clusters, cluster_sizes, labels)
    nclusters = len(clusters)

    sep = False
    while nclusters > 3 and not sep:  # while there are more than 3 clusters and they are not separated
        diffs = clusters[1:] - clusters[:-1]
        if not np.all(diffs > min_cluster_sep):
            ind_del = np.argmin(diffs)
            if method == "merge":
                clusters[ind_del + 1] = ((clusters[ind_del + 1] * cluster_sizes[ind_del + 1] +
                                          clusters[ind_del] * cluster_sizes[ind_del]) /
                                         (cluster_sizes[ind_del+1] + cluster_sizes[ind_del]))
                cluster_sizes[ind_del+1] += cluster_sizes[ind_del]

                labels[labels == ind_del] = ind_del + 1
            elif method == 'delete':
                if cluster_sizes[ind_del] > cluster_sizes[ind_del + 1]:
                    ind_del = ind_del + 1
                labels[labels == ind_del] = -1
            else:
                raise ValueError("Unknown method: {}".format(method))

            if method == "merge" or method == "delete":
                # update the labels
                for j in range(ind_del+1, nclusters):
                    labels[labels == j] = j - 1
                if verbose:
                    print('Merging clusters {} and {}'.format(ind_del,
                                                              ind_del + 1))

                clusters = np.delete(clusters, ind_del)
                cluster_sizes = np.delete(cluster_sizes, ind_del)

                nclusters -= 1

        else:
            sep = True
    return clusters, cluster_sizes, labels


def calc_score(args):
    i, j, grid_left, grid_right, lb, rb, tol, center_ind, recentered_clusters, cluster_sizes, ncluster = args
    estimated_coord = generate_src_coordinates(grid_left[i], grid_right[j], lb, rb)
    if len(estimated_coord) > 0:
        ind_src, ind_clusters, dists = unique_matches(estimated_coord.reshape(-1, 1),
                                                      recentered_clusters.reshape(-1, 1))

        ind_close = ind_clusters[dists < tol]
        if center_ind in ind_close:  # don't count the center cluster
            ind_close = ind_close[ind_close != center_ind]
        score = np.sum(cluster_sizes[ind_close])   # clusters close to a source

        ind_far = np.arange(ncluster)[~np.in1d(np.arange(ncluster), ind_close)]
        if center_ind in ind_far:
            ind_far = ind_far[ind_far != center_ind]

        score -= np.sum(cluster_sizes[ind_far])  # penalize clusters far from a source
        # penalize if there are sources not close to a cluster
        score -= (len(estimated_coord) - len(ind_close))*cluster_sizes[center_ind]
        dist_score = -np.sum(dists)
        return i, j, score, dist_score
    else:
        return i, j, -np.inf, -np.inf


def compute_best_cluster(clusters, center_ind, cluster_sizes, tol, dx=0.02, plot=False):
    """Given sorted clusters and the index of the center cluster, return the index of the two order 1 clusters by
    computing a cost function on a grid of possible order 1 source coordinates."""
    ncluster = len(clusters)
    recentered_clusters = clusters - clusters[center_ind]
    lb, rb = recentered_clusters[0] - tol, recentered_clusters[-1] + tol

    if center_ind == 0 or center_ind == ncluster-1 or lb > - 1 or rb < 1:  # not enough clusters for recovery
        return np.nan, np.nan

    nright = int(np.ceil((rb/2 - 1)/dx))
    grid_right = np.linspace(1, rb/2, nright)
    nleft = int(np.ceil((np.abs(lb)/2 - 1)/dx))
    grid_left = np.linspace(1, np.abs(lb)/2, nleft)
    pool = mp.Pool(mp.cpu_count())
    args_list = [(i, j, grid_left, grid_right, lb, rb, tol, center_ind, recentered_clusters, cluster_sizes, ncluster)
                 for i in range(len(grid_left)) for j in range(len(grid_right))]
    results = pool.map(calc_score, args_list)
    results = np.array(results)

    best_score_ind = np.argmax(results[:, 2])
    best_score = results[best_score_ind, 2]
    matches = results[results[:, 2] == best_score]
    if len(matches) > 1:
        best_dist_score_ind = np.argmax(matches[:, 3])
        best_ind = (int(matches[best_dist_score_ind, 0]), int(matches[best_dist_score_ind, 1]))
    else:
        best_ind = (int(matches[0, 0]), int(matches[0, 1]))
    pool.close()

    ind_best = np.argmin(np.abs(recentered_clusters[:, np.newaxis] -
                                np.array([-grid_left[best_ind[0]]*2,
                                          grid_right[best_ind[1]]*2])[np.newaxis, :]), axis=0)

    if plot:
        plt.figure()
        plt.plot(recentered_clusters, cluster_sizes, 'o')
        plt.plot(recentered_clusters[center_ind], 0, 'o', color='r')
        plt.plot(recentered_clusters[[*ind_best]], np.zeros(2), 'x', color='g')
        gen_coord = generate_src_coordinates(np.abs(recentered_clusters[ind_best[0]] / 2),
                                             recentered_clusters[ind_best[1]] / 2, lb, rb)
        gen_coord2 = generate_src_coordinates(grid_left[best_ind[0]], grid_right[best_ind[1]]
                                              , lb, rb)
        plt.plot(gen_coord, np.zeros_like(gen_coord), 'x', color='r')
        plt.plot(gen_coord2, np.zeros_like(gen_coord2), 'x', color='b')
        print('Best score: {}'.format(best_score))
        plt.show()
    return ind_best


def find_dimensions_clusters(image_pos, basis, prune=0, max_dist=0.25, min_cluster_sep=1.5, threshold=0.1, max_iter=300,
                             src_pos=None, merge=True, post_process=True,
                             verbose=False, plot=False):
    """Find the dimensions of the room from the image positions and the (reconstructed) orthonormal basis of the room.
    The projections of the image positions on the basis are used to find the room dimensions by clustering on each
    axis, 'bandwidth' is the bandwidth of the clustering algorithm.
    If src_pos is not None, the order 1 image sources are found by taking the closest source (in the relevant cluster)
    to the order 1 cluster centers in each direction and are used to infer the room dimensions.
    If prune is True, the clusters are pruned by removing from a given axis cluster any source that is at a distance
    superior to 'max_dist' from the cluster centers on any axis. The clusters are then trimmed using the parameter
     'threshold' (see mean_shift_clustering), keeping a minimum of 3 clusters.
    """
    room_dim = np.zeros([2, 3])
    image_pos = image_pos.copy()
    image_pos_old = image_pos.copy()
    all_clusters, all_labels, projections, cluster_sizes = [], [], [], []
    # project the image positions on the basis, shape (3, n_sources)
    projections = np.sum(image_pos[np.newaxis, :, :] * basis[:, np.newaxis, :], axis=2)

    if plot:
        fig, ax = plt.subplots(3, 3, figsize=(15, 5))

    for i in range(3):  # generate the clusters
        # kmeans clustering, keep all non empty clusters
        clusters, cluster_size, labels = kmeans_clustering(projections[i], threshold=threshold,
                                                           niter=max_iter)
        all_clusters.append(clusters.flatten())
        all_labels.append(labels)
        cluster_sizes.append(cluster_size)

        if plot:
            ax[0, i].hist(projections[i], bins=100)
            ax[0, i].set_yticks(cluster_size)
            ax[0, i].plot(clusters, cluster_size, 'o')

    projections_old = projections.copy()

    old_nb_src = projections.shape[1]
    it, converged, nb_src = 0, False, old_nb_src
    ind_keep_old = np.arange(nb_src)
    t1 = time.time()
    while it < prune and not converged:  # prune the clusters, iterate 'prune' times
        it += 1
        dist_list = []
        for i in range(3):
            # check if smallest distance between projections and clusters is inferior to max dist, shape (n_sources, )
            dist_list.append(np.min(np.abs(projections[i, :, np.newaxis] -  # shape (n_sources, n_clusters)
                                           all_clusters[i][np.newaxis, :]), axis=1) <= max_dist)

        dist = np.array(dist_list)  # shape (3, n_sources)
        # find the indices for which projection is close to a cluster, shape (n_sources,)
        ind_keep = np.all(dist, axis=0)
        ind_keep_old = ind_keep_old[ind_keep]  # indexation of the remaining sources in the original array

        nb_src_new = np.sum(ind_keep)
        if nb_src == nb_src_new:  # check if stationary
            converged = True
        else:
            nb_src = nb_src_new

        projections = projections[:, ind_keep]
        image_pos = image_pos[ind_keep]
        all_clusters_old = all_clusters.copy()
        all_clusters, all_labels, cluster_sizes = [], [], []
        for i in range(3):  # recompute the clusters
            clusters, cluster_size, labels = kmeans_clustering(projections[i],
                                                               init=all_clusters_old[i], niter=max_iter,
                                                               threshold=threshold)
            all_clusters.append(clusters.flatten())
            all_labels.append(labels)
            cluster_sizes.append(cluster_size)

    if verbose:
        if converged:
            print('Converged after {} iterations'.format(it))
        else:
            print('Max number of iterations reached for pruning')

        print('Time for pruning: {}s'.format(time.time() - t1))
        print("Number of sources before and after pruning: {} -> {}".format(old_nb_src, nb_src))

    if plot:
        for i in range(3):
            ax[1, i].hist(projections[i], bins=100)
            ax[1, i].set_yticks(cluster_sizes[i])
            ax[1, i].plot(all_clusters[i], cluster_sizes[i], 'o')

    # approximated position of the source (order 0) using the clusters
    src_pos_est = np.zeros(3) if src_pos is None else src_pos.copy()
    src_masks = []
    for i in range(3):
        if merge:  # merge clusters that are too close together
            all_clusters[i], cluster_sizes[i], all_labels[i] = merge_clusters(all_clusters[i], cluster_sizes[i],
                                                                              all_labels[i], min_cluster_sep)

        else:
            all_clusters[i], cluster_sizes[i], all_labels[i] = sort_clusters(all_clusters[i], cluster_sizes[i],
                                                                             all_labels[i])

        if src_pos is None:  # get the closest source in the cluster to the cluster center and project it
            ind_mid = np.argmin(np.abs(all_clusters[i]))  # index might have shifted after trimming
            src_pos_est += all_clusters[i][ind_mid]*basis[i]
            src_masks.append(all_labels[i] == ind_mid)
        if len(all_clusters[i]) < 3:
            print('Warning: not enough clusters found for dimension recovery')

    # relabel the sources using the cluster positions, keep only those that verify the distance criterion
    all_dist_list, min_dist_list = [], []
    for i in range(3):
        all_dist_list.append(np.abs(projections_old[i, :, np.newaxis] -  # shape (n_sources, n_clusters)
                                    all_clusters[i][np.newaxis, :]))
        min_dist_list.append(np.min(all_dist_list[i], axis=1) <= max_dist)

    min_dist = np.array(min_dist_list)  # shape (3, n_sources)

    old_ind = np.zeros(len(image_pos_old), dtype=bool)
    old_ind[ind_keep_old] = True
    ind_keep = np.all(min_dist, axis=0) | old_ind

    image_pos, projections = image_pos_old[ind_keep], projections_old[:, ind_keep]

    for i in range(3):
        all_labels[i] = np.argmin(all_dist_list[i][ind_keep], axis=1)
        cluster_sizes[i] = np.bincount(all_labels[i], minlength=len(all_clusters[i]))

    if plot:
        for i in range(3):
            ax[2, i].hist(projections[i], bins=100)
            ax[2, i].plot(all_clusters[i], cluster_sizes[i], 'o')

        plt.show()

    if src_pos is None:
        src_pos_est = image_pos[np.argmin(np.sum((image_pos - src_pos_est)**2, axis=1))]

    for i in range(3):  # find the dimensions of the room in each direction
        clusters = all_clusters[i]

        if src_pos is None:  # use the cluster closest to 0 as reference
            ind_mid = np.argmin(np.abs(clusters))
        else:  # use the cluster closest to the projection of the source position as reference
            ind_mid = np.argmin(np.abs(clusters - np.dot(src_pos_est, basis[i])))

        if post_process and len(clusters) > 2 and ind_mid > 0:  # find the best clusters to use for dimension recovery
            ind_inf, ind_sup = compute_best_cluster(clusters, ind_mid, cluster_sizes[i], max_dist, plot=plot)
        else:
            if 0 < ind_mid < len(clusters) - 1:
                ind_inf, ind_sup = ind_mid - 1, ind_mid + 1
            else:
                ind_inf, ind_sup = np.nan, np.nan

        if len(clusters) > 2 and ind_inf is not np.nan and ind_sup is not np.nan:
            # center the clusters around the source position
            clusters = clusters - np.dot(src_pos_est, basis[i])
            order1_clusters = np.stack([clusters[ind_inf],
                                        clusters[ind_sup]])  # take the two closest clusters

            estimated_pos = src_pos_est[np.newaxis, :] + basis[i][np.newaxis, :] * order1_clusters[:, np.newaxis]

            # extract cluster mask for the two opposing walls
            mask_inf = all_labels[i] == ind_inf
            mask_sup = all_labels[i] == ind_sup

            # find the closest sources (in the relevant clusters) to the estimated positions
            # shape 2x(n_sources in cluster, 3)->(n_sources in cluster,)
            dists_inf = np.sum((image_pos[np.newaxis, mask_inf, :] -
                                estimated_pos[0, np.newaxis, :]) ** 2, axis=-1)
            dists_sup = np.sum((image_pos[np.newaxis, mask_sup, :] -
                                estimated_pos[1, np.newaxis, :]) ** 2, axis=-1)
            closest_src = np.concatenate([image_pos[mask_inf][np.argmin(dists_inf, axis=-1)],
                                          image_pos[mask_sup][np.argmin(dists_sup, axis=-1)]], axis=0)

            room_dim[:, i] = np.abs(np.sum((closest_src - src_pos_est[np.newaxis, :]) *
                                           basis[i][np.newaxis, :], axis=-1)).flatten()

        else:
            print('Warning: not enough clusters found for dimension recovery')
            room_dim[:, i] = np.nan

    return room_dim / 2., src_pos_est


def kernel_dot(a1, r1, a2, r2, kernel, kernel_scale):
    """Kernel dot product between two linear combinations of diracs."""
    distances2 = np.sum(jnp.square(r1[:, jnp.newaxis, :] - r2[jnp.newaxis, :, :]), axis=-1)  # shape (n1, n2)
    return np.sum(kernel(distances2, kernel_scale)*a1[:, jnp.newaxis]*a2[jnp.newaxis, :])


def gaussian_kernel(x, sigma=1.):
    """Gaussian kernel."""
    return np.exp(-x/(2*sigma**2))


def find_dimensions_ns(image_pos, basis, amplitudes, fusion=0.5, cone_width=10):
    """Try to find the room parameters using a cloud of reconstructed image sources and a vector basis normal to the
    walls. Try to identify the order 1 sources by searching from the source in each basis direction in a cone of width
    'cone_width' (in degrees).
    Args: - image_pos: positions of the image sources
          - basis: vector basis normal to the walls
          - amplitudes: amplitudes of the image sources in sparse reconstruction formulation
          - fusion: distance threshold to merge sources, set to 0 to consider only one source
          - cone_width: width of the search cones in degrees, is extended progressively if no source is found
    Return: - room_dim: estimated distances of the source to each wall, shape (2, 3)
             - src_pos: estimated source position, shape (3,)
             - src_ampl: estimated source amplitude (1 - src_ampl**2 gives the absorption coefficient)
             - order1_ampl: estimated order 1 source amplitudes, shape (3, 2) (each line corresponds to a normal)
             - order1_pos: estimated order 1 source positions, shape (6, 3) (order1_ampl.flatten() of shape (6,)
             follows the same order)
    """

    norms = np.linalg.norm(image_pos, axis=-1)
    src_ind = np.argmin(norms)
    src_pos = image_pos[src_ind]
    if fusion > 0:  # merge sources closer than fusion to the source
        dist_src = np.linalg.norm(image_pos[:, :]-np.reshape(src_pos, [-1, 3])[:, :], axis=-1)
        close = dist_src < fusion
        ampl_close = amplitudes[close]
        image_pos_close = image_pos[close]
        src_ampl = np.sum(ampl_close)
        src_pos = np.sum(image_pos_close*ampl_close[:, np.newaxis], axis=0) / src_ampl
        amplitudes, image_pos = amplitudes[~close], image_pos[~close]   # delete merged sources
    else:
        src_ampl = amplitudes[src_ind]
        image_pos, amplitudes = np.delete(image_pos, src_ind, axis=0), np.delete(amplitudes, src_ind, axis=0)

    projections = np.sum(image_pos[:, np.newaxis, :] * basis[np.newaxis, :, :], axis=2)  # shape: (nb_images, 3)
    src_pos_proj = np.sum(src_pos[np.newaxis, :] * basis, axis=1)  # shape: (3,)

    coord_est, dim_est, ampl_est = [], [], []
    for i in range(3):  # loop over the 3 dimensions
        lcone_is_empty, rcone_is_empty, tmp_width = True, True, cone_width
        while (lcone_is_empty or rcone_is_empty) and tmp_width < 180:
            # find indexes of sources located in the cone of width cone_width (in deg), direction i around the source
            ind_tunnel = (np.arccos(np.clip(np.abs(projections[:, i] - src_pos_proj[i]) /
                                            np.linalg.norm(image_pos - np.reshape(src_pos, [1, 3]), axis=-1), -1., 1.))
                          < np.deg2rad(tmp_width))

            # get nb_neigh nearest neighbors in each direction
            if lcone_is_empty:
                indm = projections[ind_tunnel][:, i] <= src_pos_proj[i]  # sources in the left cone
                neighm, amplm = image_pos[ind_tunnel][indm], amplitudes[ind_tunnel][indm]
                projm = projections[ind_tunnel][indm]
                if len(neighm) > 0:
                    lcone_is_empty = False
            if rcone_is_empty:
                indp = projections[ind_tunnel][:, i] > src_pos_proj[i]  # sources in the right cone
                neighp, amplp = image_pos[ind_tunnel][indp], amplitudes[ind_tunnel][indp]
                projp = projections[ind_tunnel][indp]
                if len(neighp) > 0:
                    rcone_is_empty = False

            tmp_width *= 1.25

        indsortp = np.argmin(np.linalg.norm(neighp - np.reshape(src_pos, [1, 3]), axis=-1))
        neighp, projp = neighp[indsortp], projp[indsortp]
        indsortm = np.argmin(np.linalg.norm(neighm - np.reshape(src_pos, [1, 3]), axis=-1))
        neighm, projm = neighm[indsortm], projm[indsortm]
        amplm, amplp = amplm[indsortm], amplp[indsortp]
        if fusion > 0:
            closem = np.linalg.norm(neighm - image_pos, axis=-1) <= fusion
            closep = np.linalg.norm(neighp - image_pos, axis=-1) <= fusion

            amplm_tmp = amplitudes[closem]
            amplm = np.sum(amplm_tmp)
            neighm = np.sum(image_pos[closem]*amplm_tmp[:, np.newaxis], axis=0) / amplm
            projm = np.sum(projections[closem][:, i]*amplm_tmp) / amplm
            amplp_tmp = amplitudes[closep]
            amplp = np.sum(amplp_tmp)
            neighp = np.sum(image_pos[closep]*amplp_tmp[:, np.newaxis], axis=0) / amplp
            projp = np.sum(projections[closep][:, i]*amplp_tmp) / amplp

        ampl_est.append([amplm, amplp]), coord_est.append(neighm), coord_est.append(neighp)
        dim_est.append([(src_pos_proj[i] - projm)/2, (projp - src_pos_proj[i])/2])

    dim_est, ampl_est, coord_est = np.array(dim_est), np.array(ampl_est), np.stack(coord_est, axis=0)
    return np.reshape(dim_est, [3, 2]).T, src_pos, src_ampl, np.reshape(ampl_est, [3, 2]), coord_est


if __name__ == "__main__":
    pass
