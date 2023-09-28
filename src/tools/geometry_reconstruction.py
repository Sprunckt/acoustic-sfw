import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
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
from itertools import product
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


def image_source_score(image_pos, coord, penalization, tol):
    is_close = np.abs(image_pos[:, np.newaxis] - coord[np.newaxis, :]) < tol
    score = np.sum(is_close)
    score -= penalization*np.sum(np.any(~is_close, axis=0))

    return score


def simplify_hist(hist, ind_sep, it_max=None, window_size=3):
    """Extract bin spikes from an histogram array. Start by smoothing the histogram by applying a sliding window mean
    around each bin. At each iteration k clear a space of width 2*ind_sep around the k greatest remaining value.
    At the end the remaining array should not contain contiguous non-zero values."""

    hist = np.convolve(hist, np.ones(window_size), mode='same')

    if it_max is None:
        it_max = np.inf
    k, stat = 0, False
    while k < it_max and not stat:
        tmp = hist.copy()
        ind_sort = np.argsort(hist)[-1-k]
        max_val = hist[ind_sort]
        tmp[np.maximum(ind_sort-ind_sep, 0):ind_sort+ind_sep] = 0
        tmp[ind_sort] = max_val
        if np.all(tmp == hist):
            stat = True
        else:
            hist = tmp
            k += 1
    return hist


def find_order1(image_pos, amplitudes, fusion=0., tol=0.1):
    """
    Simple algorithm to find the order 1 image sources from a cloud of image sources, by sorting the sources by distance
    and eliminating the points located in the half-spaces containing the order 1 sources found at the preceding
    iterations. It assumes the source and the order 1 image sources are present. Not robust to noise/false positives.
    """
    norms = np.linalg.norm(image_pos, axis=1)
    min_dist_ind = np.argmin(norms)
    source_pos = image_pos[min_dist_ind]  # find the source position as the closest source from the microphone
    image_sources = image_pos[np.arange(len(image_pos)) != min_dist_ind]  # image sources (removed the source)
    amplitudes = amplitudes[np.arange(len(image_pos)) != min_dist_ind]
    n_image_src = len(image_sources)
    dist = np.linalg.norm(source_pos[np.newaxis, :] - image_sources, axis=1)  # distances to source
    order1 = np.empty([6, 3], dtype=float)
    wall_pos = np.empty([6, 3], dtype=float)
    ampl_order1 = np.empty(6, dtype=float)
    sorted_dist_ind = np.argsort(dist)

    closest_src = image_sources[sorted_dist_ind[0]]  # image source that is closest to the source (should be order 1)
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
        curr_source = image_sources[sorted_dist_ind[it_dist]]
        for k in range(nb_found):  # check if the point is in the same half-space as another order 1 source
            if same_half_space(order1[k], curr_source, normal_vect[k], origins[k], tol=tol):
                found = False
                break
        if found:
            if fusion > 0:  # merge sources at dist < fusion of curr_source, average by amplitudes
                is_close = np.linalg.norm(curr_source - image_sources, axis=1) < fusion
                ampl_order1[nb_found] = np.sum(amplitudes[is_close])
                order1[nb_found] = (np.sum(image_sources[is_close] * amplitudes[is_close, np.newaxis], axis=0) /
                                    ampl_order1[nb_found])
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
    return order1[:nb_found], wall_pos[:nb_found], source_pos, ampl_order1[:nb_found]


def hough_transform(image_pos, dphi, dtheta, thresh):
    """Fill an accumulator acc(theta, phi) with the number of points contained in the corresponding plan passing
    through origin. A point is considered to be contained in the plane if its distance to the plane is inferior to
    'thresh'."""
    theta_bins = np.arange(0, np.pi + dtheta, dtheta)
    phi_bins = np.arange(-np.pi / 2, np.pi / 2, dphi)
    ntheta = len(theta_bins)
    normal_vectors = np.stack([np.sin(phi_bins[:, np.newaxis]) * np.cos(theta_bins[np.newaxis, :]),
                               np.sin(phi_bins[:, np.newaxis]) * np.sin(theta_bins[np.newaxis, :]),
                               np.repeat(np.cos(phi_bins), ntheta).reshape(-1, ntheta)], axis=-1)

    acc = np.zeros_like(normal_vectors[:, :, 0])

    for k in range(len(image_pos)):
        # using Hesse normal form to update the accumulator
        acc[np.abs(np.einsum("ijk,k->ij", normal_vectors, image_pos[k])) < thresh] += 1

    return acc, theta_bins, phi_bins


def find_best_planes(source, image_pos, dphi, dtheta, thresh, nplanes=2, excl_phi=np.pi/45, excl_theta=np.pi/45,
                     plot=False, verbose=False, degrees=False):
    """Return the coordinates of an orthonormal basis passing by the three planes that contain the source and most of
    the image sources positions in cartesian coordinates."""

    if degrees:
        dtheta = degrees_to_rad(dtheta)
        dphi = degrees_to_rad(dphi)
        excl_phi = degrees_to_rad(excl_phi)
        excl_theta = degrees_to_rad(excl_theta)

    exclusion_length_theta = int(np.rint(excl_theta/dtheta))
    exclusion_length_phi = int(np.rint(excl_phi/dphi))

    # translating the coordinates to use the source as the new origin
    image_pos = image_pos - source

    acc, theta_bins, phi_bins = hough_transform(image_pos, dphi, dtheta, thresh)

    old_acc = acc.copy() if plot else None

    res = []
    for i in range(nplanes):
        best_ind = np.argsort(acc.reshape(-1))[-1]
        ind = np.unravel_index(best_ind, shape=np.shape(acc))
        theta, phi = theta_bins[ind[1]], phi_bins[ind[0]]
        res.append((theta, phi))
        acc[np.maximum(ind[0] - exclusion_length_theta, 0):ind[0] + exclusion_length_theta,
            ind[1] - exclusion_length_phi:ind[1] + exclusion_length_phi] = 0.

    res = [spherical_to_cartesian(1., el[0], el[1]) for el in res]
    if nplanes == 2:
        coord1, coord2 = res[:2]
        if verbose:
            print("Dot product between the first two vectors : ", np.dot(coord1, coord2))

        coord3 = np.cross(coord1, coord2)
        basis = np.stack([coord1, coord2, coord3])
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ims = ax.imshow(old_acc, extent=(0, np.pi, -np.pi / 2, np.pi / 2), origin='lower')
            fig.colorbar(ims)

            ax = fig.add_subplot(132, projection="3d")
            coord1p, coord2p = 10 * coord1, 10 * coord2

            ax.scatter(image_pos[:, 0], image_pos[:, 1], image_pos[:, 2])
            ax.scatter(0, 0, 0, color='red')
            ax.plot3D([0., coord1p[0]], [0., coord1p[1]], [0., coord1p[2]])
            ax.plot3D([0., coord2p[0]], [0., coord2p[1]], [0., coord2p[2]])

            ax = fig.add_subplot(133, projection="3d")
            transpos = image_pos @ basis.T
            ax.scatter(transpos[:, 0], transpos[:, 1], transpos[:, 2])
            ax.scatter(0, 0, 0, color='red')
            ax.plot3D([0., 1], [0., 0], [0., 0])
            ax.plot3D([0., 0], [0., 1], [0., 0])
            ax.plot3D([0., 0], [0., 0], [0., 1])
            plt.show()
    else:
        basis = np.stack(res)

    if plot and nplanes != 2:
        print("plot not supported for nplanes != 2")

    return basis


def pos_to_dim(full_image_pos, dx=0.02, sep=1., min_wall_dist=0.5, max_room_dim=15., plot=False):
    """Return the clusters of coordinate-wise distances between the image sources (source included).
    This assumes the coordinates are given in the referencial of the rectangular room (ie the basis vectors follow the
    directions of the room walls. Two
    distances are in the same cluster if they differ by less than 'thresh'."""

    diffs = []
    for i in range(3):
        tmp = full_image_pos[:, np.newaxis, i] - full_image_pos[np.newaxis, :, i]
        diffs.append(tmp[tmp > 0].flatten())

    diffs = np.stack(diffs, axis=-1) / 2.
    ind_sep = int(np.rint(sep / dx))  # minimum index separation between two spikes
    thresh_factor = 2

    bins = np.arange(min_wall_dist, max_room_dim + dx, dx)
    dims, dists = np.zeros(3), np.zeros(3)
    for i in range(3):
        # compute the coordinate distance histogram
        tmp, _ = np.histogram(diffs[:, i], bins=bins)

        # delete the values under a relative threshold
        thresh = np.max(tmp) / thresh_factor
        tmp[tmp < thresh] = 0

        # extract spikes
        tmp = simplify_hist(tmp, ind_sep, it_max=None, window_size=3)

        best_ind = np.sort(np.where(tmp > 0)[0])[:3]

        d1, dd = bins[best_ind][:2]

        is_pos = np.sum(np.abs(full_image_pos[:, i] - 2*d1) < dx) > np.sum(np.abs(full_image_pos[:, i] + 2*d1) < dx)

        if is_pos:  # check if d1 is the distance to the wall with positive coordinate
            # model where dd=d2
            coord1 = generate_src_coordinates(d1, dd, lower_bound=np.min(full_image_pos[:, i]),
                                              upper_bound=np.max(full_image_pos[:, i]))

            # model where dd=dim (d1=d2)
            coord2 = generate_src_coordinates(d1, dd - d1, lower_bound=np.min(full_image_pos[:, i]),
                                              upper_bound=np.max(full_image_pos[:, i]))
        else:
            coord1 = generate_src_coordinates(dd, d1, lower_bound=np.min(full_image_pos[:, i]),
                                              upper_bound=np.max(full_image_pos[:, i]))
            coord2 = generate_src_coordinates(dd - d1, d1, lower_bound=np.min(full_image_pos[:, i]),
                                              upper_bound=np.max(full_image_pos[:, i]))

        # check which configuration is more likely
        penalization = 1  # penalization factor of false negatives
        best_model = np.argmax([image_source_score(full_image_pos[:, i], coord1, penalization=penalization, tol=dx),
                                image_source_score(full_image_pos[:, i], coord2, penalization=penalization, tol=dx)])

        dim = dd + d1 if (best_model == 0) else dd
        if not is_pos:  # update d1 to be the distance to the wall of positive coordinate
            d1 = dim - d1
        dims[i] = (dim + dx / 2)   # thresh/2 offset relative to bin western edge
        dists[i] = (d1 + dx / 2)

    if plot:
        fig, ax = plt.subplots(1, 3)
        for i in range(3):
            ax[i].hist(diffs[:, i], bins=bins)
        plt.show()

    return dims, dists


def sigmoid_der(x):
    return np.exp(-x) * expit(x)**2


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
    def __init__(self, image_pos):
        self.bandwidth = 1.
        self.image_pos = image_pos.copy()

        self.norms = np.linalg.norm(self.image_pos, axis=-1)
        self.n_src = len(self.image_pos)
        self.image_pos_transf = cartesian_to_spherical(self.image_pos)
        self.sin_pos, self.cos_pos = np.sin(self.image_pos_transf[:, 1:]), np.cos(self.image_pos_transf[:, 1:])
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
        return self.image_pos_transf[:, 0] * (cos_u*self.cos_pos + sin_u*self.sin_pos)

    def fit(self, gridparam=2000, niter=10000, tol=1e-10, bvalues=None, verbose=False, plot=False):
        """Reconstruct the distance of the source to each wall.
           args:-method (str): 'distance' or 'histogram' to use the distance or histogram cost function.
                -gridparam (union(int, float)): if int, number of points to sample randomly on the unit sphere.
           if float, the angular resolution of the grid in degrees.
                -niter (int): number of iterations
                -tol (float): tolerance for the stopping criterion
                -verbose (bool): print the cost function value at each iteration
        """
        if isinstance(gridparam, int):
            grid = sample_sphere(gridparam)  # sample ngrid points randomly in the unit sphere

        elif isinstance(gridparam, float):
            grid, _, _ = create_grid_spherical(1., 1., 0.1, gridparam, gridparam, cartesian=False)
            grid = grid[:len(grid) // 2+1, :]  # only use half of the sphere to use the central symmetry
        else:
            raise ValueError("gridparam should be an int or a float.")

        if bvalues is None:
            if verbose:
                print("Using default bandwidth value.")
            bvalues = [1.]

        self.bandwidth = bvalues[0]

        tstart = time.time()

        tc1 = time.time()
        # grid search for the initial guess
        costval = np.apply_along_axis(self.costfun, 1, grid[:, 1:], bvalues[0])
        tc2 = time.time()

        if verbose:
            print("time for costval : ", tc2-tc1)
        bestind = np.argmin(costval)
        u0, initval = grid[bestind][1:], costval[bestind]
        for k in range(len(bvalues)):  # loop over the decreasing bandwidth values
            bandwidth = bvalues[k]
            initval = self.costfun(u0, bandwidth)
            minimizer = jaxopt.ScipyMinimize(fun=self.costfun, method='BFGS', maxiter=niter,
                                             options={'gtol': tol})
            res = minimizer.run(u0, bandwidth)
            u0 = res.params

            if verbose:
                print("Minimizing for bandwidth ", self.bandwidth)
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
        rand_gen = np.random.RandomState(0)
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
        self.cos_pos = np.cos(self.image_pos_transf[:, 1])
        self.sin_pos = np.sin(self.image_pos_transf[:, 1])

        # grid search for the initial guess over the half circle
        grid = np.linspace(0, np.pi, 4000).reshape([-1, 1])
        self.bandwidth = bvalues[0]

        tc1 = time.time()
        costval = np.apply_along_axis(self.costfun2, 1, grid, bvalues[0])
        tc2 = time.time()

        if verbose:
            print("Time for costfun2: ", tc2 - tc1)

        bestind = np.argmin(costval)
        u0 = grid[bestind]
        for k in range(len(bvalues)):  # loop over the decreasing bandwidth values
            bandwidth = bvalues[k]
            initval = self.costfun2(u0, bandwidth)

            minimizer = jaxopt.ScipyMinimize(fun=self.costfun2, method='BFGS', maxiter=niter,
                                             options={'gtol': tol})
            res = minimizer.run(u0, bandwidth)

            u0 = res.params
            if verbose:
                print("Minimizing for bandwidth ", self.bandwidth)
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


def find_dimensions(image_pos, basis, prune=0, max_dist=0.25, min_cluster_sep=1.5, threshold=0.1, max_iter=300,
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


def measure_norm2(a1, r1, a2, r2, kernel, kernel_scale):
    """Measure of the norm of a linear combination of diracs."""
    return (kernel_dot(a1, r1, a1, r1, kernel, kernel_scale) +
            kernel_dot(a2, r2, a2, r2, kernel, kernel_scale) -
            2*kernel_dot(a1, r1, a2, r2, kernel, kernel_scale))


def generate_src_ind(kmax):
    """Generate the coordinates of the image sources in one direction, given the distances of the source to the walls
    and bounds (assume the origin is located at the source)."""
    coord_list = []

    for k in range(kmax):
        coord_list.append((k, k))
        coord_list.append((-k, -k))
        coord_list.append((-k-1, -k))
        coord_list.append((k, k+1))

    return np.array(coord_list[1:-2])


def gaussian_kernel(x, sigma=1.):
    """Gaussian kernel."""
    return np.exp(-x/(2*sigma**2))


def coordinates_from_index(indexes, basis, src_pos, dim):
    """Compute the coordinates of the image sources from the indexes and the basis."""
    return (src_pos[np.newaxis, :] +
            2*np.sum(np.sum(indexes[:, :, :]*dim[np.newaxis, :, :],
                            axis=-1)[:, :, np.newaxis]*basis[np.newaxis, :, :], axis=1))


def find_dimensions_ns(image_pos, basis, ball_size, kernel, kernel_scale, amplitudes, fusion=0.,
                       nb_neigh=1, cone_width=10, use_ampl=False, verbose=False):
    """Try to find the room parameters using a cloud of reconstructed image sources and a vector basis normal to the
    walls. Try to identify the order 1 sources by searching from the source in each basis direction in a cone of width
    'cone_width' (in degrees). The 'nb_neigh' closest candidates are kept, and the best combination of supposed order 1
    sources is found by comparing the reconstructed 'image_pos' to the corresponding exact image sources distribution.
    The comparison is made using a kernel norm between the two dirac measures clouds.
    Returns: (dimensions, source_pos, ampl1) where dimensions contains the distance of the source to the walls (shape
    (2, 3)), source_pos is the size 3 array of source coordinates, ampl1 contains the reflection coefficients of the
    walls
    """

    norms = np.linalg.norm(image_pos, axis=-1)
    # image_pos, norms = image_pos[norms <= ball_size], norms[norms <= ball_size]
    src_ind = np.argmin(norms)
    src_pos = image_pos[src_ind]
    # remove sources closer than min_sep to the source
    dist_src = np.linalg.norm(image_pos[:, :]-np.reshape(src_pos, [-1, 3])[:, :], axis=-1)
    del_mask = (0 < dist_src) & (dist_src < fusion)
    image_pos, amplitudes = image_pos[~del_mask, :], amplitudes[~del_mask]
    # remove source from image positions
    norms = norms[~del_mask]
    src_ind = np.argmin(norms)

    image_posd = np.delete(image_pos, src_ind, axis=0)

    projections = np.sum(image_pos[:, np.newaxis, :] * basis[np.newaxis, :, :], axis=2)  # shape: (nb_images, 3)
    src_pos_proj = projections[src_ind]
    # remove the source
    projections, amplitudes_d = np.delete(projections, src_ind, axis=0), np.delete(amplitudes, src_ind, axis=0)

    best_cost = np.inf
    best_guess, best_ampl, best_src = None, None, None
    coord_neigh, dim_neigh, ampl_neigh = [], [], []
    for i in range(3):  # loop over the 3 dimensions
        lcone_is_empty, rcone_is_empty, tmp_width = True, True,cone_width
        while (lcone_is_empty or rcone_is_empty) and tmp_width < 180:
            # find indexes of sources located in the cone of width cone_width (in deg), direction i around the source
            ind_tunnel = (np.arccos(np.clip(np.abs(projections[:, i] - src_pos_proj[i]) /
                                            np.linalg.norm(image_posd - np.reshape(src_pos, [1, 3]), axis=-1), -1., 1.))
                          < np.deg2rad(tmp_width))

            # get nb_neigh nearest neighbors in each direction
            if lcone_is_empty:
                indm = projections[ind_tunnel][:, i] <= src_pos_proj[i]  # sources in the left cone
                neighm, amplm = image_posd[ind_tunnel][indm], amplitudes_d[ind_tunnel][indm]
                projm = projections[ind_tunnel][indm]
                if len(neighm) > 0:
                    lcone_is_empty = False
            if rcone_is_empty:
                indp = projections[ind_tunnel][:, i] > src_pos_proj[i]  # sources in the right cone
                neighp, amplp = image_posd[ind_tunnel][indp], amplitudes_d[ind_tunnel][indp]
                projp = projections[ind_tunnel][indp]
                if len(neighp) > 0:
                    rcone_is_empty = False

            tmp_width *= 1.25

        indsortp = np.argsort(np.linalg.norm(neighp - np.reshape(src_pos, [1, 3]), axis=-1))
        neighp, projp = neighp[indsortp][:nb_neigh], projp[indsortp][:nb_neigh, i]
        indsortm = np.argsort(np.linalg.norm(neighm - np.reshape(src_pos, [1, 3]), axis=-1))
        neighm, projm = neighm[indsortm][:nb_neigh], projm[indsortm][:nb_neigh, i]
        amplm, amplp = amplm[indsortm][:nb_neigh], amplp[indsortp][:nb_neigh]
        if fusion > 0:
            closem = np.linalg.norm(neighm[np.newaxis, :, :] - image_posd[:, np.newaxis, :], axis=-1) <= fusion
            closep = np.linalg.norm(neighp[np.newaxis, :, :] - image_posd[:, np.newaxis, :], axis=-1) <= fusion

            for j in range(len(neighm)):
                amplm_tmp = amplitudes_d[closem[:, j]]
                amplm[j] = np.sum(amplm_tmp)
                neighm[j] = np.sum(image_posd[closem[:, j]]*amplm_tmp[:, np.newaxis], axis=0) / amplm[j]
                projm[j] = np.sum(projections[closem[:, j]][:, i]*amplm_tmp) / amplm[j]
            for j in range(len(neighp)):
                amplp_tmp = amplitudes_d[closep[:, j]]
                amplp[j] = np.sum(amplp_tmp)
                neighp[j] = np.sum(image_posd[closep[:, j]]*amplp_tmp[:, np.newaxis], axis=0) / amplp[j]
                projp[j] = np.sum(projections[closep[:, j]][:, i]*amplp_tmp) / amplp[j]

        ampl_neigh.append([amplm, amplp])
        coord_neigh.append([neighm, neighp])
        dim_neigh.append([(src_pos_proj[i] - projm)/2, (projp - src_pos_proj[i])/2])

    if use_ampl:
        ampl_recons = amplitudes
    else:
        ampl_recons = np.ones(len(image_pos))

    def costfun(x, a):
        dist = x.reshape([3, 2])
        dim = np.sum(dist, axis=1)
        a = a.reshape([3, 2])
        max_num_src = []
        for j in range(3):
            max_num_src.append(int(np.maximum((ball_size - src_pos[j] + 2 * dist[j, 0]) / (2 * dim[j]),
                                              (ball_size + src_pos[j] + 2 * dist[j, 1]) / (2 * dim[j]))) + 1)

        indexes = [generate_src_ind(max_num_src[i]) for i in range(3)]  # generate the indexes of the image sources
        indexes_full = np.array(list(product(indexes[0], indexes[1], indexes[2])))


        full_src = coordinates_from_index(indexes_full, basis, src_pos, x.reshape([3, 2]))
        ind_keep = np.linalg.norm(full_src, axis=1) <= ball_size

        full_src = full_src[ind_keep, :]
        if use_ampl:
            ampl_fact = [np.power(a[i][0], np.abs(indexes[i][:, 0])) * np.power(a[i][1], np.abs(indexes[i][:, 1]))
                         for i in range(3)]
            full_ampl = np.prod(np.array(list(product(ampl_fact[0], ampl_fact[1], ampl_fact[2]))), axis=1)
            full_ampl = full_ampl[ind_keep]
        else:
            full_ampl = np.ones(len(full_src))

        return (measure_norm2(full_ampl, full_src,
                              ampl_recons, image_pos, kernel, kernel_scale))

    td = time.time()
    if nb_neigh > 1:
        # explore all the possible combinations of the n nearest neighbors, and keep the best one
        for i in product(range(len(dim_neigh[0][0])), range(len(dim_neigh[0][1]))):
            for j in product(range(len(dim_neigh[1][0])), range(len(dim_neigh[1][1]))):
                for k in product(range(len(dim_neigh[2][0])), range(len(dim_neigh[2][1]))):
                    guess = np.array([dim_neigh[0][0][i[0]], dim_neigh[0][1][i[1]],
                                      dim_neigh[1][0][j[0]], dim_neigh[1][1][j[1]],
                                      dim_neigh[2][0][k[0]], dim_neigh[2][1][k[1]]])
                    guess_ampl = np.array([ampl_neigh[0][0][i[0]], ampl_neigh[0][1][i[1]],
                                          ampl_neigh[1][0][j[0]], ampl_neigh[1][1][j[1]],
                                          ampl_neigh[2][0][k[0]], ampl_neigh[2][1][k[1]]])
                    res = costfun(guess, guess_ampl)
                    if res < best_cost:
                        best_cost = res
                        best_guess = guess
                        best_ampl = guess_ampl
                        best_src = np.stack([coord_neigh[0][0][i[0]], coord_neigh[0][1][i[1]],
                                             coord_neigh[1][0][j[0]], coord_neigh[1][1][j[1]],
                                             coord_neigh[2][0][k[0]], coord_neigh[2][1][k[1]]], axis=0)
    else:
        best_cost = "not computed"
        best_guess = np.array([dim_neigh[0][0][0], dim_neigh[0][1][0],
                               dim_neigh[1][0][0], dim_neigh[1][1][0],
                               dim_neigh[2][0][0], dim_neigh[2][1][0]])
        best_ampl = np.array([ampl_neigh[0][0][0], ampl_neigh[0][1][0],
                              ampl_neigh[1][0][0], ampl_neigh[1][1][0],
                              ampl_neigh[2][0][0], ampl_neigh[2][1][0]])
        best_src = np.stack([coord_neigh[0][0][0], coord_neigh[0][1][0],
                             coord_neigh[1][0][0], coord_neigh[1][1][0],
                             coord_neigh[2][0][0], coord_neigh[2][1][0]], axis=0)

    if verbose:
        print("time for comparison: {}".format(time.time() - td))
        if best_guess is None:
            print("no solution found")
            best_guess = np.nan * np.ones(6)
        else:
            print("cost: {}".format(best_cost))
            print("dimensions: {}".format(best_guess.reshape([3, 2])))

    return np.reshape(best_guess, [3, 2]).T, src_pos, np.reshape(best_ampl, [3, 2]), best_src


if __name__ == "__main__":
    pass
