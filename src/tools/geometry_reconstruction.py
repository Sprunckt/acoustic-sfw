import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from src.simulation.utils import create_grid_spherical
from scipy.optimize import minimize
import multiprocessing as mp
import os
import time
from sklearn.cluster import MeanShift
from abc import ABC, abstractmethod

threshold = 1e-2
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


def same_half_space(x, y, normal, origin):
    return (np.dot(x, normal) - origin)*(np.dot(y, normal) - origin) > 0


def generate_src_coordinates(d1, d2, lower_bound, upper_bound):
    """Generate the coordinates of the image sources in one direction, given the distances of the source to the walls
    and bounds (assume the origin is located at the source)."""
    d = d1 + d2
    coord_list = []
    curr_coord, parity, k = 0, 1, 0
    while curr_coord > lower_bound:
        curr_coord = - 2*d*k - 2*parity*d2
        if curr_coord > lower_bound:
            coord_list.append(curr_coord)
        k += parity
        parity = (parity + 1) % 2

    curr_coord, parity, k = 0, 1, 0
    while curr_coord < upper_bound:
        curr_coord = 2*d*k + 2*parity*d1
        if curr_coord < upper_bound:
            coord_list.append(curr_coord)
        k += 1*parity
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


def find_order1(image_pos):
    """
    Simple algorithm to find the order 1 image sources from a cloud of image sources, by sorting the sources by distance
    and eliminating the points located in the half-spaces containing the order 1 sources found at the preceding
    iterations. It assumes the source and the order 1 image sources are present. Not robust to noise/false positives.
    """
    norms = np.linalg.norm(image_pos, axis=1)
    min_dist_ind = np.argmin(norms)
    source_pos = image_pos[min_dist_ind]  # find the source position as the closest source from the microphone
    image_sources = image_pos[np.arange(len(image_pos)) != min_dist_ind]  # image sources (removed the source)
    n_image_src = len(image_sources)
    dist = np.linalg.norm(source_pos[np.newaxis, :] - image_sources, axis=1)  # distances to source
    order1 = np.empty([6, 3], dtype=float)
    wall_pos = np.empty([6, 3], dtype=float)
    sorted_dist_ind = np.argsort(dist)

    closest_src = image_sources[sorted_dist_ind[0]]  # image source that is closest to the source (should be order 1)
    order1[0] = closest_src
    wall_pos[0] = (closest_src + source_pos) / 2.
    normal_vect, origins = np.zeros((6, 3)),  np.zeros(6)
    normal_vect[0] = closest_src - source_pos
    origins[0] = np.dot(wall_pos[0], normal_vect[0])

    it_dist, nb_found = 1, 1
    while it_dist < n_image_src and nb_found < 6:  # loop over the image sources until an order 1 is found
        found = True
        curr_source = image_sources[sorted_dist_ind[it_dist]]
        for k in range(nb_found):  # check if the point is in the same half-space as another order 1 source
            if same_half_space(order1[k], curr_source, normal_vect[k], origins[k]):
                found = False
                break
        if found:
            order1[nb_found] = curr_source
            wall_pos[nb_found] = (source_pos + order1[nb_found]) / 2.
            normal_vect[nb_found] = order1[nb_found] - source_pos
            origins[nb_found] = np.dot(wall_pos[nb_found], normal_vect[nb_found])
            nb_found += 1

        it_dist += 1
    return order1[:nb_found], wall_pos[:nb_found], source_pos


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


def mean_shift_clustering(data, bandwidth, niter=300, threshold=1, trim_factor=0.):
    """Mean shift clustering of 1D data using sklearn's implementation. Only return the centers of the clusters
    containing more than 'threshold' points. If trim_factor > 0, only return the clusters containing more than
    trim_factor * max(cluster_size) points."""
    data = data.reshape(-1, 1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1, max_iter=niter, cluster_all=False)
    ms.fit(data)
    cluster_size = np.bincount(ms.labels_[ms.labels_ >= 0])  # number of points in each cluster
    ind_keep = cluster_size > threshold
    cluster_centers, cluster_size = ms.cluster_centers_[ind_keep], cluster_size[ind_keep]
    if len(cluster_size) == 0:
        return np.array([]), np.array([])

    if trim_factor > 0:
        maxval = np.max(cluster_size)
        ind_keep = cluster_size > trim_factor * maxval

        if ind_keep.sum() < 3:
            ind_keep = np.argsort(cluster_size)[-3:]

    return cluster_centers[ind_keep], cluster_size[ind_keep]


def find_dim(clusters, zero_centering=True):
    """Find the dimensions of the room in a given direction using the coordinate clusters."""
    sorted_centers = np.sort(clusters.flatten())
    if zero_centering:  # use the cluster closest to 0 as reference
        ind_inf = np.min(np.argsort(np.abs(sorted_centers))[:3])
    else:
        ind_inf = 0
    diffs = sorted_centers[ind_inf+1:] - sorted_centers[ind_inf:-1]

    return diffs[:2].flatten()


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
    def costfun(self, u):
        pass

    @abstractmethod
    def costfun2(self, u):
        pass

    def compute_dot_prod(self, cos_u, sin_u):
        """Compute the dot product between u and the source positions."""
        return (self.cos_pos[:, 0] * self.sin_pos[:, 1] * cos_u[np.newaxis, 0] * sin_u[np.newaxis, 1] +
                self.sin_pos[:, 0] * self.sin_pos[:, 1] * sin_u[np.newaxis, 0] * sin_u[np.newaxis, 1] +
                self.cos_pos[:, 1] * cos_u[np.newaxis, 1]) * self.image_pos_transf[:, 0]

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

        p = mp.Pool(ncores)  # create a pool of workers for parallel processing

        # grid search for the initial guess
        costval = p.map(self.costfun, grid[:, 1:])

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.view_init(2, -89)

            cart_grid = spherical_to_cartesian(np.ones(len(grid)), grid[:, 1], grid[:, 2]).T
            t = ax.scatter(cart_grid[:, 0], cart_grid[:, 1], cart_grid[:, 2], c=np.array(costval), s=70)

            plt.colorbar(t)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_zticklabels([])
            plt.show()

        bestind = np.argmin(costval)
        u0, initval = grid[bestind][1:], costval[bestind]
        for k in range(len(bvalues)):  # loop over the decreasing bandwidth values
            self.bandwidth = bvalues[k]
            initval = self.costfun(u0)
            res = minimize(self.costfun, u0, method='BFGS', jac=self.costfun_grad,
                           options={'gtol': tol, 'maxiter': niter})
            u0 = res.x
            if verbose:
                print("Minimizing for bandwidth ", self.bandwidth)
                if res.success:
                    print("First optimization successful, converged in {} iterations".format(res.nit))
                else:
                    print("First optimization failed, {} iterations, reason: {}".format(res.nit, res.message))
                print("Original and final cost function values: {} and {}".format(initval, res.fun))

        basis = [spherical_to_cartesian(1, res.x[0], res.x[1])]

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
        grid = np.linspace(0, np.pi, 3000)
        self.bandwidth = bvalues[0]
        costval = p.map(self.costfun2, grid)
        p.close()

        if plot:
            plt.plot(costval)
            plt.show()

        bestind = np.argmin(costval)
        u0 = grid[bestind]
        for k in range(len(bvalues)):  # loop over the decreasing bandwidth values
            self.bandwidth = bvalues[k]
            initval = self.costfun2(u0)
            res = minimize(self.costfun2, u0, method='BFGS', jac=self.costfun_grad2,
                           options={'gtol': tol, 'maxiter': niter})
            u0 = res.x
            if verbose:
                if res.success:
                    print("Second optimization successful, converged in {} iterations".format(res.nit))
                else:
                    print("Second optimization failed, {} iterations, reason: {}".format(res.nit, res.message))
                print("Original and final cost function values: {} and {}".format(initval, res.fun))

        basis.append(np.cos(res.x)*vec2 + np.sin(res.x)*vec3)
        basis[1] /= np.linalg.norm(basis[1])

        # basis.append(rot.inv().apply(np.array([np.cos(res.x[0]), np.sin(res.x[0]), 0.])))
        basis.append(np.cross(basis[0], basis[1]))
        basis = np.stack(basis, axis=0)

        tend = time.time()

        if verbose:
            print("Total time: {}".format(tend - tstart))

        return basis


class KernelFitter(RotationFitter):
    def __init__(self, image_pos, kernel='gaussian'):
        super().__init__(image_pos)
        self.kernel, self.kernel_grad = self.get_kernel(kernel)
        self.pairwise_distances = np.linalg.norm(
            self.image_pos[:, np.newaxis, :] - self.image_pos[np.newaxis, :, :], axis=-1)
        self.pairwise_distances[self.pairwise_distances == 0.] = 1.  # avoid division by 0

    def gaussian_kernel(self, x):
        return -np.exp(- x ** 2 / 2 / self.bandwidth ** 2)

    def gaussian_kernel_grad(self, x):
        return x * np.exp(-x ** 2 / 2 / self.bandwidth ** 2) / self.bandwidth ** 2

    def gaussian_non_square_kernel(self, x):
        return -np.exp(- np.abs(x) / self.bandwidth)

    def inverse_kernel(self, x):
        return -1 / (1 + x ** 2 / self.bandwidth ** 2)

    def inverse_kernel_grad(self, x):
        return 2 * x / (1 + x ** 2 / self.bandwidth ** 2) ** 2 / self.bandwidth ** 2

    def get_kernel(self, kernel):
        if kernel == 'gaussian':
            return self.gaussian_kernel, self.gaussian_kernel_grad
        elif kernel == 'inverse':
            return self.inverse_kernel, self.inverse_kernel_grad
        elif kernel == 'gaussian_non_square':
            return self.gaussian_non_square_kernel, None
        else:
            raise ValueError("Kernel not recognized")

    def costfun(self, u):
        """Return the cost function value for a given u."""
        dot_prod = self.compute_dot_prod(np.cos(u), np.sin(u))
        return np.sum(self.kernel(np.abs(dot_prod[:, np.newaxis] -
                                         dot_prod[np.newaxis, :])/ self.pairwise_distances))/self.n_src

    def costfun2(self, u):
        """Return the cost function value for a given u (second step in polar coordinates)."""
        dot_prod = self.compute_dot_prod_polar(np.cos(u), np.sin(u))
        return np.sum(self.kernel(np.abs(dot_prod[:, np.newaxis] -
                                         dot_prod[np.newaxis, :]) / self.pairwise_distances)) / self.n_src


class HistogramFitter(RotationFitter):
    def __init__(self, image_pos, nbins):
        scale = np.max(np.linalg.norm(image_pos, axis=-1).reshape([-1, 1]))
        super().__init__(image_pos / scale)

        # bin parametrization
        self.nbins, self.bin_width = nbins, 2 / nbins
        self.bin_centers = -1. + (np.arange(nbins) + 0.5) * self.bin_width

    def histo_proba(self, x):
        """Return the vector of length len(bin_centers) giving the probability for data sampled in x to be in each
        bin."""
        return np.sum(expit((x[np.newaxis, :] - self.bin_centers[:, np.newaxis] + self.bin_width/2.) / self.bandwidth) -
                      expit((x[np.newaxis, :] - self.bin_centers[:, np.newaxis] - self.bin_width/2.) / self.bandwidth),
                      axis=1) / self.nbins

    def histo_proba_grad(self, dot_prod, cos_u, sin_u):
        # final shape: (n_sources, n_bins, 2)
        return np.sum(self.image_pos_transf[:, np.newaxis, np.newaxis, 0] *  # radius, shape: (n_sources,)
                      np.concatenate([self.sin_pos[:, np.newaxis, 0]*self.sin_pos[:, np.newaxis, 1] *
                                      cos_u[np.newaxis, 0]*sin_u[np.newaxis, 1] -

                                      sin_u[np.newaxis, 0]*sin_u[np.newaxis, 1] *
                                      self.cos_pos[:, np.newaxis, 0]*self.sin_pos[:, np.newaxis, 1],

                                      self.cos_pos[:, np.newaxis, 0]*self.sin_pos[:, np.newaxis, 1] *
                                      cos_u[np.newaxis, 0]*cos_u[np.newaxis, 1] +

                                      self.sin_pos[:, np.newaxis, 0]*self.sin_pos[:, np.newaxis, 1] *
                                      sin_u[np.newaxis, 0]*cos_u[np.newaxis, 1] -   # concat shape (n_sources, 2)
                                      self.cos_pos[:, np.newaxis, 1]*sin_u[np.newaxis, 1]], axis=-1)[:, np.newaxis, :] *
                      (sigmoid_der((dot_prod[:, np.newaxis, np.newaxis] - self.bin_centers[np.newaxis, :, np.newaxis] + self.bin_width / 2.)
                                   / self.bandwidth) -
                       sigmoid_der((dot_prod[:, np.newaxis, np.newaxis] - self.bin_centers[np.newaxis, :, np.newaxis] - self.bin_width / 2.)
                                   / self.bandwidth)),
                      axis=0) / self.nbins / self.bandwidth

    def histo_proba_grad2(self, dot_prod, cos_u, sin_u):
        # final shape: (n_sources, n_bins)
        return np.sum(self.image_pos_transf[:, np.newaxis, 0] *  # radius, shape: (n_sources,)
                      (cos_u*self.sin_pos[:, np.newaxis] - self.cos_pos[:, np.newaxis]*sin_u) *
                      (sigmoid_der((dot_prod[:, np.newaxis] - self.bin_centers[np.newaxis, :] + self.bin_width / 2.)
                                   / self.bandwidth) -
                       sigmoid_der((dot_prod[:, np.newaxis] - self.bin_centers[np.newaxis, :] - self.bin_width / 2.)
                                   / self.bandwidth)),
                      axis=0) / self.nbins / self.bandwidth

    def costfun(self, u):
        """Return the cost function value for a given u."""
        dot_prod = self.compute_dot_prod(np.cos(u), np.sin(u))  # dot products between u and the source positions
        proba = self.histo_proba(dot_prod)
        return - np.sum(proba*np.log(proba+1e-16))

    def costfun_grad(self, u):
        cos_u, sin_u = np.cos(u), np.sin(u)
        dot_prod = self.compute_dot_prod(cos_u, sin_u)  # dot products between u and the source positions
        return -np.sum(self.histo_proba_grad(dot_prod, cos_u, sin_u) *
                       (1 + np.log(1e-16+self.histo_proba(dot_prod))[:, np.newaxis]), axis=0)

    def costfun2(self, u):
        """Return the cost function value for a given u (second step in polar coordinates)."""
        dot_prod = self.compute_dot_prod_polar(np.cos(u), np.sin(u))
        proba = self.histo_proba(dot_prod)
        return - np.sum(proba*np.log(proba+1e-16))

    def costfun_grad2(self, u):
        cos_u, sin_u = np.cos(u), np.sin(u)
        dot_prod = self.compute_dot_prod_polar(cos_u, sin_u)
        return -np.sum(self.histo_proba_grad2(dot_prod, cos_u, sin_u) *
                       (1 + np.log(1e-16+self.histo_proba(dot_prod))))


def find_dimensions(image_pos, basis, plot=False):
    room_dim = np.zeros([2, 3])

    # find the room dimensions
    for i in range(3):
        x = np.sum(image_pos * basis[i][np.newaxis, :], axis=1)

        clusters, cluster_size = mean_shift_clustering(x, 1, threshold=1, trim_factor=1 / 3.)  # mean_shift_kneighbor(x)
        if plot:
            plt.hist(x, bins=100)
            plt.plot(clusters, cluster_size, 'o')
            plt.show()
        if len(clusters) > 0:
            room_dim[:, i] = find_dim(clusters)
        else:
            room_dim[:, i] = np.nan

    return room_dim / 2.


if __name__ == "__main__":
    pass
