import numpy as np
import matplotlib.pyplot as plt
threshold = 1e-2


def spherical_to_cartesian(r, theta, phi):
    return np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])


def degrees_to_rad(alpha):
    return np.pi * alpha / 180


def same_half_space(x, y, normal, origin):
    return (np.dot(x, normal) - origin)*(np.dot(y, normal) - origin) > 0


def find_order1(image_pos):
    """
    Simple algorithm to find the order 1 image sources from a cloud of image sources, by sorting the sources by distance
    and eliminating the points located in the half-spaces containing the order 1 sources found at the preceding
    iterations. It assumes the source and the order 1 image sources are present. Not robust to noise/false positives.
    """
    norms = np.linalg.norm(image_pos, axis=1)
    min_dist_ind = np.argmin(norms)
    source_pos = image_pos[min_dist_ind]
    image_sources = image_pos[np.arange(len(image_pos)) != min_dist_ind]  # image sources (removed the source)
    n_image_src = len(image_sources)
    dist = np.linalg.norm(source_pos[np.newaxis, :] - image_sources, axis=1)  # distances to source
    order1 = np.empty([6, 3], dtype=float)
    wall_pos = np.empty([6, 3], dtype=float)
    sorted_dist_ind = np.argsort(dist)

    closest_src = image_sources[sorted_dist_ind[0]]  # image source that is closest to the source
    order1[0] = closest_src
    wall_pos[0] = (closest_src + source_pos) / 2.
    normal_vect, origins = np.zeros((6, 3)),  np.zeros(6)
    normal_vect[0] = closest_src - source_pos
    origins[0] = np.dot(wall_pos[0], normal_vect[0])

    it_dist, nb_found = 1, 1
    while it_dist < n_image_src and nb_found < 6:  # loop over the image sources until an order 1 is found
        found = True
        curr_source = image_sources[sorted_dist_ind[it_dist]]
        for k in range(nb_found):  # check if the point is in the same half-space as another order1 source
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
    return order1[:nb_found], wall_pos[:nb_found]


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

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ims = ax.imshow(acc, extent=(0, np.pi, -np.pi / 2, np.pi / 2), origin='lower')
        fig.colorbar(ims)
        plt.show()
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


if __name__ == "__main__":
    pass
