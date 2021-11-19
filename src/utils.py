import numpy as np
import matplotlib.pyplot as plt

tol = 1e-10
c = 343


def multichannel_rir_to_vec(rir: list, start_offset: int = 40, source: int = 0) -> (np.ndarray, int, int):
    """
    Convert a list of RIRs of length N corresponding to M microphones to a single flat array of length J=N*M.

    Args:
        -rir (list) : list of length M containing the RIRs associated with each microphone. rir[i] contains a list of
    the RIRs corresponding to the microphone i.
        -start_offset (int) : time step marking the beginning of each RIR
        -source (int) : index of the source considered for every microphone
    Return: tuple res, N, M where :
        -N, M are the length of each RIR and the number of microphone (every RIR is extended to have the same length)
        -res is a flat array (shape (M*N,) containing every RIR in a sequence, res[N*i] marks the beginning of the RIR
    for the ith microphone.
    """

    M = len(rir)
    N = len(rir[0][source][start_offset:])

    # find the longest RIR
    for i in range(M):
        NN = len(rir[i][source][start_offset:])
        if N < NN:
            N = NN
    res = np.empty(N*M)

    for i in range(M):
        local_size = len(rir[i][source][start_offset:])
        buffer = np.zeros(N)
        buffer[:local_size] = rir[i][source][start_offset:]
        res[i*N:(i+1)*N] = buffer

    return res, N, M


def vec_to_rir(rir_vec, m, N):
    """
    Return the mth RIR of length N from a vector containing a multichannel RIR.
    Args:
        -m (int) : the index of the microphone
        -N (int) : the length of a single RIR (all RIRs are expected to have the same length)
    """
    return rir_vec[m*N:(m + 1)*N]


def create_grid(xmin, xmax, ymin, ymax, zmin, zmax, N, flat=True):
    """
    Create a cartesian grid according to the given parameters.
    Args:
        -N (int or float) : if an int, the number of discretization points in each direction. If a float, the
    step used to discretize in every direction.
        -flat (bool) : if True, return an array of shape (nb_points, 3). Else, return a 3 tuple x, y, z where x,y,z are
    3d tensors.
    """

    if N != int(N):
        Nx = int((xmax - xmin) / N)
        Ny = int((ymax - ymin) / N)
        Nz = int((zmax - zmin) / N)
    else:
        Nx, Ny, Nz = N, N, N

    x_ = np.linspace(xmin, xmax, Nx)
    y_ = np.linspace(ymin, ymax, Ny)
    z_ = np.linspace(zmin, zmax, Nz)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    if flat:
        return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)
    else:
        return x, y, z


def plot_3d_planes(fun, slices, other_bounds, N, axis=0):
    """Plot a 3d function on the planes contained by a cartesian grid.
    Args:
        -fun (callable) : a function of 3 arguments implemented as a single argument callable
        -slices (list) : list containing the values defining each slicing plane
        -other_bounds (tuple) : a tuple or list containing the bounds for the remaining variables, e.g
    (xmin, xmax, ymin, ymax) if slicing along the z axis
        -N (int) : number of points for the linspace in the non slicing directions
        -axis (int) : axis orthogonal to the slicing plane
    """

    ax1 = np.linspace(other_bounds[0], other_bounds[1], N)
    ax2 = np.linspace(other_bounds[2], other_bounds[3], N)
    ax1, ax2 = np.meshgrid(ax1, ax2)
    ax1, ax2 = ax1.reshape(-1, 1), ax2.reshape(-1, 1)
    m, mm = np.inf, -np.inf
    if axis == 0:
        all_ax = [np.zeros([N*N, 1]), ax1, ax2]
        xlabel, ylabel, zlabel = 'y', 'z', 'x'

    elif axis == 1:
        all_ax = [ax1, np.zeros([N*N, 1]), ax2]
        xlabel, ylabel, zlabel = 'x', 'z', 'y'
    else:
        all_ax = [ax1, ax2, np.zeros([N*N, 1])]
        xlabel, ylabel, zlabel = 'x', 'y', 'z'
    all_ax = np.concatenate(all_ax, axis=1)

    N_slice = len(slices)
    for i in range(N_slice):

        all_ax[:, axis] = slices[i]
        res = np.apply_along_axis(fun, 1, all_ax)
        m = np.minimum(res, m)
        mm = np.maximum(mm, res)
        plt.imshow(res.reshape(N, N), extent=other_bounds, origin='lower')
        plt.xticks()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(zlabel + " = " + str(slices[i]))
        plt.colorbar()
        plt.show()
    print("Maximum on each slice : {} \n Minimum on each slice : {}".format(m, mm))


def create_grid_spherical(rmin, rmax, dr, dtheta, dphi, verbose=False):
    """Create a grid using spherical coordinates. Adapted form Khaoula Chahdi.
    Args:
        -rmin, rmax (float) : boundaries for the radius
        -dr (float) : step for the radius
        -dtheta, dphi (float) : steps for the angular coordinates, in degrees. dtheta will be rescaled in order to get a
    uniform covering of each sphere
    Return: a tuple (grid, sph_grid, n_per_sphere) where :
        -grid : (n, 3) array containing the cartesian coordinates
        -sph_grid : (n,3) array containing the spherical coordinates
        -n_per_sphere (int) : number of points per sphere
    """

    dtheta_scaled = (dtheta * np.pi) / 180
    n_theta = int(np.ceil(2 * np.pi / dtheta_scaled))
    dphi_scaled = (dphi * np.pi) / 180
    n_phi = int(np.ceil(np.pi / dphi_scaled) + 1)
    # number of spheres
    n_r = int(np.ceil((rmax - rmin) / dr + 1))
    grid = []
    sph_grid = []
    n = 0
    for r in np.linspace(rmin, rmax, n_r):
        for phi in np.linspace(0, np.pi, n_phi):
            theta_range = np.linspace(0, 2 * np.pi, int(np.ceil(n_theta * np.cos(phi - np.pi / 2))), endpoint=False)
            for theta in theta_range:
                x = r * np.cos(theta) * np.sin(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(phi)
                grid += [x, y, z]
                sph_grid += [r, theta, phi]
                n += 1
                if verbose and n % 100 == 0:
                    print("n:" + str(n) + " r:" + str(r) + " theta:" + str(theta) + " phi:" + str(phi))
    grid = np.array(grid).reshape(-1, 3)
    sph_grid = np.array(sph_grid).reshape(-1, 3)
    n_per_sphere = n // n_r
    return grid, sph_grid, n_per_sphere


def plot_3d_sphere(fun, radius, dtheta, dphi):
    """
    Plot the function fun on 3d spheres indicated by the list of radiuses "radius".
    dtheta, dphi give the angular discretizations for each sphere.
    """

    for i in range(len(radius)):
        r = radius[i]
        grid, _, _ = create_grid_spherical(r, r, 1, dtheta, dphi)
        res = np.apply_along_axis(fun, 1, grid)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], marker='o', c=res)
        plt.title('r = {} '.format(r))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.colorbar(sc)
        plt.show()


def disp_measure(a, x):
    """
    Print the measure defined by the array of amplitudes a and the array of positions x.
    """

    n = len(a)
    print("m =")
    for k in range(n - 1):
        print("{}*d_({},{},{}) +".format(a[k], *x[k]))
    print("{}*d_({},{},{})".format(a[-1], *x[-1]))


def compare_arrays(a, b):
    """Compute the distances between each line vector of two 2d-arrays a and b and match the smallest distances.

    Return : tuple (ind, dist) where :
        -ind is a flat array such as ind[i] is the line of b closest to the ith line of a
        -dist is a flat array containing the corresponding distances """

    assert a.shape == b.shape
    # a,b have shape (N, d), dist has shape N, N
    dist = np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=-1))
    ind_min = np.argmin(dist, axis=1)
    return ind_min, dist[np.arange(len(a)), ind_min]
