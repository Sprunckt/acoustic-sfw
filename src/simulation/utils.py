import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import json
from itertools import product, combinations
import pyroomacoustics as pra

tol = 1e-10
c = 343
# standard pyroomacoustics offset for RIR start
std_off = pra.constants.get("frac_delay_length") // 2


def multichannel_rir_to_vec(rir: list, start_offset: int = std_off, source: int = 0) -> (np.ndarray, int, int):
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


def create_grid_spherical(rmin, rmax, dr, dtheta, dphi, verbose=False) -> (np.ndarray, np.ndarray, int):
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


def plot_room(mic, src, ampl, reconstr_src):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    src_ind = np.argmax(ampl)
    ax.scatter(*src[src_ind], label='source', marker='x')
    sec_src_ind = ampl < np.max(ampl)

    wall_intersect = (src[src_ind] + src[sec_src_ind]) / 2
    xmin, ymin, zmin = np.min(wall_intersect, axis=0)
    xmax, ymax, zmax = np.max(wall_intersect, axis=0)

    vertices = product([xmin, xmax], [ymin, ymax], [zmin, zmax])
    edges_plus = combinations(vertices, 2)
    for edge in edges_plus:
        x1, x2, y1, y2, z1, z2 = edge[0][0], edge[1][0], edge[0][1], edge[1][1], edge[0][2], edge[1][2]
        if len(np.unique([x1, x2])) + len(np.unique([y1, y2])) + len(np.unique([z1, z2])) == 4:
            ax.plot3D([x1, x2], [y1, y2], [z1, z2], color='k')

    ax.scatter(mic[:, 0], mic[:, 1], mic[:, 2], label='microphones', marker='+')
    ax.scatter(src[sec_src_ind, 0], src[sec_src_ind, 1], src[sec_src_ind, 2], label='image sources',
               marker='X', alpha=0.7, s=3, color='blue')
    ax.scatter(reconstr_src[:, 0], reconstr_src[:, 1], reconstr_src[:, 2], label='reconstructed sources',
               marker='o',  edgecolor='k')
    plt.legend()
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


def compare_arrays(a, b, unique=False):
    """Compute the distances between each line vector of two 2d-arrays a and b and match the smallest distances from
    the lines of a to the lines of b (compare_arrats(a, b) != compare_arrays(b, a)).

    Return : tuple (ind, dist) where :
        -ind is a flat array such as ind[i] is the line of b closest to the ith line of a
        -dist is a flat array containing the corresponding distances """

    # a,b have shape (N1, d), (N2,d), dist has shape N1, N2
    dist = np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=-1))
    # shape N1
    ind_min = np.argmin(dist, axis=1)
    return ind_min, dist[np.arange(len(a)), ind_min]


def unique_matches(a, b, ampl) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the distances between each line vector of two 2d-arrays a and b and match the smallest distances from
    the lines of a to the lines of b. If more than one line of a is matched to the same line of b, the array ampl is 
    used to eliminate the duplicate matches.
    
    Args:-ampl (ndarray):flat array of amplitudes used to sort the matches. If a[k] and a[l] are matched to b[m] and
    ampl[k] < ampl[l] only a[l] is considered. If all amplitudes are equal the smallest index wins.
    
    Return: tuple(inda, indb, dist) where :
        -inda, indb are the arrays giving the index of matches between the lines of a and b
        -dist is the corresponding array of distances
    """
    assert len(a) == len(ampl), "invalid shapes"

    ind_min, dist = compare_arrays(a, b)
    unique = np.unique(ind_min)
    final_inda_list, final_indb_list, final_dist_list = [], [], []
    for ind in unique:
        tmp_ampl = np.zeros_like(ampl)
        matches = ind_min == ind
        tmp_ampl[matches] = ampl[matches]
        best_match = np.argmax(tmp_ampl)
        final_inda_list.append(best_match)
        final_indb_list.append(ind_min[best_match])
        final_dist_list.append(dist[best_match])

    return np.array(final_inda_list), np.array(final_indb_list), np.array(final_dist_list)


def correlation(a1, a2):
    assert a1.size == a2.size, "vector lengths do not match"
    prod = np.linalg.norm(a1) * np.linalg.norm(a2)
    if prod < tol:
        return 0
    else:
        return np.dot(a1, a2) / prod


def array_to_list(arr):
    """Converts arr to a list if arr is an ndarray, leave as it is otherwise."""
    if type(arr) == np.ndarray:
        return arr.tolist()
    else:
        return arr


def dict_to_json(dico, path):
    dd = dico.copy()
    for key in dd:
        dd[key] = array_to_list(dd[key])
    fd = open(path, 'w')
    json.dump(dd, fd)
    fd.close()


def json_to_dict(path, list_to_array=True):
    fd = open(path, 'r')
    dump = json.load(fd)
    fd.close()
    if list_to_array:
        for key in dump:
            if type(dump[key]) == list:
                dump[key] = np.array(dump[key])
    return dump


def save_results(file_name, image_pos, ampl, reconstr_pos, reconstr_ampl, rir, reconstr_rir, N, rmax, **kwargs):
    exp_res = dict(image_pos=array_to_list(image_pos), ampl=array_to_list(ampl),
                   reconstr_pos=array_to_list(reconstr_pos), reconstr_ampl=array_to_list(reconstr_ampl),
                   rir=array_to_list(rir), reconstr_rir=array_to_list(reconstr_rir), N=N, rmax=rmax)
    for arg in kwargs:
        exp_res[arg] = kwargs[arg]
    dict_to_json(exp_res, file_name)
