import numpy as np
from typing import Tuple
import json
import pyroomacoustics as pra

tol = 1e-10
c = 343
# standard pyroomacoustics offset for RIR start
std_off = pra.constants.get("frac_delay_length") // 2


def multichannel_rir_to_vec(rir: list, start_offset: int = std_off, source: int = 0, cutoff: int = -1) -> (np.ndarray, int, int):
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

    if cutoff > 0:
        length = int(np.minimum(N, cutoff))
    else:
        length = N

    res = np.empty(length*M, dtype=float)

    for i in range(M):
        local_size = len(rir[i][source][start_offset:])
        buffer = np.zeros(length)
        buffer[:local_size] = rir[i][source][start_offset:length + start_offset]
        res[i*length:(i+1)*length] = buffer

    return res, length, M


def vec_to_rir(rir_vec, m, N):
    """
    Return the mth RIR of length N from a vector containing a multichannel RIR.
    Args:
        -m (int) : the index of the microphone
        -N (int) : the length of a single RIR (all RIRs are expected to have the same length)
    """
    return rir_vec[m*N:(m + 1)*N]


def cut_vec_rir(rir_vec, M, N, Ncut):
    """
    Cut a flat RIR up to a given time sample Ncut (each of the M RIRs are cut) and return the corresponding vector.
    """
    return rir_vec.reshape(M, N)[:, :Ncut].flatten()


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


def create_grid_spherical(rmin, rmax, dr, dtheta, dphi, verbose=False, cartesian=True) -> (np.ndarray, np.ndarray, int):
    """Create a grid using spherical coordinates. Adapted from Khaoula Chahdi.
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
                if cartesian:
                    x = r * np.cos(theta) * np.sin(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(phi)
                else:
                    x, y, z = r, theta, phi

                grid += [x, y, z]
                sph_grid += [r, theta, phi]
                n += 1
                if verbose and n % 100 == 0:
                    print("n:" + str(n) + " r:" + str(r) + " theta:" + str(theta) + " phi:" + str(phi))
    grid = np.array(grid).reshape(-1, 3)
    sph_grid = np.array(sph_grid).reshape(-1, 3)
    n_per_sphere = n // n_r

    return grid, sph_grid, n_per_sphere


def create_grid_spherical_multiple(rmin, nspheres, dr, dtheta, dphi) -> (np.ndarray, np.ndarray, int):
    """Create a grid using spherical coordinates. Adapted from Khaoula Chahdi.
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
    grid = []
    sph_grid = []

    phi_range, dphi = np.linspace(0, np.pi, n_phi, retstep=True)
    phi_range2 = np.concatenate([[0.], phi_range + dphi/2])
    phi_range2[-1] = np.pi

    phi_ranges = [phi_range, phi_range2]
    r_range = np.arange(-nspheres, nspheres + 1)*dr + rmin
    for i, r in enumerate(r_range):
        curr_range = phi_ranges[i % 2]
        for phi in curr_range:
            theta_range, dtheta = np.linspace(0, 2 * np.pi, int(np.ceil(n_theta * np.cos(phi - np.pi / 2))),
                                              endpoint=False, retstep=True)

            for theta in theta_range:
                theta = theta+dtheta/(1+i%2)
                x = r * np.cos(theta) * np.sin(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(phi)
                grid += [x, y, z]
                sph_grid += [r, theta, phi]

    grid = np.array(grid).reshape(-1, 3)
    sph_grid = np.array(sph_grid).reshape(-1, 3)

    return grid, sph_grid


def disp_measure(a, x):
    """
    Print the measure defined by the array of amplitudes a and the array of positions x.
    """

    n = len(x)
    print("m =")
    for k in range(n - 1):
        print("{}*d_({},{},{}) +".format(a[k], *x[k]))
    print("{}*d_({},{},{})".format(a[-1], *x[-1]))


def compare_arrays(a, b):
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


def unique_matches(a, b, ampl=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the distances between each line vector of two 2d-arrays a and b and match the smallest distances from
    the lines of a to the lines of b. If more than one line of a is matched to the same line of b, the array ampl can be
    used to eliminate the duplicate matches. If ampl is None, the smallest distance is retained.
    
    Args:-ampl (ndarray):flat array of amplitudes used to sort the matches. If a[k] and a[l] are matched to b[m] and
    ampl[k] < ampl[l] only a[l] is considered. If all amplitudes are equal the smallest index wins. If ampl is None,
    return the closest match in distance.
    
    Return: tuple(inda, indb, dist) where :
        -inda, indb are the arrays giving the index of matches between the lines of a and b.
    len(inda) = len(indb) <= len(b) (at most len(b) unique matches can be found).
        -dist is the corresponding array of distances
    """

    ind_min, dist = compare_arrays(a, b)  # for each line of a, compute the closest line of b
    unique = np.unique(ind_min)  # get the indices of the lines of b closest to those of a, without repetitions
    final_inda_list, final_indb_list, final_dist_list = [], [], []
    for ind in unique:  # loop over the indices of the lines of b closest to a
        matches = ind_min == ind  # indices of the lines of a matched to the line ind of b
        if ampl is None:
            tmp_dist = np.full_like(dist, np.inf, dtype='float')
            tmp_dist[matches] = dist[matches]
            best_match = np.argmin(tmp_dist)  # index of the line of a closest to the line indexed by ind in b
        else:
            tmp_ampl = np.zeros_like(ampl)
            tmp_ampl[matches] = ampl[matches]
            best_match = np.argmax(tmp_ampl)
        final_inda_list.append(best_match)
        final_indb_list.append(ind)
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


def save_results(res_path, rir_path, image_pos, ampl, reconstr_pos, reconstr_ampl,
                 rir, reconstr_rir, N, orders=None, **kwargs):
    exp_res = dict(image_pos=array_to_list(image_pos), ampl=array_to_list(ampl), orders=array_to_list(orders),
                   reconstr_pos=array_to_list(reconstr_pos), reconstr_ampl=array_to_list(reconstr_ampl), N=N)
    rir_dict = dict(rir=array_to_list(rir), reconstr_rir=array_to_list(reconstr_rir))
    for arg in kwargs:
        exp_res[arg] = array_to_list(kwargs[arg])

    dict_to_json(exp_res, res_path)
    dict_to_json(rir_dict, rir_path)

