import numpy as np
import pyroomacoustics as pra
import json
from .utils import multichannel_rir_to_vec, array_to_list, c


def load_antenna(file_path='data/eigenmike32_cartesian.csv', mic_size=1.):
    """Load the eigenmike32 spherical microphone array and dilate it by the factor mic_size
    Source: https://www.locata.lms.tf.fau.de/files/2020/01/Documentation_LOCATA_final_release_V1.pdf
    """
    return np.genfromtxt(file_path, delimiter=',') * mic_size


def simulate_rir(room_dim, fs, src_pos, mic_array, max_order, cutoff=-1,
                 origin=None, absorptions=None, save=None, verbose=False, return_full=False, max_src=5000):
    """Simulate a RIR using pyroomacoustics, using the parameters given in a .json configuration file/dictionary of
    parameters.
    The configuration file must contain the following fields :
        -room_dim (list) : dimensions of the room
        -fs (int) : sampling frequency
        -src_pos (list) : coordinates of the source
        -mic_array (array) : position of every microphone ((M, d) shape)
        -max_order (int) : reflection order
        -origin (list, optional) : new origin for the coordinate system (translates the sources and microphones)
        -absorption (dict, optional) : dictionary containing the absorption rates (floats) for each wall (east, west,
    north, south, floor and ceiling)
        -cutoff (float, optional) : maximum length for the RIR (in seconds)
    If save is a string, write the rir, rir length, sources and amplitudes to a file.
    Return :
        -measurements : vector containing every RIR of length N in a sequence
        -N : length of a given RIR
        -src_pos:  array (shape (K, d)) containing the positions of the sources
        -ampl : vector containing the correpsonding amplitudes
        -mic_array : array (shape (M, d)) giving the positions of the microphones (possibly translated if origin is
    given in the configuration file)
    """

    if absorptions is None:
        absorptions = {"east": pra.Material(0.1), "west": pra.Material(0.1),
                       "north": pra.Material(0.1), "south": pra.Material(0.1),
                       "ceiling": pra.Material(0.1), "floor": pra.Material(0.1)}
        if verbose:
            print("Using default absorption rate for each wall")
    else:
        for key, val in absorptions.items():  # convert the absorption rates to PRA materials
            absorptions[key] = pra.Material(val)

    all_flat_materials = absorptions

    room = pra.ShoeBox(room_dim, fs=fs,
                       materials=all_flat_materials, max_order=max_order)

    # add the source
    room.add_source(src_pos)

    room.add_microphone_array(mic_array.T)

    # Simulate RIR with image source method
    room.compute_rir()

    # get the image sources and corresponding amplitudes
    full_src = room.sources[0].get_images(max_order=max_order).T
    full_ampl = room.sources[0].get_damping(max_order=max_order).flatten()
    orders = room.sources[0].orders

    if cutoff > 0:
        # compute the discretized cutoff
        dcutoff = int(cutoff * fs)
        # compute the maximal detection distance to a microphone
        max_dist = cutoff * c

        # compute the distances between the sources (n_sources, 3) and microphones (M, 3), shape (M, n_src)
        dist = np.sqrt(np.sum((full_src[np.newaxis, :, :] - mic_array[:, np.newaxis, :]) ** 2, axis=2))
        # find the sources that are at a distance at most max_dist of every microphone (shape n_src)
        remaining_src_ind = np.all(dist < max_dist, axis=0)
        src, ampl, orders = full_src[remaining_src_ind, :], full_ampl[remaining_src_ind], orders[remaining_src_ind]

        nb_src = np.sum(remaining_src_ind)
        if max_src <= nb_src:
            print("max_src threshold too low")
            max_src = nb_src

        if return_full:
            dist_sorted_ind = np.argsort(dist[0])  # indices of the sorted distances to the first microphone
            full_src, full_ampl = full_src[dist_sorted_ind[:max_src], :], full_ampl[dist_sorted_ind[:max_src]]

    else:
        dcutoff = -1
        src, ampl = full_src.copy(), full_ampl.copy()

    # assemble the multichannel rir in a single array, N, M the number of time samples and microphones
    measurements, N, M = multichannel_rir_to_vec(room.rir, cutoff=dcutoff)
    measurements = measurements / 4 / np.pi  # rescaling factor
    if origin is not None:
        origin = np.array(origin).reshape(1, 3)
        src -= origin
        mic_array -= origin
        if return_full:
            full_src -= origin

    if type(save) == str:
        res = dict(mic_array=array_to_list(mic_array), image_pos=array_to_list(src),
                   ampl=array_to_list(ampl), N=N, rir=array_to_list(measurements))

        if origin is not None:
            res["origin"] = array_to_list(origin)
        fd = open(save, 'w')
        json.dump(res, fd)
        fd.close()

    if return_full:
        return measurements, N, src, ampl, mic_array, orders, full_src, full_ampl
    else:
        return measurements, N, src, ampl, mic_array, orders

