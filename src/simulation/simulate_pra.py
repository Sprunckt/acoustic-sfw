import numpy as np
import pyroomacoustics as pra
import json
from .utils import multichannel_rir_to_vec, array_to_list, json_to_dict


def load_antenna(file_path='data/eigenmike32_cartesian.csv', mic_size=1.):
    """Load the eigenmike32 spherical microphone array and dilate it by the factor mic_size
    Source: https://www.locata.lms.tf.fau.de/files/2020/01/Documentation_LOCATA_final_release_V1.pdf
    """
    return np.genfromtxt(file_path, delimiter=', ') * mic_size


def simulate_rir(conf_path, save=None):
    """Simulate a RIR using pyroomacoustics, using the parameters given in a .json configuration file/dictionary of
    parameters.
    The configuration file must contain the following fields :
        -room_dim (list) : dimensions of the room
        -fs (int) : sampling frequency
        -src_pos (list) : coordinates of the source
        -mic_array (array) : position of every microphone ((M, d) shape)
        -max_order (int) : reflection order
        -origin (list, optional) : new origin for the coordinate system (translates the sources and microphones)

    If save is a string, write the rir, rir length, sources and amplitudes to a file.
    Return :
        -measurements : vector containing every RIR of length N in a sequence
        -N : length of a given RIR
        -src_pos:  array (shape (K, d)) containing the positions of the sources
        -ampl : vector containing the correpsonding amplitudes
        -mic_array : array (shape (M, d)) giving the positions of the microphones (possibly translated if origin is
    given in the configuration file)
    """

    if type(conf_path) == str:
        param_dict = json_to_dict(conf_path)
    else:
        param_dict = conf_path

    room_dim, fs = param_dict["room_dim"], param_dict["fs"]
    max_order = param_dict.get("max_order")

    all_flat_materials = {"east": pra.Material(0.1), "west": pra.Material(0.1),
                          "north": pra.Material(0.1), "south": pra.Material(0.1),
                          "ceiling": pra.Material(0.1), "floor": pra.Material(0.1)}

    room = pra.ShoeBox(room_dim, fs=fs,
                       materials=all_flat_materials, max_order=max_order)

    # add the source
    room.add_source(param_dict["src_pos"])

    mic_array = param_dict["mic_array"]

    room.add_microphone_array(mic_array.T)

    # Simulate RIR with image source method
    room.compute_rir()

    # assemble the multichannel rir in a single array
    measurements, N, M = multichannel_rir_to_vec(room.rir)  # N, M : number of time samples and microphones

    # get the image sources and corresponding amplitudes
    src = room.sources[0].get_images(max_order=max_order).T
    ampl = room.sources[0].get_damping(max_order=max_order).flatten()

    origin = param_dict.get("origin")
    if origin is not None:
        origin = np.array(origin).reshape(1, 3)
        src -= origin
        mic_array -= origin

    if type(save) == str:
        res = dict(mic_array=array_to_list(mic_array), image_pos=array_to_list(src),
                   ampl=array_to_list(ampl), N=N, rir=array_to_list(measurements))

        if origin is not None:
            res["origin"] = array_to_list(origin)
        fd = open(save, 'w')
        json.dump(res, fd)
        fd.close()

    return measurements, N, src, ampl, mic_array
