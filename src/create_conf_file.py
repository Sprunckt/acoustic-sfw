import os
from src.simulation.utils import dict_to_json, json_to_dict

directory = "room_db_full"
room_param = True  # if True create a conf file for random simulation, else a conf file for SFW experiments

param_dict = dict()
if room_param:  # parameters for room generation
    # bounds for the room dimensions generation
    param_dict["xlim"] = [2, 10]
    param_dict["ylim"] = [2, 10]
    param_dict["zlim"] = [2, 5]

    param_dict["mic_size"] = 5.
    param_dict["mic_src_sep"] = 1.
    param_dict["src_wall_sep"] = 1.
    param_dict["mic_wall_sep"] = 1.

    # additional constraint to set the antenna and source altitudes (set to None to keep random)
    param_dict["z_src"] = None
    param_dict["z_mic"] = None

    # minimal and maximal absorptions
    param_dict["min_abs"] = 0.01
    param_dict["max_abs"] = 0.3

else:  # parameters to configure SFW and the simulations
    param_dict["max_order"] = 1
    param_dict["fs"] = 16000
    param_dict["dr"] = 0.3
    param_dict["rmin"] = 1.
    param_dict["rmax"] = 1.  # set rmin = rmax to search on a sphere, set rmax to None to set rmax automatically
    param_dict["dphi"] = 25
    param_dict["normalization"] = 1
    param_dict["min_norm"] = 0
    param_dict["lambda"] = 1e-6
    param_dict["ideal"] = True  # ideal operator if True, pra simulation otherwise
    param_dict["spherical_search"] = True  # set to True to search on a single sphere (in that case rmax=rmin=1)
    param_dict["mic_size"] = 5.  # overwrite the microphone positions by placing an antenna with the given radius factor
    param_dict["max_iter"] = 8
    param_dict["use_two_antennas"] = False
    param_dict["domain"] = "time"
    param_dict["use_absorptions"] = True


param_path = os.path.join(directory, "parameters.json")
if not os.path.exists(param_path):
    dict_to_json(param_dict, param_path)

else:  # check if the parameters match the conf file
    param_dict2 = json_to_dict(param_path, list_to_array=False)
    mod = False
    for key in param_dict:
        if param_dict[key] != param_dict2[key]:
            mod = True
            print("parameter {} was {} and is now {}".format(key, param_dict2[key], param_dict[key]))

    if not mod:
        print("The configuration file for experiment {} exists and matches the given parameters".format(directory))
    else:
        dict_to_json(param_dict, param_path)

