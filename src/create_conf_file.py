import os
import json

directory = "sfw_exp3"
param_dict = dict()
# bounds for the room dimensions
param_dict["xlim"] = [2, 10]
param_dict["ylim"] = [2, 10]
param_dict["zlim"] = [2, 10]

param_dict["mic_size"] = 5.
param_dict["mic_src_sep"] = 1.
param_dict["src_wall_sep"] = 1.
param_dict["mic_wall_sep"] = 1.
param_dict["dr"] = 0.3
param_dict["rmin"] = 1.5
param_dict["dphi"] = 25
param_dict["fs"] = 16000
# additional constraint to set the antenna and source altitudes (set to None to keep random)
param_dict["z_src"] = 1.
param_dict["z_mic"] = 1.

param_path = os.path.join(directory, "parameters.json")
if not os.path.exists(param_path):
    fd = open(param_path, 'w')
    json.dump(param_dict, fp=fd)
    fd.close()
else:  # check if the parameters match the conf file
    fd = open(param_path, 'r')
    param_dict2 = json.load(fd)
    for key in param_dict:
        if param_dict[key] != param_dict2[key]:
            print("Error : the parameters have been modified ")
            print("parameter {} was {} and is now {}".format(key, param_dict[key], param_dict2[key]))
            exit(1)
    print("The configuration file for experiment {} exists and matches the given parameters".format(directory))
    fd.close()
