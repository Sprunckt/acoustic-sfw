"""
Creating the parameter files for multiple random experiments and launching simulations
"""

import numpy as np
from scipy.spatial.transform import Rotation
import json
import pandas as pd
import os
import sys
from src.simulation.utils import dict_to_json
from src.simulation.simulate_pra import load_antenna, simulate_rir

tol_recov = 2e-2

directory = "room_db_full"
if not os.path.exists(directory):
    os.mkdir(directory)

op_path = os.path.join(directory, "operator.json")
if not os.path.exists(op_path):
    fd = open(op_path, "w")
    json.dump(dict(exp_id=0), fd)
    fd.close()


def update_op(idd):
    fd1 = open(op_path, "r")
    curr_dict = json.load(fd1)
    fd1.close()

    fd2 = open(op_path, "w")
    curr_dict["exp_id"] = idd
    json.dump(curr_dict, fd2)
    fd2.close()


def get_id():
    fd1 = open(op_path, "r")
    curr_dict = json.load(fd1)
    idd = curr_dict["exp_id"]
    fd1.close()
    return idd


def choose_random(lb, ub, fixed=None):
    if fixed is None:
        return np.random.uniform(lb, ub)
    else:
        return fixed


conf_path = os.path.join(directory, "parameters.json")
if not os.path.exists(conf_path):
    print("Error : the conf file must be created first.")
    param_dict = None
    exit(1)
else:  # loading the global parameters for the experiments
    fd = open(conf_path, 'r')
    param_dict = json.load(fd)

df_path = os.path.join(directory, "results.csv")
if not os.path.exists(df_path):
    df_res = pd.DataFrame(columns=["exp_id", "min_dist", "mean_dist", "max_dist", "nb_recov"])
else:
    df_res = pd.read_csv(df_path)

new_exp = 1000
n_generated = 0
while new_exp > n_generated:
    n_generated += 1
    # defining the room dimensions
    room_dim = [np.random.uniform(*param_dict["xlim"]),
                np.random.uniform(*param_dict["ylim"]),
                np.random.uniform(*param_dict["zlim"])]
    max_order = 1

    xroom, yroom, zroom = room_dim
    # check if the source or microphone altitudes are preset
    z_src = param_dict.get("z_src")
    z_mic = param_dict.get("z_mic")

    # lower and upper bounds for the random coordinates
    lb = param_dict["mic_wall_sep"]
    ub = (xroom - param_dict["mic_wall_sep"], yroom - param_dict["mic_wall_sep"], zroom - param_dict["mic_wall_sep"])
    fixed_src, fixed_mic = [None, None, z_src], [None, None, z_mic]
    args_src = [(lb, ub[i], fixed_src[i]) for i in range(3)]
    args_mic = [(lb, ub[i], fixed_mic[i]) for i in range(3)]

    src_pos = np.array([choose_random(*args_src[i]) for i in range(3)])
    mic_pos = np.array([choose_random(*args_mic[i]) for i in range(3)])

    k, max_reject = 0, 100000
    while np.linalg.norm(src_pos - mic_pos) < param_dict["mic_src_sep"] and k < max_reject:
        src_pos = np.array([choose_random(*args_src[i]) for i in range(3)])
        mic_pos = np.array([choose_random(*args_mic[i]) for i in range(3)])

        k += 1

    if k == max_reject:
        print("Error : failed to place the antenna")
        n_generated -= 1
    else:
        sim_param = dict(src_pos=src_pos, room_dim=room_dim, origin=mic_pos)

        exp_id = get_id()
        str_id = "exp_" + str(exp_id)

        # generate 3 random angles for random rotations around Ox,Oy,Oz
        sim_param["rotation_mic"] = Rotation.random().as_euler("xyz", degrees=True)
        sim_param["rotation_walls"] = Rotation.random().as_euler("xyz", degrees=True)

        # generate random absorption coefficients
        abs_coeff = np.random.uniform(param_dict["min_abs"], param_dict["max_abs"], size=6)
        absorptions = dict(north=abs_coeff[0], south=abs_coeff[1], east=abs_coeff[2], west=abs_coeff[3],
                           floor=abs_coeff[4], ceiling=abs_coeff[5])
        sim_param["absorptions"] = absorptions

        dict_to_json(sim_param, os.path.join(directory, str_id + "_param.json"))

        print("Exp {} done".format(exp_id))
        update_op(exp_id + 1)
