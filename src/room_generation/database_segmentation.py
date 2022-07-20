import numpy as np
import os
from src.simulation.utils import json_to_dict
from src.simulation.simulate_pra import simulate_rir, load_antenna
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp


def aggregate_db(path):
    file_paths = os.listdir(path)
    entries = []
    for f in file_paths:
        f_split = f.split("_")
        if f_split[-1] == "param.json":
            params = json_to_dict(os.path.join(path, f))
            entries.append([f_split[1], params["room_dim"][0], params["room_dim"][1], params["room_dim"][2],
                            params["origin"][0], params["origin"][1], params["origin"][2],
                            params["src_pos"][0], params["src_pos"][1], params["src_pos"][2]])

    df = pd.DataFrame(entries, columns=["exp_id", "room_dim_x", "room_dim_y", "room_dim_z",
                                        "origin_x", "origin_y", "origin_z", "src_pos_x", "src_pos_y", "src_pos_z"])
    df["volume"] = df.apply(lambda line: line.room_dim_x*line.room_dim_y*line.room_dim_z, axis=1)
    df["surface"] = df.apply(lambda line: 2*(line.room_dim_x*line.room_dim_y + line.room_dim_y*line.room_dim_z
                                             + line.room_dim_x*line.room_dim_z), axis=1)

    return df


class ParallelSrc:
    def __init__(self, df, mic_array, simulation_time):
        self.mic_array = mic_array
        self.simulation_time = simulation_time
        self.df = df

    def _append_nb_src(self, line):
        _, _, _, ampl, _, _ = simulate_rir(line[:3], fs=16000, src_pos=line[3:6], mic_array=self.mic_array,
                                           max_order=20, cutoff=self.simulation_time, origin=line[6:])
        return len(ampl)

    def compute_nb_src(self):
        p = mp.Pool(8)
        maparr = np.stack([self.df["room_dim_x"], self.df["room_dim_y"], self.df["room_dim_z"], self.df["src_pos_x"],
                           self.df["src_pos_y"], self.df["src_pos_z"], self.df["origin_x"], self.df["origin_y"],
                           self.df["origin_z"]], axis=1)

        gr_opt = p.map(self._append_nb_src, maparr)
        p.close()
        self.df["nb_src"] = gr_opt


if __name__ == "__main__":
    antenna = load_antenna(mic_size=1.)

    dff = aggregate_db("room_db_full")
    ps = ParallelSrc(dff, antenna, 0.05)
    ps.compute_nb_src()
    plt.hist(dff.volume, edgecolor='k')
    plt.show()
    plt.hist(dff.surface, edgecolor='k')
    plt.show()

    plt.hist(dff.nb_src, edgecolor='k')
    plt.show()
    print(dff["nb_src"].min(), dff["nb_src"].max())
    dff.to_csv("db_info.csv")
    plt.show()

