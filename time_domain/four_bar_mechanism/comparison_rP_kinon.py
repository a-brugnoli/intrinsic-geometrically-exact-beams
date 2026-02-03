import pandas as pd
import matplotlib.pyplot as plt
import os
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

path_file = os.path.dirname(os.path.abspath(__file__))

folder_results = path_file + "/results/"
case = "gyrator_soft"

df_mine = pd.read_csv( folder_results + "/" + case + "/" + "time_signal_rP.csv")

time_vec_mine = df_mine["time"].to_numpy()
r_x_mine = df_mine["x"].to_numpy()
r_y_mine = df_mine["y"].to_numpy()

df_kinon = pd.read_csv(folder_results + "time_signal_rP_kinon.csv")
time_vec_kinon = df_kinon["time"].to_numpy()
r_x_kinon = df_kinon["x"].to_numpy()
r_y_kinon = df_kinon["y"].to_numpy()

plt.figure()
plt.plot(time_vec_mine, r_x_mine, label='Present')
plt.plot(time_vec_kinon, r_x_kinon, label='Kinon (2025)', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('$r_x^P$')
plt.grid(True)
plt.savefig(folder_results + f"comparison_r_x_P_{case}.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec_mine, r_y_mine, label='Present')
plt.plot(time_vec_kinon, r_y_kinon, label='Kinon (2025)', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('$r_y^P$')
plt.grid(True)
plt.savefig(folder_results + f"comparison_r_y_P_{case}.pdf", format="pdf")
plt.tight_layout()

