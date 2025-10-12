import os
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_trajectory(gt_csv):
    import pandas as pd
    gt = pd.read_csv(gt_csv, comment='#')
    plt.plot(gt['p_RS_R_x [m]'], gt['p_RS_R_y [m]'])
    plt.title('Ground Truth Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.show()
