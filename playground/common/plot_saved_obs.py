import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
from pathlib import Path

# Default styling for dense subplot grids:
# - keep titles readable without overlapping
# - avoid legends covering signals
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", type=str, required=False, default="mujoco_saved_obs.pkl"
)
parser.add_argument(
    "--save-dir",
    type=str,
    required=False,
    default=None,
    help="If set, save plots as PNGs into this directory.",
)
parser.add_argument(
    "--no-show",
    action="store_true",
    help="If set, do not open interactive windows (useful for headless runs).",
)
args = parser.parse_args()

save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
if save_dir:
    save_dir.mkdir(parents=True, exist_ok=True)
data_stem = Path(args.data).name
if data_stem.endswith(".pkl"):
    data_stem = data_stem[: -len(".pkl")]


init_pos = np.array(
    [
        0.002,
        0.053,
        -0.63,
        1.368,
        -0.784,
        0.0,
        0,
        0,
        0,
        # 0,
        # 0,
        -0.003,
        -0.065,
        0.635,
        1.379,
        -0.796,
    ]
)

joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

obses = pickle.load(open(args.data, "rb"))

num_dofs = 14
dof_poses = []  # (dof, num_obs)
actions = []  # (dof, num_obs)

for i in range(num_dofs):
    print(i)
    dof_poses.append([])
    actions.append([])
    for obs in obses:
        dof_poses[i].append(obs[13 : 13 + num_dofs][i])
        actions[i].append(obs[26 : 26 + num_dofs][i])

# plot action vs dof pos

nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig_w = max(10.0, nb_cols * 3.2)
fig_h = max(7.0, nb_rows * 2.4)
fig, axs = plt.subplots(
    nb_rows, nb_cols, sharex=True, sharey=True, figsize=(fig_w, fig_h), constrained_layout=True
)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(actions[i * nb_cols + j], label="action", linewidth=1.0)
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="dof_pos", linewidth=1.0)
        # Avoid per-subplot legends (they hide the data); use a figure-level legend instead.
        axs[i, j].set_title(f"{joints_order[i * nb_cols + j]}", fontsize=7)

# Single legend for the whole figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", framealpha=0.8)

fig.suptitle(f"{args.data}")
if save_dir:
    out_path = save_dir / f"{data_stem}_actions_vs_dof_pos.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
if not args.no_show:
    plt.show()

obses_names = [
    "gyro x",
    "gyro y",
    "gyro z",
    "accelo x",
    "accelo y",
    "accelo z",
    # commands
    "command 0",
    "command 1",
    "command 2",
    "command 3",
    "command 4",
    "command 5",
    "command 6",
    # dof pos
    "pos_" + str(joints_order[0]),
    "pos_" + str(joints_order[1]),
    "pos_" + str(joints_order[2]),
    "pos_" + str(joints_order[3]),
    "pos_" + str(joints_order[4]),
    "pos_" + str(joints_order[5]),
    "pos_" + str(joints_order[6]),
    "pos_" + str(joints_order[7]),
    "pos_" + str(joints_order[8]),
    "pos_" + str(joints_order[9]),
    "pos_" + str(joints_order[10]),
    "pos_" + str(joints_order[11]),
    "pos_" + str(joints_order[12]),
    "pos_" + str(joints_order[13]),
    # dof vel
    "vel_" + str(joints_order[0]),
    "vel_" + str(joints_order[1]),
    "vel_" + str(joints_order[2]),
    "vel_" + str(joints_order[3]),
    "vel_" + str(joints_order[4]),
    "vel_" + str(joints_order[5]),
    "vel_" + str(joints_order[6]),
    "vel_" + str(joints_order[7]),
    "vel_" + str(joints_order[8]),
    "vel_" + str(joints_order[9]),
    "vel_" + str(joints_order[10]),
    "vel_" + str(joints_order[11]),
    "vel_" + str(joints_order[12]),
    "vel_" + str(joints_order[13]),
    # action
    "last_action_" + str(joints_order[0]),
    "last_action_" + str(joints_order[1]),
    "last_action_" + str(joints_order[2]),
    "last_action_" + str(joints_order[3]),
    "last_action_" + str(joints_order[4]),
    "last_action_" + str(joints_order[5]),
    "last_action_" + str(joints_order[6]),
    "last_action_" + str(joints_order[7]),
    "last_action_" + str(joints_order[8]),
    "last_action_" + str(joints_order[9]),
    "last_action_" + str(joints_order[10]),
    "last_action_" + str(joints_order[11]),
    "last_action_" + str(joints_order[12]),
    "last_action_" + str(joints_order[13]),
    "last_last_action_" + str(joints_order[0]),
    "last_last_action_" + str(joints_order[1]),
    "last_last_action_" + str(joints_order[2]),
    "last_last_action_" + str(joints_order[3]),
    "last_last_action_" + str(joints_order[4]),
    "last_last_action_" + str(joints_order[5]),
    "last_last_action_" + str(joints_order[6]),
    "last_last_action_" + str(joints_order[7]),
    "last_last_action_" + str(joints_order[8]),
    "last_last_action_" + str(joints_order[9]),
    "last_last_action_" + str(joints_order[10]),
    "last_last_action_" + str(joints_order[11]),
    "last_last_action_" + str(joints_order[12]),
    "last_last_action_" + str(joints_order[13]),
    "last_last_last_action_" + str(joints_order[0]),
    "last_last_last_action_" + str(joints_order[1]),
    "last_last_last_action_" + str(joints_order[2]),
    "last_last_last_action_" + str(joints_order[3]),
    "last_last_last_action_" + str(joints_order[4]),
    "last_last_last_action_" + str(joints_order[5]),
    "last_last_last_action_" + str(joints_order[6]),
    "last_last_last_action_" + str(joints_order[7]),
    "last_last_last_action_" + str(joints_order[8]),
    "last_last_last_action_" + str(joints_order[9]),
    "last_last_last_action_" + str(joints_order[10]),
    "last_last_last_action_" + str(joints_order[11]),
    "last_last_last_action_" + str(joints_order[12]),
    "last_last_last_action_" + str(joints_order[13]),
    "motor_targets_" + str(joints_order[0]),
    "motor_targets_" + str(joints_order[1]),
    "motor_targets_" + str(joints_order[2]),
    "motor_targets_" + str(joints_order[3]),
    "motor_targets_" + str(joints_order[4]),
    "motor_targets_" + str(joints_order[5]),
    "motor_targets_" + str(joints_order[6]),
    "motor_targets_" + str(joints_order[7]),
    "motor_targets_" + str(joints_order[8]),
    "motor_targets_" + str(joints_order[9]),
    "motor_targets_" + str(joints_order[10]),
    "motor_targets_" + str(joints_order[11]),
    "motor_targets_" + str(joints_order[12]),
    "motor_targets_" + str(joints_order[13]),
    "contact left",
    "contact right",
    "imitation_phase 1",
    "imitation_phase 2"
    # ref (ignored)
]
# print(len(obses_names))
# exit()


# obses = [[56 obs at time 0], [56 obs at time 1], ...]

nb_obs = len(obses[0])
print(nb_obs)
nb_rows = int(np.sqrt(nb_obs))
nb_cols = int(np.ceil(nb_obs / nb_rows))

fig_w = max(12.0, nb_cols * 1.6)
fig_h = max(10.0, nb_rows * 1.2)
fig, axs = plt.subplots(
    nb_rows, nb_cols, sharex=True, sharey=True, figsize=(fig_w, fig_h), constrained_layout=True
)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_obs:
            break
        axs[i, j].plot([obs[i * nb_cols + j] for obs in obses], linewidth=0.8)
        axs[i, j].set_title(obses_names[i * nb_cols + j], fontsize=6)

# set ylim between -5 and 5

for ax in axs.flat:
    ax.set_ylim([-5, 5])


fig.suptitle(f"{args.data}")
if save_dir:
    out_path = save_dir / f"{data_stem}_obs_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
if not args.no_show:
    plt.show()
