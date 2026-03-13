import os
import math
import pickle
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from pykalman import KalmanFilter
from natsort import natsorted


def euclidean_distance(p1, p2):
    """Euclidean distance for 2D or 3D points."""
    if len(p1) == 3 and len(p2) == 3:
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )

    if len(p1) == 2 and len(p2) == 2:
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    raise ValueError("Point dimension mismatch")


def run_kalman(col_vals):
    """Apply Kalman filter to 3D trajectory."""

    first_valid = np.where(~np.isnan(col_vals))[0][0]
    initial_state_mean = [
        col_vals[first_valid, 0], 0,
        col_vals[first_valid, 1], 0,
        col_vals[first_valid, 2], 0
    ]

    transition_matrix = [
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
    ]

    observation_matrix = [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ]

    masked_data = np.ma.array(col_vals, mask=np.isnan(col_vals))

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
    )

    init_n = 2
    kf = kf.em(masked_data[:init_n], n_iter=5)

    state_means, state_cov = kf.filter(masked_data[:init_n])

    out = np.zeros((3, masked_data.shape[0]))
    out[:, :init_n] = np.vstack([
        state_means[:, 0],
        state_means[:, 2],
        state_means[:, 4]
    ])

    state_mean = state_means[-1]
    state_cov = state_cov[-1]

    for i in range(init_n, masked_data.shape[0]):
        state_mean, state_cov = kf.filter_update(
            state_mean,
            state_cov,
            masked_data[i]
        )

        out[0, i] = state_mean[0]
        out[1, i] = state_mean[2]
        out[2, i] = state_mean[4]

    return [out[0], out[1], out[2]]


def run_interpolation_3d(predictions3d):
    """Convert dictionary format and apply Kalman filtering."""

    reformatted = {}

    for frame, keypoints in predictions3d.items():
        frame_dict = {}
        for name, value in keypoints.items():
            if isinstance(value, float):
                frame_dict[f"{name}_x"] = np.nan
                frame_dict[f"{name}_y"] = np.nan
                frame_dict[f"{name}_z"] = np.nan
            else:
                frame_dict[f"{name}_x"] = value[0]
                frame_dict[f"{name}_y"] = value[1]
                frame_dict[f"{name}_z"] = value[2]

        reformatted[frame] = frame_dict

    data = pd.DataFrame.from_dict(reformatted, orient="index")

    kalman_data = data.copy()
    names = natsorted({c[:-2] for c in data.columns})

    for name in  tqdm(
        names,
        desc="Keypoints filtering",
        position=1,
        leave=False
    ):
        cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
        vals = data[cols].to_numpy()

        filtered = run_kalman(vals)

        kalman_data[cols[0]] = filtered[0]
        kalman_data[cols[1]] = filtered[1]
        kalman_data[cols[2]] = filtered[2]

    new_df = pd.DataFrame(columns=names)

    for name in names:
        cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
        vals = kalman_data[cols].values.tolist()

        vals = [v if not any(np.isnan(v)) else np.nan for v in vals]
        new_df[name] = vals

    new_df = new_df.applymap(np.array)
    new_df.index = data.index

    final_dict = new_df.to_dict(orient="index")

    before = pd.DataFrame.from_dict(predictions3d, orient="index")
    nan_before = np.sum(before.isna().sum().to_numpy())
    nan_after = np.sum(new_df.isna().sum().to_numpy())

    percent_removed = (nan_after - nan_before) * 100 / new_df.size
    #print(f"Total Removed: {percent_removed}%")

    return final_dict, percent_removed


def apply_kalman(eval_dir, sequences, model_name):
    percentage = {model_name: {}}
    print("Applying Kalman filter...")

    for seq in tqdm(sequences, desc="Processing sequences", position=0):
        #print(f"Sequence: {seq}, Model: {model_name}")

        file_path = os.path.join(
            eval_dir,
            f"SeqEval_Points3D_{model_name}_Seq{seq}.p"
        )

        predictions = pickle.load(open(file_path, "rb"))

        kalman_out, removed = run_interpolation_3d(predictions)
        percentage[model_name][seq] = removed

        out_path = os.path.join(
            eval_dir,
            f"SeqEval_Kalman3DRerun_{model_name}_Seq{seq}.p"
        )

        pickle.dump(kalman_out, open(out_path, "wb"))

    df = pd.DataFrame.from_dict(percentage)

    #print("Mean % Removed:")
    #print(df.apply(np.mean, axis=0))


