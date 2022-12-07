""" Script to generate and save aruco sticker detection """
import numpy as np
import pickle
from helpers.io import *
from helpers.img_process import *

# Args
dataset_path = f'/home/vishal/vishal/camera_calibration/datasets/unanno36_sorted256_theta_5_70_8_phi_0_360_16'
num_parts = 64
n_stickers = 36
aruco_id_start = 512
verbose = True
aruco_dict = aruco.DICT_5X5_1000
save_path = f'/home/vishal/vishal/camera_calibration/calibration/data/unanno36_sorted256_theta_5_70_8_phi_0_360_16_aruco.pkl'

# Read dataset
dataset = read_dataset(dataset_path, num_parts, v=verbose)

# Detect ArUco stickers
dataset_aruco, arr_aruco_dets = detect_aruco(dataset, n_stickers, aruco_id_start, 
                                             aruco_dict, v=verbose)

# Save
with open(save_path, 'wb') as f:
    pickle.dump(dataset_aruco, f)
    if verbose:
        print(f"Saved to {save_path}")
