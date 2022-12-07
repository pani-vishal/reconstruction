""" Helper script to write JSON config files for the optimization.py script. """
import json
from cv2 import aruco

# Inputs (always change these)
dataset_folder = '/home/vishal/vishal/camera_calibration/datasets'
dataset_name = 'unanno36_sorted256_theta_5_70_8_phi_0_360_16'
# !!! The last front slash is important !!!
save_folder = '/home/vishal/vishal/camera_calibration/calibration/outputs/plots/'
num_parts = 64
n_stickers = 36
## mode options: 'none', 'shift', 'homography'
mode = 'none'
## Set to None if you don't want to plot polar error plot, else: [#thetas, #phis]
plot_polar_info = [8, 16]

# json save_location (always change this)
config_name = 'analysis36.json'
config_path = f'/home/vishal/vishal/camera_calibration/calibration/data/configs_analysis/{config_name}'

# Inputs (change these if needed)
## None if starting from default params, path/to/params.npy if starting from existing params
path_params = f'/home/vishal/vishal/camera_calibration/calibration/outputs/params/{dataset_name}.npy'
## None if aruco stickers haven't been detected yet
path_aruco = f'/home/vishal/vishal/camera_calibration/calibration/data/{dataset_name}_aruco.pkl'
board_rows = 29
board_cols = 18
yfov = 12.5
dist = 901.4990809603678
aruco_dict = aruco.DICT_5X5_1000
aruco_start_id = 512

# Json dicitonary
dict_json = {
    'dataset': f'{dataset_folder}/{dataset_name}',
    'save_folder': save_folder,
    'num_parts': num_parts,
    'n_stickers': n_stickers,
    'path_params': path_params,
    'path_aruco': path_aruco,
    'board_rows': board_rows,
    'board_cols': board_cols,
    'yfov': yfov,
    'dist': dist,
    'aruco_dict': aruco_dict,
    'aruco_start': aruco_start_id,
    'mode': mode,
    'plot_polar_info': plot_polar_info
}

json_info = json.dumps(dict_json, indent=4)

with open(config_path, "w") as json_file:
    json_file.write(json_info)
    print("Written to file: ", config_path)


