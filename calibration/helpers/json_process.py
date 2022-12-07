""" Helper script to write JSON config files for the process.py script. """
import json
from cv2 import aruco

# Inputs (always change these)
dataset_folder = '/home/vpani/rgl/datasets'
dataset_name = 'unanno36_sorted256_theta_5_70_8_phi_0_360_16'
path_params = f'/home/vpani/rgl/recon/calibration/outputs/params/npp_{dataset_name}.npy'
path_save_folder = f'/home/vpani/rgl/recon/calibration/outputs/params/{dataset_name}'
num_parts = 64
n_stickers = 36
save_annotations = True

# json save_location (always change this)
config_name = 'process36.json'
config_path = f'/home/vpani/rgl/recon/calibration/data/configs_process/{config_name}'

# Inputs (change these if needed) 
## None if aruco stickers haven't been detected yet
path_aruco = f'/home/vpani/rgl/recon/calibration/data/{dataset_name}_aruco.pkl' 
board_rows = 29
board_cols = 18
yfov = 12.5
dist = 901.4990809603678
aruco_dict = aruco.DICT_5X5_1000
aruco_start_id = 512

# Json dicitonary
dict_json = {
    'dataset': f'{dataset_folder}/{dataset_name}',
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
    'path_save_folder': path_save_folder,
    'save_annotations': save_annotations,
}

json_info = json.dumps(dict_json, indent=4)

with open(config_path, "w") as json_file:
    json_file.write(json_info)
    print("Written to file: ", config_path)
