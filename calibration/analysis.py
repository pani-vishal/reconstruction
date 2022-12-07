""" Script to analyze the results of a calibration process. """
import argparse
import json
import os
import sys
import numpy as np
import cv2
import copy
from helpers.io import *
from helpers.img_process import *
from helpers.camera_model import *
from helpers.visualize import *

# Setting up the parser
parser = argparse.ArgumentParser(description='Process and store images based on their angles and vargeo.')
parser.add_argument('-c', '--config', type=str, help='Name of the config file (must be in ./data/configs_process/)', required=True)
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

# Handling the config file
args = parser.parse_args()
verbose = args.verbose
config_name = args.config
config_path = f'./data/configs_analysis/{config_name}'
# config_path = f'/home/vishal/vishal/camera_calibration/calibration/data/configs_process/{config_name}'

with open(config_path, 'r') as f:
    if verbose:
        bprint(f"Reading config file: {config_path}\n", div='*')
    config = json.load(f)
    if verbose:
        bprint(config, ppr=True)

# Dict mode
dict_mode = {'none': 0, 'shift': 1, 'homography': 2}

# Inputs
dataset_path = config['dataset']
num_parts = config['num_parts']
n_stickers = config['n_stickers']
board_rows = config['board_rows']
board_cols = config['board_cols']
yfov = config['yfov']
dist = config['dist']
aruco_id_start = config['aruco_start']
aruco_dict = config['aruco_dict']
path_params = config['path_params']
mode = dict_mode[config['mode']]
save_folder = config['save_folder']
plot_polar_info = config['plot_polar_info']
path_aruco = config['path_aruco']

# Loading aruco dataset if exist.
if path_aruco is not None and os.path.exists(path_aruco):
    with open(path_aruco, 'rb') as f:
        dataset_aruco = pickle.load(f)
        if verbose:
            bprint("Loaded aruco parameters", div='*')

# Save path must exists.
if not os.path.exists(save_folder):
    print(f"Error: {save_folder} does not exist. Please change the path or make a folder first.")
    sys.exit(1)

# Optimized parameters must be given.
params = load_params(path_params, verbose)

if verbose:
    bprint(f"Analysis mode: {config['mode']} ({mode})", div='*')
aruco_param_start = len(params) - 2 * n_stickers - 1

# Read dataset
dataset = read_dataset(dataset_path, num_parts, v=verbose)

# Detect the aruco corners
if mode > 0:
    if path_aruco is None:
        dataset_aruco, arr_aruco_dets = detect_aruco(dataset, n_stickers, aruco_id_start, aruco_dict, v=verbose)

# Camera configuration
im_height, im_width = dataset[0][1].shape[:2]
camera_config = {
    'board_dim' : [board_rows, board_cols],
    'width'     : im_width,
    'height'    : im_height,
    'aspect'    : im_width / im_height,
    'yfov'      : yfov, # 100 mm lens. OpenCV estimates 110.6, which comes out to 12.5 yfov
    'dist'      : dist, # Initial camera distance in mm
}

# Analysis
dict_loss = {}
dict_ids = {}
dict_angles = {}
all_vargeos = []
for index, entry in enumerate(dataset):
    angles, wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners = entry[0] 
    theta_i, phi_i, theta_o, phi_o = angles
    key = f"thetai_{theta_i}_phii_{phi_i}_thetao_{theta_o}_phio_{phi_o}_vargeo_{vargeo}"
    all_vargeos.append(vargeo)

    if ch_ids is None:
        continue
    
    # Consider ArUco stickers only if mode is not "none".
    if mode > 0:
        aruco_ids, aruco_corners = dataset_aruco[index]
        preds_corners, preds_aruco = compute_position(np.arange(28*17), aruco_ids, pg_cam_matrix, params, 
                                                    camera_config, aruco_param_start, aruco_id_start)
    else:
        preds_corners, preds_aruco = compute_position(np.arange(28*17), None, pg_cam_matrix, params, 
                                            camera_config, aruco_param_start, aruco_id_start)
    # Shift
    if mode > 0:
        ## Calculate aruco center from corners
        aruco_center = np.mean(aruco_corners, axis=1)
        d_aruco = aruco_center - preds_aruco
        shift = np.mean(d_aruco, axis=0)
        M = np.float32([[1,0,-shift[0]],[0,1,-shift[1]]])
    # Homography
    if mode > 1:
        H, _ = cv2.findHomography(aruco_center-shift, preds_aruco)

    ## Calculate error
    preds = preds_corners[ch_ids]
    targs = ch_corners[:, 0, :]
    if mode > 0:
        targs = (M @ np.hstack((targs, np.ones((targs.shape[0],1)))).T).T

    if mode > 1:
        targs = mmul_p(H, np.hstack((targs, np.ones((len(targs),1)))))[:, :2]

    d = preds - targs
    dist = np.sqrt(np.sum(d*d, axis=-1))

    dict_loss[key] = dist
    dict_ids[key] = ch_ids
    dict_angles[key] = [*list(angles), vargeo]

    # Global stats
    arr_global_losses = []
    for _, losses in dict_loss.items():
        arr_global_losses.extend(list(losses))
    arr_global_losses = np.array(arr_global_losses)
    global_median = np.median(arr_global_losses)
    global_mean = np.mean(arr_global_losses)
    global_min = np.min(arr_global_losses)
    global_max = np.max(arr_global_losses)
    global_std = np.std(arr_global_losses)
    
    # Best and worst image
    best_k = best_key = None
    best_error = 1e8
    worst_k = worst_key = None
    worst_error = 0
    for keyl, losses in dict_loss.items():
        loss_mean = losses.mean()
        if loss_mean > worst_error:
            worst_error = loss_mean
            worst_key = keyl
        if loss_mean < best_error:
            best_error = loss_mean
            best_key = keyl

    # Best stats
    arr_best_losses = np.array(dict_loss[best_key])
    best_median = np.median(arr_best_losses)
    best_mean = np.mean(arr_best_losses)
    best_min = np.min(arr_best_losses)
    best_max = np.max(arr_best_losses)
    best_std = np.std(arr_best_losses)

    # Worst stats
    arr_worst_losses = np.array(dict_loss[worst_key])
    worst_median = np.median(arr_worst_losses)
    worst_mean = np.mean(arr_worst_losses)
    worst_min = np.min(arr_worst_losses)
    worst_max = np.max(arr_worst_losses)
    worst_std = np.std(arr_worst_losses)

    dict_data = {}
    dict_data['global'] = {
        'arr': arr_global_losses,
        'mean': global_mean,
        'std': global_std,
        'median': global_median,
        'min': global_min,
        'max': global_max
    }

    dict_data['best'] = {
        'arr': arr_best_losses,
        'mean': best_mean,
        'std': best_std,
        'median': best_median,
        'min': best_min,
        'max': best_max,
        'best_key': best_key,
        'dets': len(dict_ids[best_key])
    }

    dict_data['worst'] = {
        'arr': arr_worst_losses,
        'mean': worst_mean,
        'std': worst_std,
        'median': worst_median,
        'min': worst_min,
        'max': worst_max,
        'worst_key': worst_key,
        'dets': len(dict_ids[worst_key])
    }

if verbose:
    bprint(dict_data, ppr=True)

# Saving stats as json file
dict_data_save = copy.deepcopy(dict_data)
dict_data_save['global'].pop('arr')
dict_data_save['best'].pop('arr')
dict_data_save['worst'].pop('arr')
json_info = json.dumps(dict_data_save, indent=4)
## Hacky, ik
json_path = f'{save_folder}../analysis_stats/stats_{config_name.split(".")[0]}_{config["mode"]}.json'
with open(json_path, "w") as json_file:
    json_file.write(json_info)
    bprint(f"Written to file: {json_path}")

# Plots
plot_violin(dict_data, save_folder, config['mode'], config_name, v=verbose)

all_vargeos = np.unique(np.array(all_vargeos))
if plot_polar_info is not None:
    for vargeo in all_vargeos:
        plot_polar(dict_loss, dict_angles, vargeo, plot_polar_info, save_folder,
                   config['mode'], config_name, v=verbose)