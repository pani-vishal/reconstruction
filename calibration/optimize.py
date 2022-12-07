""" Script to optimize and store the parameters for a given dataset """
import argparse
import json
import os
from pickletools import optimize
import numpy as np
from helpers.io import *
from helpers.camera_model import *
from helpers.img_process import *
from tqdm import tqdm

from functools import partial
from scipy import optimize as opt

# Setting up the parser
parser = argparse.ArgumentParser(description='Optimize and store the parameters for a given dataset')
parser.add_argument('-c', '--config', type=str, help='Name of the config file (must be in ./data/configs_optimize/)', required=True)
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

# Handling the config file
args = parser.parse_args()
verbose = args.verbose
config_name = args.config
config_path = f'./data/configs_optimize/{config_name}'

with open(config_path, 'r') as f:
    if verbose:
        bprint(f"Reading config file: {config_path}\n", div='*')
    config = json.load(f)
    if verbose:
        bprint(config, ppr=True)

# Inputs
dataset_path = config['dataset']
num_parts = config['num_parts']
n_stickers = config['n_stickers']
save = config['save']
board_rows = config['board_rows']
board_cols = config['board_cols']
yfov = config['yfov']
dist = config['dist']
maxiter = config['maxiter']
aruco_id_start = config['aruco_start']
aruco_dict = config['aruco_dict']
path_params = config['path_params']
path_aruco = config['path_aruco']


## Loading aruco dataset if exist.
if path_aruco is not None and os.path.exists(path_aruco):
    with open(path_aruco, 'rb') as f:
        dataset_aruco = pickle.load(f)
        if verbose:
            bprint("Loaded aruco parameters", div='*')

# Read dataset
dataset = read_dataset(dataset_path, num_parts, v=verbose)

# Detect the aruco corners
if path_aruco is None:
    dataset_aruco, arr_aruco_dets = detect_aruco(dataset, n_stickers, aruco_id_start, aruco_dict, v=verbose)

# Optimize the parameters
im_height, im_width = dataset[0][1].shape[:2]
camera_config = {
    'board_dim' : [board_rows, board_cols],
    'width'     : im_width,
    'height'    : im_height,
    'aspect'    : im_width / im_height,
    'yfov'      : yfov, # 100 mm lens. OpenCV estimates 110.6, which comes out to 12.5 yfov
    'dist'      : dist, # Initial camera distance in mm
}

# Setting up the default parameters for the optimization
default_params = np.array([
    0, 0, 0, # 0, 1, 2
    -camera_config['board_dim'][0] * 5.0,  # 3
    -camera_config['board_dim'][1] * 5.0,  # 4
    0,                              # 5
    12.453840727371169,             # 6
    3.29086869e+03, 2.11082254e+03, # 7, 8
    0, 0, 0,                        # 9, 10, 11
    0, 0, 0,                        # 12, 13, 14
    0, 0, -camera_config['dist'],          # 15, 16, 17
    0, 0, 0,                        # 18, 19, 20
    0, 0, 0,                        # 21, 22, 23
])
aruco_param_start = len(default_params)
default_params_aruco = np.zeros((n_stickers * 2 + 1,))
default_params = np.concatenate((default_params, default_params_aruco))
# Don't want to display default parameters if the user wants to use existing parameters
if verbose and path_params is None:
    bprint(f"Default paramaters (len: {len(default_params)})", div='*')
    bprint(default_params)

# If parameters already exist, use them as the starting point
if path_params is not None and os.path.exists(path_params):
    default_params = np.load(path_params)
    if verbose:
        print(f"Loading parameters from: {path_params}")
        # TODO: Beautify this (numbers dont make any sense!!)
        bprint(f"Paramaters (len: {len(default_params)})", div='*')
        bprint(default_params)


# Optimize the parameters
## Setting up the optimization partials
objective_partial = partial(objective, dataset=dataset,
                                       config=camera_config,
                                       aruco_param_start=aruco_param_start,
                                       dataset_aruco=dataset_aruco)

if verbose:
    result = opt.minimize(objective_partial, default_params,
                          options={'disp': True, 'maxfun': 1000000, 'maxiter': maxiter},
                          method='L-BFGS-B')
    bprint()
else:
    result = opt.minimize(objective_partial, default_params,
                          options={'maxfun': 1000000, 'maxiter': maxiter},
                          method='L-BFGS-B')


# Save the parameters
with open(save, 'wb') as f:
    np.save(f, np.array(result.x))
    if verbose:
        bprint(f"Saved parameters to {save}", div='--X--')
