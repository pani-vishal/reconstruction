""" Script to process and store images based on their angles and vargeo. """
import argparse
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helpers.io import *
from helpers.img_process import *
from helpers.camera_model import *
from skimage import draw

# Setting up the parser
parser = argparse.ArgumentParser(description='Process and store images based on their angles and vargeo.')
parser.add_argument('-c', '--config', type=str, help='Name of the config file (must be in ./data/configs_process/)', required=True)
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

# Handling the config file
args = parser.parse_args()
verbose = args.verbose
config_name = args.config
config_path = f'./data/configs_process/{config_name}'
# config_path = f'/home/vishal/vishal/camera_calibration/calibration/data/configs_process/{config_name}'

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
board_rows = config['board_rows']
board_cols = config['board_cols']
yfov = config['yfov']
dist = config['dist']
aruco_id_start = config['aruco_start']
aruco_dict = config['aruco_dict']
path_params = config['path_params']
path_aruco = config['path_aruco']
path_save_folder = config['path_save_folder']
save_anno = config['save_annotations']

# Handling the save folder
if not os.path.exists(path_save_folder):
    os.makedirs(path_save_folder)
    path_unannotated = os.path.join(path_save_folder, 'unannotated')
    os.makedirs(path_unannotated)
else:
    path_unannotated = os.path.join(path_save_folder, 'unannotated')
    if not os.path.exists(path_unannotated):
        os.makedirs(path_unannotated)

if save_anno:
    path_annotated = os.path.join(path_save_folder, 'annotated')
    if not os.path.exists(path_annotated):
        os.makedirs(path_annotated)

# Loading aruco dataset if exist.
if path_aruco is not None and os.path.exists(path_aruco):
    with open(path_aruco, 'rb') as f:
        dataset_aruco = pickle.load(f)
        if verbose:
            bprint("Loaded aruco parameters", div='*')

# Optimized parameters must be given
params = load_params(path_params, v=verbose)
aruco_param_start = len(params) - 2 * n_stickers - 1

# Read dataset
dataset = read_dataset(dataset_path, num_parts, v=verbose)

# Detect the aruco corners
if path_aruco is None:
    dataset_aruco, arr_aruco_dets = detect_aruco(dataset, n_stickers, aruco_id_start, aruco_dict, v=verbose)

### Intertactive part ###
# Setting up the data for responding to the prompts
thetas = []
phis = []
vargeos = []
for i, entry in enumerate(dataset):
    angles, wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners = entry[0]
    theta_i, phi_i, theta_o, phi_o = angles
    thetas.append(theta_o)
    phis.append(phi_o)
    vargeos.append(vargeo)

thetas = np.array(thetas)
phis = np.array(phis)
vargeos = np.array(vargeos)

thetas_uniq = np.unique(thetas)
phis_uniq = np.unique(phis)
vargeos_uniq = np.unique(vargeos)

# Prompting the user
mode = None
valid_responses = ['a', 'm', 's', 'all', 'multi', 'single']
while mode not in valid_responses:
    print("Image selection mode: ((a)ll, (m)ulti, (s)ingle)")
    mode = str(input("Enter mode: ")).lower()
    if mode not in valid_responses:
        print(f"!!!! Invalid response. Please try again !!! Must be in {valid_responses}")

# If above passes, then default mode is 'all'
bprint(f"Mode: {mode}", div='*')
thetas_proc = thetas_uniq.copy()
phis_proc = phis_uniq.copy()
vargeos_proc = vargeos_uniq.copy()

if not (mode == 'a' or mode == 'all'):
    thetas_proc = np.full(1, np.nan)
    phis_proc = np.full(1, np.nan)
    vargeos_proc = np.full(1, np.nan)
if mode == 's' or mode == 'single':
    while not np.all(np.isin(thetas_proc, thetas_uniq)):
        print("Available thetas: ", thetas_uniq)
        raw_res = str(input("Enter thetas to process (separated by spaces): ")).split(" ")
        thetas_proc = np.array([float(r) for r in raw_res])
        if not np.all(np.isin(thetas_proc, thetas_uniq)):
            print(f"!!!! Invalid response. Please try again !!! Must be in {thetas_uniq}")

    while not np.all(np.isin(phis_proc, phis_uniq)):
        print("Available phis: ", phis_uniq)
        raw_res = str(input("Enter phis to process (separated by spaces): ")).split(" ")
        phis_proc = np.array([float(r) for r in raw_res])
        if not np.all(np.isin(phis_proc, phis_uniq)):
            print(f"!!!! Invalid response. Please try again !!! Must be in {phis_uniq}")

    while not np.all(np.isin(vargeos_proc, vargeos_uniq)):
        print("Available vargeos: ", vargeos_uniq)
        raw_res = str(input("Enter vargeos to process (separated by spaces): ")).split(" ")
        vargeos_proc = np.array([float(r) for r in raw_res])
        if not np.all(np.isin(vargeos_proc, vargeos_uniq)):
            print(f"!!!! Invalid response. Please try again !!! Must be in {vargeos_uniq}")

elif mode == 'm' or mode == 'multi':
    flag_done = False
    thetas_temp = []
    phis_temp = []
    vargeos_temp = []
    while not flag_done:
        theta_choice = np.nan
        while theta_choice not in thetas_uniq:
            print("Available thetas: ", thetas_uniq)
            theta_choice = float(input("Enter theta to process: "))
            if theta_choice not in thetas_uniq:
                print(f"!!!! Invalid response. Please try again !!! Must be in {thetas_uniq}")
        thetas_temp.append(theta_choice)
        print()

        phi_choice = np.nan
        theta_mask = thetas == theta_choice
        phis_masked = np.unique(phis[theta_mask])
        while phi_choice not in phis_uniq:
            print("Available phis: ", phis_masked)
            phi_choice = float(input("Enter phi to process: "))
            if phi_choice not in phis_masked:
                print(f"!!!! Invalid response. Please try again !!! Must be in {phis_masked}")
        phis_temp.append(phi_choice)
        print()

        vargeo_choice = np.nan
        all_mask = (theta_mask * (phis == phi_choice))
        vargeos_masked = np.unique(vargeos[all_mask])
        while vargeo_choice not in vargeos_uniq:
            print("Available vargeos: ", vargeos_masked)
            vargeo_choice = float(input("Enter vargeo to process: "))
            if vargeo_choice not in vargeos_masked:
                print(f"!!!! Invalid response. Please try again !!! Must be in {vargeos_masked}")
        vargeos_temp.append(vargeo_choice)
        print()

        print(f"Processing thetas: {thetas_temp}")
        print(f"Processing phis: {phis_temp}")
        print(f"Processing vargeos: {vargeos_temp}")
        print("Done? (y/n)")
        flag_done = str(input("Enter response: ")).lower() == 'y'
        if not flag_done:
            bprint(None, div='*')

    thetas_proc = np.array(thetas_temp)
    phis_proc = np.array(phis_temp)
    vargeos_proc = np.array(vargeos_temp)

bprint()
print(f"Processing thetas: {thetas_proc}")
print(f"Processing phis: {phis_proc}")
print(f"Processing vargeos: {vargeos_proc}")
### ----------------- ###


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


# Processing the images
print()
for index, entry in enumerate(dataset):
    angles, wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners = entry[0] # No image required
    theta_i, phi_i, theta_o, phi_o = angles
    
    # Filter images
    if not(theta_o in thetas_proc and phi_o in phis_proc and vargeo in vargeos_proc):
        continue
    
    # if no charuco corner detected only output the image after being transformed
    flag_ch = True
    if ch_ids is None:
        flag_ch = False
    
    image = np.array(entry[1])
    rows, cols = image.shape[:2]
    aruco_ids, aruco_corners = dataset_aruco[index]
    preds_corners, preds_aruco = compute_position(np.arange(28*17), aruco_ids, pg_cam_matrix, params, 
                                                  camera_config, aruco_param_start, aruco_id_start)
    if verbose:
            print(f'{"*"*5}\tTransforming image with angles {angles} and vargeo {vargeo}\t{"*"*5}')

    if len(preds_aruco) == 0:
        if verbose:
            print(f"{'!'*10}\tCan't transform image, no aruco detected\t{'!'*10}")
        continue

    # Shift
    ## Calculate aruco center from corners
    aruco_center = np.mean(aruco_corners, axis=1)
    d_aruco = aruco_center - preds_aruco
    shift = np.mean(d_aruco, axis=0)
    shift_mat = np.float32([[1,0,-shift[0]],[0,1,-shift[1]]])

    # Homography
    shift_homo, _ = cv2.findHomography(aruco_center-shift, preds_aruco)

    # Calculate error
    if flag_ch:
        preds = preds_corners[ch_ids]
        targs = ch_corners[:, 0, :]
        ## Shift
        targs = (shift_mat @ np.hstack((targs, np.ones((targs.shape[0],1)))).T).T
        err_shift = reproj_err(preds, targs)

        ## Homography
        targs = mmul_p(shift_homo, np.hstack((targs, np.ones((len(targs),1)))))[:, :2]
        err = reproj_err(preds, targs)

    aruco_center = (shift_mat @ np.hstack((aruco_center, np.ones((aruco_center.shape[0],1)))).T).T
    aruco_center = mmul_p(shift_homo, np.hstack((aruco_center, np.ones((len(aruco_center),1)))))[:, :2]
    

    ## Transform image
    image = cv2.warpAffine(image, shift_mat, (cols,rows))
    image = cv2.warpPerspective(image, shift_homo, (cols,rows))

    # Save current image
    path_save_img = os.path.join(path_unannotated, f'{theta_i}_{phi_i}_{theta_o}_{phi_o}_{vargeo}.png')
    plt.imsave(path_save_img, image)
    if verbose:
        print(path_save_img)
    plt.close()
    
    # Annotate image
    if save_anno and flag_ch:
        # Redundant, but just in case
        image_anno = image.copy()

        # Targets
        ## Charuco
        for i, p in enumerate(targs):
            rr, cc = draw.ellipse(int(p[1]), int(p[0]), 7, 7, shape=image_anno.shape)
            image_anno[rr, cc, 0] = 43
            image_anno[rr, cc, 1] = 222
            image_anno[rr, cc, 2] = 67
            if len(rr) > 0:
                cv2.putText(
                    image_anno, #numpy array on which text is written
                    f"{i}", #text
                    (cc[0], rr[0]), #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    1, #font size
                    (209, 80, 0, 255), #font color
                    3) #font stroke
        ## Aruco
        for i, p in enumerate(aruco_center):
            rr, cc = draw.ellipse(int(p[1]), int(p[0]), 10, 10, shape=image_anno.shape)
            image_anno[rr, cc, 2] = 43
            image_anno[rr, cc, 1] = 222
            image_anno[rr, cc, 0] = 120
            cv2.putText(
                    image_anno, #numpy array on which text is written
                    f"{aruco_ids[i]}", #text
                    (cc[0], rr[0]), #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    1, #font size
                    (209, 20, 25, 255), #font color
                    3) #font stroke

        # Predictions
        ## Charuco
        for i, p in enumerate(preds_corners[ch_ids]):
            rr, cc = draw.ellipse(int(p[1]), int(p[0]), 7, 7, shape=image_anno.shape)
            image_anno[rr, cc, 0] = 255
        ## Aruco
        for i, p in enumerate(preds_aruco):
            rr, cc = draw.ellipse(int(p[1]), int(p[0]), 10, 10, shape=image_anno.shape)
            image_anno[rr, cc, 0] = 225

        # Error
        cv2.putText(
                    image_anno,
                    f"Reproj. Err.: {err:.3f}", #text
                    (200, 200), #position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                    2, #font size
                    (209, 20, 25, 255), #font color
                    3) #font stroke

        path_save_img = os.path.join(path_annotated, f'{theta_i}_{phi_i}_{theta_o}_{phi_o}_{vargeo}.png')
        plt.imsave(path_save_img, image_anno)
        plt.close()