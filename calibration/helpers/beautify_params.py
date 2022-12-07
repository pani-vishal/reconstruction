""" Helper script to beautify the parameters of a calibration file. (Heavily hardcoded) """
import json
import numpy as np

path_params = f'/home/vishal/vishal/camera_calibration/calibration/outputs/params/unanno36_sorted256_theta_5_70_8_phi_0_360_16.npy'
path_save = f'/home/vishal/vishal/camera_calibration/calibration/outputs/params/unanno36_sorted256_theta_5_70_8_phi_0_360_16.json'
n_stickers = 36

params = np.load(path_params)

# Handling the ArUco sticker parameters
aruco_start_idx = len(params) - n_stickers * 2 - 1
dict_aruco = {512 + i: list(params[aruco_start_idx + i * 2: aruco_start_idx + i * 2 + 2]) for i in range(n_stickers)}
dict_aruco['aruco_z'] = params[-1]

# Handling the camera parameters
dict_params = {
    'sample_holder' : {
        'rotation' : list(params[:3]),
        'translation' : list(params[3:6]),
    },
    'camera' : { 
        'fov' : params[6],
        'principal point' : list(params[7:9]),
    },
    'sample_to_head_sphere' : { 
        'rotation' : list(params[9:12]),
        'translation' : list(params[12:15]),
    },
    'head_sphere_to_head' : {
        'rotation': 'pg_cam_matrix.T',
        'translation' : list(params[15:18]),
    },
    'head_to_camera' : {
        'rotation' : list(params[18:21]),
        'translation' : list(params[21:24]),
    },
    'aruco': dict_aruco
}

# Save json
with open(path_save, 'w') as f:
    json.dump(dict_params, f, indent=4)
    print(f'Saved json file to {path_save}')