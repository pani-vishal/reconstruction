""" Script containing all the helper functions for image processing"""
import numpy as np
from tqdm import tqdm
from cv2 import aruco
from .io import *

def detect_aruco(dataset: dict, n_stickers: int, aruco_start: int=512, aruco_dict: int=aruco.DICT_5X5_1000, v: bool=False):
    """ Detect the aruco corners in the dataset.
        
        Input:
            dataset:        dict, containing the charuco dataset (must also contain the image of the individual capture)
            n_stickers:     int, number of aruco stickers on the sample holder
            aruco_start:    int, index of the first parameter of the aruco stickers
            aruco_dict:     int, aruco dictionary
            v:              bool, if True, output information
        Output:
            Returns a list containing (aruco_ids (#dets), aruco_corners (#dets, 4, 2)) and number of detections for each image.
    """
    # Setting up the ArUco dictionary
    aruco_dict = aruco.Dictionary_get(aruco_dict)
    aruco_params = aruco.DetectorParameters_create()
    aruco_params.adaptiveThreshWinSizeMax = 50
    aruco_params.adaptiveThreshWinSizeStep = 4
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX 
    aruco_params.cornerRefinementMaxIterations = 100
    aruco_params.cornerRefinementMinAccuracy = 0.01

    # Array of all the possible marker ids
    marker_ids = np.arange(aruco_start, aruco_start + n_stickers)
    
    # Detecting the aruco corners
    dataset_aruco = []
    arr_aruco_dets = []
    for entry in tqdm(dataset, desc="Detecting ArUco stickers", disable=not(v)):
        image = entry[1]
        (corners, ids, _) = aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        corners = np.squeeze(corners)
        ids = np.squeeze(ids)

        # Mask out the "charuco" ids, only ArUco ids are needed
        mask = np.isin(ids, marker_ids)
        ids = ids[mask]
        corners = corners[mask]

        arr_aruco_dets.append(len(ids))

        if len(ids) == 0:
            dataset_aruco.append([None, None])
        else:
            dataset_aruco.append([ids, corners])
    if v:
        arr_aruco_dets = np.array(arr_aruco_dets)
        msg = f'ArUco dets mean: {arr_aruco_dets.mean()}, min: {arr_aruco_dets.min()}, max: {arr_aruco_dets.max()}'
        bprint(msg)

    return dataset_aruco, arr_aruco_dets