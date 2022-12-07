""" Helper functions for the camera model. """
import numpy as np
from .linear_alg import *
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

def build_aruco_pos(ids: np.ndarray, params: np.ndarray, param_start: int, aruco_id_start: int=512, param_end: int=-1) -> np.array:
    """
        Builds an array of shape (n_det_stickers, 4) that contain the current position of the stickers

        Input:
            ids:            array of shape (n_det_stickers,) containing the id of the sticker
            params:         array of shape (n_params,) containing the parameters of the model
            param_start:    int, index of the first parameter of the aruco stickers
            aruco_id_start: int, id of the first aruco sticker
            param_end:      int, index of the last parameter of the aruco stickers

        Output:
            params_aruco: array of shape (n_det_stickers, 4) containing the current position of the stickers
    """
    # [:-1] since last param is the z offset of the stickers and its common for all
    params_aruco = params[param_start:-1]
    # Get the ids to be 0 indexed and multiplied by 2 since each sticker has 2 params
    ids_shifted_x = (ids - aruco_id_start) * 2
    ids_shifted_y = ids_shifted_x + 1
    # Interleave the arrays to get the proper order of the param ids and reshape to [#, 2]
    params_aruco_ids = np.empty((ids_shifted_x.size*2,), dtype=ids_shifted_x.dtype)
    params_aruco_ids[0::2] = ids_shifted_x
    params_aruco_ids[1::2] = ids_shifted_y
    params_aruco = params_aruco[params_aruco_ids]
    params_aruco = params_aruco.reshape(-1, 2)
    # Build the array of shape [n_stickers, 4] that contain the current position of the stickers
    params_aruco = np.hstack((params_aruco, 
                              np.full((params_aruco.shape[0], 1), params[param_end]),
                              np.ones((params_aruco.shape[0], 1))))

    return params_aruco


def compute_position(index: np.ndarray, index_aruco: np.ndarray, pg_cam_matrix: np.ndarray, params: np.ndarray, 
                     config: dict, aruco_param_start: int, aruco_id_start: int=512, aruco_param_end: int=-1) -> tuple[np.ndarray, np.ndarray]:
    """
        Computes the position of the stickers and the aruco markers.

        Input:
            index:              array of shape (n_det_charuco,) containing the index of the charuco corners
            index_aruco:        array of shape (n_det_aruco,) containing the index of the aruco stickers
                                or None if no aruco stickers are detected/provided
            pg_cam_matrix:      array of shape (4, 4) containing the camera matrix
            params:             array of shape (n_params,) containing the parameters of the model
            config:             dict, configuration of the camera
            aruco_param_start:  int, index of the first parameter of the aruco stickers
            aruco_id_start:     int, id of the first aruco sticker
            aruco_param_end:    int, index of the last parameter of the aruco stickers

        Output:
            pts_grid:   array of shape (n_det_charuco, 2) containing the position of the charuco corners
            pts_aruco:  array of shape (n_det_aruco, 2) containing the position of the aruco stickers
                        (if no aruco stickers are detected/provided, it is None)
    """
    index, params = np.asarray(index), np.asarray(params)
    n, m = config['board_dim'][0] - 1, len(index)

    # Compute grid pixel positions
    ij = np.flip(np.column_stack((10.0 * (index % n)+10, 10.0 * (index // n)+10, np.zeros(m), np.ones(m))), axis=0)

    # Establish grid positions in sample space
    ij_to_sample = rot_scipy(params[:3])
    ij_to_sample[0:3, 3] = params[3:6]

    # Perspective projection (Camera to pixel space)
    cam_proj = perspective_matrix(deg2rad(params[6]), 0.1, 1000, config['aspect'])
    cam_scr  = np.diag((config['width'] / 2, config['height'] / 2, 1, 1))
    cam_scr[0:2, 3] = (config['width'] / 2, config['height'] / 2)
    cam_to_pixel = cam_scr @ cam_proj @ np.array(
                [[0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    # Sample to head sphere space
    sample_to_head_sphere = rot_scipy(params[9:12])
    sample_to_head_sphere[0:3, 3] = params[12:15]

    # Head sphere to head space
    head_transl = np.eye(4)
    head_transl[:3, 3] = params[15:18]
    head_sphere_to_head = head_transl @ pg_cam_matrix.T

    # Head to camera space
    head_to_cam = rot_scipy(params[18:21])
    head_to_cam[:3, 3] = params[21:24]
    
    # Sample to pixel space
    sample_to_pixel = cam_to_pixel @ head_to_cam @ head_sphere_to_head @ sample_to_head_sphere

    # Final transformation
    ij_to_pixel = sample_to_pixel @ ij_to_sample

    # Positions of charuco corners
    pts_grid = mmul_p(ij_to_pixel, ij)[..., 0:2]
    
    # If no aruco stickers are detected/provided, return None
    if index_aruco is None:
        return pts_grid, None

    # Else compute the position of the aruco stickers
    pos_aruco = build_aruco_pos(index_aruco, params, 
                                param_start=aruco_param_start, 
                                aruco_id_start=aruco_id_start, 
                                param_end=aruco_param_end)
    pts_aruco = mmul_p(sample_to_pixel, pos_aruco)[..., 0:2]

    return pts_grid, pts_aruco


def compute_position_mitsuba(index: np.ndarray, index_aruco: np.ndarray, pg_cam_matrix: np.ndarray, params: np.ndarray, 
                     config: dict, aruco_param_start: int, aruco_id_start: int=512, aruco_param_end: int=-1) -> tuple[np.ndarray, np.ndarray]:
    """
        Computes the position of the stickers and the aruco markers.

        Input:
            index:              array of shape (n_det_charuco,) containing the index of the charuco corners
            index_aruco:        array of shape (n_det_aruco,) containing the index of the aruco stickers
                                or None if no aruco stickers are detected/provided
            pg_cam_matrix:      array of shape (4, 4) containing the camera matrix
            params:             array of shape (n_params,) containing the parameters of the model
            config:             dict, configuration of the camera
            aruco_param_start:  int, index of the first parameter of the aruco stickers
            aruco_id_start:     int, id of the first aruco sticker
            aruco_param_end:    int, index of the last parameter of the aruco stickers

        Output:
            pts_grid:   array of shape (n_det_charuco, 2) containing the position of the charuco corners
            pts_aruco:  array of shape (n_det_aruco, 2) containing the position of the aruco stickers
                        (if no aruco stickers are detected/provided, it is None)
    """
    index, params = np.asarray(index), np.asarray(params)
    n, m = config['board_dim'][0] - 1, len(index)

    # Compute grid pixel positions
    ij = np.flip(np.column_stack((10.0 * (index % n)+10, 10.0 * (index // n)+10, np.zeros(m), np.ones(m))), axis=0)

    # Establish grid positions in sample space
    ij_to_sample = rot_scipy(params[:3])
    ij_to_sample[0:3, 3] = params[3:6]

    # Perspective projection (Camera to pixel space)
    fov_y = params[6]
    fov_x = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(fov_y / 2)) * config['aspect']))
    cam_proj = np.array(mi.perspective_projection(mi.ScalarVector2i(config['width'], config['height']), 
                                             mi.ScalarVector2i(config['width'], config['height']), 
                                             mi.ScalarVector2i(0, 0), 
                                             fov_x=-fov_x*2, near_clip=0.1, far_clip=1000).matrix)
    cam_proj = np.squeeze(cam_proj)
    cam_scr  = np.diag((config['width'] / 2, config['height'] / 2, 1, 1))
    cam_scr[0:2, 3] = (config['width'] / 2, config['height'] / 2)
    cam_to_pixel = cam_scr @ cam_proj @ np.array(
                [[0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    # Sample to head sphere space
    sample_to_head_sphere = rot_scipy(params[9:12])
    sample_to_head_sphere[0:3, 3] = params[12:15]

    # Head sphere to head space
    head_transl = np.eye(4)
    head_transl[:3, 3] = params[15:18]
    head_sphere_to_head = head_transl @ pg_cam_matrix.T

    # Head to camera space
    head_to_cam = rot_scipy(params[18:21])
    head_to_cam[:3, 3] = params[21:24]
    
    # Sample to pixel space
    sample_to_pixel = cam_to_pixel @ head_to_cam @ head_sphere_to_head @ sample_to_head_sphere

    # Final transformation
    ij_to_pixel = sample_to_pixel @ ij_to_sample

    # Positions of charuco corners
    pts_grid = mmul_p(ij_to_pixel, ij)[..., 0:2]
    
    # If no aruco stickers are detected/provided, return None
    if index_aruco is None:
        return pts_grid, None

    # Else compute the position of the aruco stickers
    pos_aruco = build_aruco_pos(index_aruco, params, 
                                param_start=aruco_param_start, 
                                aruco_id_start=aruco_id_start, 
                                param_end=aruco_param_end)
    pts_aruco = mmul_p(sample_to_pixel, pos_aruco)[..., 0:2]

    return pts_grid, pts_aruco



def objective(params: np.ndarray, dataset: dict, config: dict, aruco_param_start: int=24, aruco_id_start: int=512, 
              aruco_param_end: int=-1, dataset_aruco: dict=None, square_norm: bool=False) -> float:
    """
        Objective function for the optimization procedure.

        Input:
            params:             array of shape (n_params,) containing the parameters of the model
            dataset:            dict, containing the charuco dataset (may also contain the image of the individual capture)
            config:             dict, configuration of the camera
            aruco_param_start:  int, index of the first parameter of the aruco stickers
            aruco_id_start:     int, id of the first aruco sticker
            aruco_param_end:    int, index of the last parameter of the aruco stickers
            dataset_aruco:      list ((aruco_ids (#dets), aruco_corners (#dets, 4, 2))), containing the aruco dataset.
            square_norm:        bool, if True, the norm is squared
        
        Output:
            float, value of the objective function
    """
    accum, pt_count = 0, 0

    for i, entry in enumerate(dataset):
        angles, wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners = entry[0] # No image required
        theta_i, phi_i, theta_o, phi_o = angles
        
        if ch_ids is None:
            continue
        
        # Pass None instead of aruco_ids if no aruco stickers are detected/provided
        if dataset_aruco is None:
            # TODO: The np.arange(28*17) can be optimized to only the detected charuco corners
            preds_corners, preds_aruco = compute_position(np.arange(28*17), None, pg_cam_matrix, params, config, aruco_param_start, aruco_id_start, aruco_param_end)
        else:
            aruco_ids, aruco_corners = dataset_aruco[i]         
            preds_corners, preds_aruco = compute_position(np.arange(28*17), aruco_ids, pg_cam_matrix, params, config, aruco_param_start, aruco_id_start, aruco_param_end)   
        
        # Don't consider aruco loss if no aruco stickers are detected/provided
        if dataset_aruco is None or preds_aruco is None:
            d = preds_corners[ch_ids] - ch_corners[:, 0, :]
            dist2 = np.sum(d*d, axis=-1)
            pt_count += len(ch_ids)
            accum += dist2.sum() if square_norm else np.sqrt(dist2).sum()
        else:
            aruco_center = np.mean(aruco_corners, axis=1)
            d_aruco = preds_aruco - aruco_center
            dist2_aruco = np.sum(d_aruco*d_aruco, axis=-1)
            d_corners = preds_corners[ch_ids] - ch_corners[:, 0, :]
            dist2_corners = np.sum(d_corners*d_corners, axis=-1)
            # Loss is just the mean distance between the charuco corners and aruco centers detections and predictions
            pt_count += len(ch_ids) + len(aruco_ids)
            accum += dist2_corners.sum() + dist2_aruco.sum() if square_norm else np.sqrt(dist2_corners).sum() + np.sqrt(dist2_aruco).sum()

    return accum / pt_count


def objective_mitsuba(params: np.ndarray, dataset: dict, config: dict, aruco_param_start: int=24, aruco_id_start: int=512, 
              aruco_param_end: int=-1, dataset_aruco: dict=None, square_norm: bool=False) -> float:
    """
        Objective function for the optimization procedure.

        Input:
            params:             array of shape (n_params,) containing the parameters of the model
            dataset:            dict, containing the charuco dataset (may also contain the image of the individual capture)
            config:             dict, configuration of the camera
            aruco_param_start:  int, index of the first parameter of the aruco stickers
            aruco_id_start:     int, id of the first aruco sticker
            aruco_param_end:    int, index of the last parameter of the aruco stickers
            dataset_aruco:      list ((aruco_ids (#dets), aruco_corners (#dets, 4, 2))), containing the aruco dataset.
            square_norm:        bool, if True, the norm is squared
        
        Output:
            float, value of the objective function
    """
    accum, pt_count = 0, 0

    for i, entry in enumerate(dataset):
        angles, wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners = entry[0] # No image required
        theta_i, phi_i, theta_o, phi_o = angles
        
        if ch_ids is None:
            continue
        
        # Pass None instead of aruco_ids if no aruco stickers are detected/provided
        if dataset_aruco is None:
            # TODO: The np.arange(28*17) can be optimized to only the detected charuco corners
            preds_corners, preds_aruco = compute_position_mitsuba(np.arange(28*17), None, pg_cam_matrix, params, config, aruco_param_start, aruco_id_start, aruco_param_end)
        else:
            aruco_ids, aruco_corners = dataset_aruco[i]         
            preds_corners, preds_aruco = compute_position_mitsuba(np.arange(28*17), aruco_ids, pg_cam_matrix, params, config, aruco_param_start, aruco_id_start, aruco_param_end)   
        
        # Don't consider aruco loss if no aruco stickers are detected/provided
        if dataset_aruco is None or preds_aruco is None:
            d = preds_corners[ch_ids] - ch_corners[:, 0, :]
            dist2 = np.sum(d*d, axis=-1)
            pt_count += len(ch_ids)
            accum += dist2.sum() if square_norm else np.sqrt(dist2).sum()
        else:
            aruco_center = np.mean(aruco_corners, axis=1)
            d_aruco = preds_aruco - aruco_center
            dist2_aruco = np.sum(d_aruco*d_aruco, axis=-1)
            d_corners = preds_corners[ch_ids] - ch_corners[:, 0, :]
            dist2_corners = np.sum(d_corners*d_corners, axis=-1)
            # Loss is just the mean distance between the charuco corners and aruco centers detections and predictions
            pt_count += len(ch_ids) + len(aruco_ids)
            accum += dist2_corners.sum() + dist2_aruco.sum() if square_norm else np.sqrt(dist2_corners).sum() + np.sqrt(dist2_aruco).sum()

    return accum / pt_count


def get_to_world(params, pg_cam_matrix):
    """ Computes the transformation from camera space to world space"""

    # Sample to head sphere space
    sample_to_head_sphere = rot_scipy(params[9:12])
    sample_to_head_sphere[0:3, 3] = params[12:15]

    # Head sphere to head space
    head_transl = np.eye(4)
    head_transl[:3, 3] = params[15:18]
    head_sphere_to_head = head_transl @ pg_cam_matrix.T

    # Head to camera space
    head_to_cam = rot_scipy(params[18:21])
    head_to_cam[:3, 3] = params[21:24]
    
    # Sample to pixel space
    flip = np.array([[0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])
    sample_to_camera = flip @ head_to_cam @ head_sphere_to_head @ sample_to_head_sphere
    # sample_to_camera = flip @ head_sphere_to_head
    return inverse_homogenous_matrix(sample_to_camera)