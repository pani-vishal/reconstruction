""" Helper functions for the input/output tasks. """
import pickle
import pprint
import os
import numpy as np
import sys
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4, width=80, compact=False)


def bprint(string=None, ppr=False, div='='):
    """ Prints with a border."""
    if string is not None:
        if ppr:
            pp.pprint(string)
        else:
            print(string)
    print(div * 20)
    print()


def read_dataset(dataset_path: str, num_parts: int, v: bool=False):
    """ 
        Reads the dataset from the pickle file. 
        Dataset structure:
        [
            [[angles (theta_i, phi_i, theta_o, phi_o), wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners], image/None],
            .
            .
            .
        ]
    """
    dataset = []
    for part in tqdm(range(num_parts), desc=f"Reading {dataset_path}", disable=not(v)):
        f = open(dataset_path + "_part" + str(part) + ".pickle", "rb")
        dataset += pickle.load(f)
        f.close()
    if v:
        bprint()
    return dataset


def get_dataset_angles(dataset):
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
    return thetas_uniq, phis_uniq, vargeos_uniq


def load_params(path_params, v: bool=False):
    """ Loads the parameters from the pickle file."""
    if os.path.exists(path_params):
        with open(path_params, 'rb') as f:
            params = np.load(f)
        if v:
            bprint(f"Loaded parameters from {path_params}", div='*')
    else:
        print(f"Error: {path_params} does not exist. Please run calibration/optimize.py first.")
        sys.exit(1)

    return params
