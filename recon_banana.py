import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from calibration.helpers.io import *
from calibration.helpers.camera_model import *
from calibration.helpers.linear_alg import *
from mitsuba import ScalarTransform4f as T
from cholespy import CholeskySolverF, MatrixType

import drjit as dr
import scipy.sparse as sp


# Reconstruction Functions
def to_world(dataset, arr_angles, params):
  """ Get to world matrix for a given set of angles """
  # Filtering can be optimized
  arr_mtx = []

  for angles_inp in arr_angles:
    for i, entry in enumerate(dataset):
      angles, wo_measured, vargeo, pg_cam_matrix, ch_ids, ch_corners = entry[0] # No image required
      theta_i, phi_i, theta_o, phi_o = angles

      if theta_o == angles_inp[0] and \
         phi_o == angles_inp[1] and \
         vargeo == angles_inp[2]:

        mtx = get_to_world(params, pg_cam_matrix)
        mtx[:3, 3] *= 0.001
        arr_mtx.append(mtx)

  return np.array(arr_mtx)

def tfm_plane(params):
  ij_to_sample = rot_scipy(params[:3])
  ij_to_sample[0:3, 3] = params[3:6] * 0.001
  return T(ij_to_sample) @ T.translate([.31/2, .18/2, 0])


def name_to_angles(name):
  """ Convert file name to angles """
  name_split = name.split('_')
  theta_o = int(name_split[2])
  phi_o = int(name_split[3])
  vargeo = int(name_split[4].split('.')[0])
  return [theta_o, phi_o, vargeo]

def get_random_captures(path, n, seed=42):
  """ Get random captures from a folder """
  np.random.seed(seed)
  files = os.listdir(path)
  files = [f for f in files if f.endswith('.png')]
  files = np.random.choice(files, n)
  angles = [name_to_angles(f) for f in files]
  files_path = [os.path.join(path, f) for f in files]
  return files_path, np.array(angles)
#################################

# Largesteps functions
def compute_laplacian(n_verts, faces, lambda_):

  # Neighbor indices
  ii = faces[:, [1, 2, 0]].flatten()
  jj = faces[:, [2, 0, 1]].flatten()
  adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
  adj_values = np.ones(adj.shape[1], dtype=np.float64) * lambda_

  # Diagonal indices, duplicated as many times as the connectivity of each index
  diag_idx = np.stack((adj[0], adj[0]), axis=0)

  diag = np.stack((np.arange(n_verts), np.arange(n_verts)), axis=0)

  # Build the sparse matrix
  idx = np.concatenate((adj, diag_idx, diag), axis=1)
  values = np.concatenate((-adj_values, adj_values, np.ones(n_verts)))

  return sp.csc_matrix((values, idx))

def to_differential(verts, faces, lambda_):
  L_csc = compute_laplacian(len(verts)//3, faces.numpy().reshape((-1,3)), lambda_)
  return mi.TensorXf((L_csc @ verts.numpy().reshape((-1,3))))