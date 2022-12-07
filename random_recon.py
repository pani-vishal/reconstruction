import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from calibration.helpers.io import *
from calibration.helpers.camera_model import *
from calibration.helpers.linear_alg import *
from mitsuba import ScalarTransform4f as T

import drjit as dr

# Functions
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

# Params
n_captures = 6
seed = 42
def_sample_count = 128
def_res_x = 6576 // 2
def_res_y = 4384 // 2

path_tex_targ = 'scenes/textures/charuco.png'
path_tex_init = 'scenes/textures/init.png'
path_captures = '/home/vpani/rgl/recon/captures'
folder_dataset = '../datasets/'
folder_params = 'data'
dataset_name = 'unanno36_sorted256_theta_5_70_8_phi_0_360_16'
path_dataset = os.path.join(folder_dataset, dataset_name)
path_params = os.path.join(folder_params, 'npp_' + dataset_name + '.npy')

cap_paths, cap_angs = get_random_captures(path_captures, n_captures, seed=seed)

# Loading dataset and params
dataset = read_dataset(path_dataset, 64, v=True)
params_cam = load_params(path_params, v=True)

# Reconstruction
arr_to_worlds = to_world(dataset, cap_angs, params_cam)
mtx_plane = tfm_plane(params_cam)

def make_batch_cameras(yfov, resx, resy, arr_to_worlds):
  dict_cameras = {}

  for i, mtx in enumerate(arr_to_worlds):
    dict_cameras[f"camera_{i}"] = {
      'type': 'perspective',
      'fov_axis': 'x',
      'near_clip': 0.001,
      'far_clip': 1000.0,
      'fov': yfov * 1.5,
      'to_world': T(mtx),
    }
  
  dict_sensors = {
    'type': 'batch',
    # Sampler
    'sampler': {
      'type': 'independent',
      'sample_count': def_sample_count,
    },
    # Film
    'film': {
      'type': 'hdrfilm',
      # Change the film size to match the batch size
      'width': arr_to_worlds.shape[0] * resx,
      'height': resy,
      'rfilter': {
        'type': 'tent',
      }
    }
  }

  dict_sensors.update(dict_cameras)

  return dict_sensors

def make_board_scene(path_tex, resx, resy, mtx_plane, arr_to_worlds, params_cam):
  dict_sensors = make_batch_cameras(params_cam[6], resx, resy, arr_to_worlds)
  sensors = mi.load_dict(dict_sensors)

  scene =  mi.load_dict({
    'type': 'scene',
    # Integrator
    'integrator': {
      'type': 'prb',
      'max_depth': 8,
    },

    # Sensor
    'sensors': sensors, 

    # BSDFS
    'charuco': {
        'type': 'twosided',
        'material': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': path_tex,
                'filter_type': 'bilinear',
            }
        },
    },

    # Light
    'light': {
      'type': 'constant',
      'radiance': {
        'type': 'rgb',
        'value': 1.0,
      }
    },

    # Shapes
    'tex_plane': {
      'type': 'obj',
      'filename': 'scenes/meshes/tex_plane.obj',
      'to_world': T(mtx_plane) @ T.rotate(axis=[0,1,0], angle=-90),
      'bsdf': {
        'type': 'ref',
        'id': 'charuco',
      },
    }
  })

  return scene

scene_opt = make_board_scene(path_tex_init, def_res_x, def_res_y, mtx_plane, arr_to_worlds, params_cam)

def gen_ref_img(path_caps, resx, resy):
  ref_img = None
  print("Generating reference image...")
  for path in tqdm(path_caps):
    img = mi.TensorXf(mi.Bitmap(path)
          .convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
          .resample([resx, resy]))
    if ref_img is None:
      ref_img = img
    else:
      ref_img = np.hstack((ref_img, img))
  
  return mi.TensorXf(ref_img)

ref_img = gen_ref_img(cap_paths, def_res_x, def_res_y)

# Optimization
params = mi.traverse(scene_opt)
param_key = 'charuco.brdf_0.reflectance.data'
opt = mi.ad.Adam(lr=0.1)
opt[param_key] = params[param_key]
params.update(opt)

iteration_count = 100
errors = []
for it in range(iteration_count):
    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene_opt, params, seed=it, spp=1)
    
    # Evaluate the objective function from the current rendered image
    loss = dr.mean(dr.sqr(image - ref_img))

    # Backpropagate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[param_key] = dr.clamp(opt[param_key], 0.0, 1.0)

    # Update the scene state to the new optimized values
    params.update(opt)
    
    # Display loss
    print(f"Iteration {it:02d}: parameter error = {loss[0]:6f}", end='\r')
    errors.append(loss)
print('\nOptimization complete.')

print("Saving optimized texture...")
params_ref_bm = mi.util.convert_to_bitmap(params[param_key]).convert(mi.Bitmap.PixelFormat.RGB,
                                                              mi.Struct.Type.UInt8, False)
mi.util.write_bitmap(f'recon_tex_{n_captures}.png', params_ref_bm, write_async=True)

mi.util.write_bitmap(f'ref_img_{n_captures}.png', ref_img, write_async=True)

init_img = mi.render(scene_opt, spp=4)
mi.util.write_bitmap(f'final_img_{n_captures}.png', init_img, write_async=True)

