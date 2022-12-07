""" Helper functions containing all the linear algebraic functions. """
import numpy as np
from scipy.spatial.transform import Rotation as R

def rot_scipy(angles):
    ''' 
        Convert angles to a rotation matrix using scipy.

        Input:
            angles: array of shape (3,) containing the angles in degrees to be converted.
        
        Output:
            R: array of shape (4, 4) containing the homogenous rotation matrix.
    '''
    rot = np.eye(4)
    rot[:3, :3] = R.from_rotvec(angles, degrees=True).as_matrix()
    return rot


def perspective_matrix(fov, near, far, aspect):
    ''' Compute an OpenGL-style perspective matrix '''
    c = 1 / np.tan(0.5 * fov)
    recip = 1 / (near - far)
    return np.array([
        [c / aspect, 0, 0, 0],
        [0, c, 0, 0],
        [0, 0, (near + far) * recip, 2 * near * far * recip],
        [0, 0, -1, 0]]
    )


def deg2rad(value):
    ''' Helper function to convert degrees to radians '''
    return value * np.pi / 180


def mmul(A, v):
    ''' Perform one or more (vectorized) matrix-vector multiplications '''
    v = np.asarray(v)
    return (A @ v.T).T


def mmul_p(A, v):
    ''' Perform one or more (vectorized) matrix-vector multiplications
        (homogeneous coordinate version) '''
    v = mmul(A, v)
    return v[..., :-1] / v[..., -1, None]


def reproj_err(preds, targs):
    ''' Compute the reprojection error '''
    d = preds - targs
    dist = np.sqrt(np.sum(d*d, axis=-1))
    return dist.mean()


def inverse_homogenous_matrix(A):
    """ 
        Compute the inverse transformation of a homogenous matrix. 
        ref: https://mathematica.stackexchange.com/questions/106257/how-do-i-get-the-inverse-of-a-homogeneous-transformation-matrix
    """
    inv_rot = A[:3, :3].T
    inv_transl = -inv_rot @ A[:3, 3]
    inv_tfm = np.eye(4)
    inv_tfm[:3, :3] = inv_rot
    inv_tfm[:3, 3] = inv_transl
    return inv_tfm