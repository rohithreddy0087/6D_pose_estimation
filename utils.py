import numpy as np
import math

_EPS = np.finfo(float).eps * 4.0  

def quaternion_to_rotation_matrix(quaternion):
    quat = np.asarray(quaternion, dtype=np.float64)
    quat_norm = np.dot(quat, quat)
    if quat_norm < _EPS:
        return np.identity(4) 
    
    quat *= math.sqrt(2.0 / quat_norm)
    
    outer_prod = np.outer(quat, quat)
    
    rotation_matrix = np.array([
        [1.0 - outer_prod[2, 2] - outer_prod[3, 3], outer_prod[1, 2] - outer_prod[3, 0], outer_prod[1, 3] + outer_prod[2, 0], 0.0],
        [outer_prod[1, 2] + outer_prod[3, 0], 1.0 - outer_prod[1, 1] - outer_prod[3, 3], outer_prod[2, 3] - outer_prod[1, 0], 0.0],
        [outer_prod[1, 3] - outer_prod[2, 0], outer_prod[2, 3] + outer_prod[1, 0], 1.0 - outer_prod[1, 1] - outer_prod[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return rotation_matrix