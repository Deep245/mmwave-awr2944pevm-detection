# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np


def create_det(detections):
    """DataFrame (X, Y, Z, Vx, Vy, Vz) -> Nx6 [x, y, z, vx, vy, vz]."""
    if len(detections) == 0:
        return np.zeros((0, 6))

    return detections[['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']].to_numpy(dtype=float)
