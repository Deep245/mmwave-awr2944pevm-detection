# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np
from scipy.spatial.transform import Rotation


def adjust_orientation(elev_tilt, az_tilt, sensor_height, x, y, z):
    """Rotate xyz for elevation/azimuth tilt (degrees). Returns x, y, z."""
    if elev_tilt != 0 or az_tilt != 0:
        radar_coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        elev_tilt_rad = np.deg2rad(elev_tilt)
        az_tilt_rad = np.deg2rad(az_tilt)

        # ZYX order matches MATLAB eul2rotm
        R = Rotation.from_euler('ZYX', [az_tilt_rad, elev_tilt_rad, 0]).as_matrix()

        rotated_coords = (R @ radar_coords.T).T
        x = rotated_coords[:, 0]
        y = rotated_coords[:, 1]
        z = rotated_coords[:, 2]

    if sensor_height != 0:
        z = z + az_tilt

    return x, y, z
