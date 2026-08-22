# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np


def estimate_vel(points):
    """
    Project radial velocity onto each point's line of sight.

    Args:
        points: Nx4 array with [x, y, z, radial_velocity]

    Returns:
        Nx6 array with [x, y, z, vx, vy, vz]
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    vr = points[:, 3]

    # v = vr * r_hat, i.e. vr*cos(phi)cos(theta), vr*cos(phi)sin(theta), vr*sin(phi)
    # written as x/R, y/R, z/R so |v| == |vr| exactly.
    R = np.sqrt(x**2 + y**2 + z**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        vx = vr * (x / R)
        vy = vr * (y / R)
        vz = vr * (z / R)

    # R ~ 0 (point at the sensor origin)
    vx[~np.isfinite(vx)] = 0
    vy[~np.isfinite(vy)] = 0
    vz[~np.isfinite(vz)] = 0

    return np.column_stack([points[:, :3], vx, vy, vz])
