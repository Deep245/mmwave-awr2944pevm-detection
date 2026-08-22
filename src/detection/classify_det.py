# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np

# Velocities derive from radial velocity, so tangential motion reads vr ~ 0.
STATIC_SPEED_THRESHOLD = 0.03  # m/s


def classify_det(detections):
    """Split Nx6 detections by speed. Returns (staticmeas, dynamicmeas)."""
    detections = np.asarray(detections, dtype=float)

    if detections.ndim != 2 or detections.shape[0] == 0:
        return np.zeros((0, 6)), np.zeros((0, 6))

    speed = np.linalg.norm(detections[:, 3:6], axis=1)
    is_static = speed < STATIC_SPEED_THRESHOLD

    return detections[is_static], detections[~is_static]
