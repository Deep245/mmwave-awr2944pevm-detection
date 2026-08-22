# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np
import pandas as pd


def process_det(det, time_in_seconds, xLowerBound, xHigerBound,
                yLowerBound, yHigerBound, zLowerBound, zHigerBound):
    """
    Filter Nx6 [x, y, z, vx, vy, vz] detections by bounds.

    Returns a DataFrame with columns t, X, Y, Z, Vx, Vy, Vz.
    """
    detections_list = []

    for i in range(det.shape[0]):
        X = det[i, 0]
        Y = det[i, 1]
        Z = det[i, 2]

        # MATLAB processDet checks only vx; vr=0 makes all three exactly 0
        if abs(det[i, 3]) != 0:
            Vx = det[i, 3]
            Vy = det[i, 4]
            Vz = det[i, 5]
        else:
            Vx = 0.0
            Vy = 0.0
            Vz = 0.0

        if (X > xLowerBound and X < xHigerBound) and \
           (Y > yLowerBound and Y < yHigerBound) and \
           (Z > zLowerBound and Z < zHigerBound):
            detections_list.append({
                't': time_in_seconds,
                'X': X, 'Y': Y, 'Z': Z,
                'Vx': Vx, 'Vy': Vy, 'Vz': Vz
            })

    return pd.DataFrame(detections_list)
