# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np

sensorHeight = 0
az_tilt = 0
elev_tilt = 0

# Bounds filtering, in meters
xLowerBound = -1.5
xHigerBound = 1.5
yLowerBound = 0.5
yHigerBound = 15
zLowerBound = -1.5
zHigerBound = 1.5

# MinNumPoints maps to DBSCAN min_samples. At 1 nothing is ever labelled
# noise, so the downstream noise filter is inert; raise to 2+ to enable it.
clusterer_params = {
    'MinNumPoints': 1,
    'Epsilon': 1.0,
    'EnableDisambiguation': False
}

# CV Kalman filter over dynamic cluster centroids.
kf_params = {
    'sigma_range': 0.05,   # m, along the line of sight
    'sigma_theta': 0.052,  # rad; cross-range sigma = range * this
    'sigma_vr': 0.08,      # m/s, Doppler accuracy
    'sigma_vt': 3.0,       # m/s, tangential velocity is unmeasured - keep large
    'max_effective_points': 4.0,

    'sigma_a': 2.0,        # m/s^2 RMS unmodelled acceleration
    'use_gravity': False,  # True for ballistic targets
    'gravity': (0.0, 0.0, -9.81),

    'gate_mahalanobis': 16.0,  # chi2(3 dof): 7.81=95%, 11.34=99%
    'min_hits': 2,
    'max_misses': 3,
    'default_dt': 0.05,
}
