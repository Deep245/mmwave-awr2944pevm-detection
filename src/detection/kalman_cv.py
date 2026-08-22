# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np

"""
Constant-velocity Kalman filter over DBSCAN cluster centroids.

Measurement is the cluster's [x, y, z, vx, vy, vz], so H = I6.
Measurement noise is anisotropic about the line of sight: the radar is precise
along r_hat (range, Doppler) and blind across it, so tangential velocity gets a
large variance and is recovered from position differences instead.
"""

KF_STATE_FIELDS = ('id', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                   'speed', 'age', 'hits', 'misses', 'nis')

I3 = np.eye(3)


def cv_transition(dt):
    F = np.eye(6)
    F[0:3, 3:6] = dt * I3
    return F


def cv_process_noise(dt, sigma_a):
    """Piecewise white-noise-acceleration Q."""
    q = float(sigma_a) ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = (dt4 / 4.0) * I3
    Q[0:3, 3:6] = (dt3 / 2.0) * I3
    Q[3:6, 0:3] = (dt3 / 2.0) * I3
    Q[3:6, 3:6] = dt2 * I3
    return q * Q


def gravity_input(dt, gravity):
    g = np.asarray(gravity, dtype=float)
    return np.concatenate([0.5 * dt * dt * g, dt * g])


def measurement_noise(position, num_points, params):
    """
    Direction-dependent 6x6 R for one cluster:
        R_pos = sigma_range^2 * P_par + (R*sigma_theta)^2 * P_perp
        R_vel = sigma_vr^2    * P_par + sigma_vt^2        * P_perp
    scaled down by the effective point count.
    """
    pos = np.asarray(position, dtype=float)
    rng = float(np.linalg.norm(pos))

    if rng < 1e-6:
        r_hat = np.array([0.0, 1.0, 0.0])
        rng = 1e-6
    else:
        r_hat = pos / rng

    P_par = np.outer(r_hat, r_hat)
    P_perp = I3 - P_par

    sigma_cross = rng * float(params['sigma_theta'])
    R_pos = (float(params['sigma_range']) ** 2) * P_par + (sigma_cross ** 2) * P_perp
    R_vel = (float(params['sigma_vr']) ** 2) * P_par + (float(params['sigma_vt']) ** 2) * P_perp

    R = np.zeros((6, 6))
    R[0:3, 0:3] = R_pos
    R[3:6, 3:6] = R_vel

    # Centroid averaging helps, but DBSCAN membership churn is not averaged out
    n_eff = max(1.0, min(float(num_points), float(params['max_effective_points'])))
    return R / n_eff


class Track:
    def __init__(self, track_id, measurement, R):
        self.id = int(track_id)
        self.x = np.asarray(measurement, dtype=float).copy()

        # Trust the first position; distrust the first velocity
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = R[0:3, 0:3]
        self.P[3:6, 3:6] = R[3:6, 3:6]

        self.age = 0
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.nis = 0.0

    @property
    def position(self):
        return self.x[0:3]

    @property
    def velocity(self):
        return self.x[3:6]

    @property
    def speed(self):
        return float(np.linalg.norm(self.x[3:6]))

    def predict(self, dt, params):
        F = cv_transition(dt)
        self.x = F @ self.x
        if params.get('use_gravity', False):
            self.x = self.x + gravity_input(dt, params['gravity'])
        self.P = F @ self.P @ F.T + cv_process_noise(dt, params['sigma_a'])
        self.age += 1

    def update(self, measurement, R):
        """Full-state update (H = I6). Returns the NIS."""
        z = np.asarray(measurement, dtype=float)
        innovation = z - self.x
        S = self.P + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ S_inv
        self.x = self.x + K @ innovation

        # Joseph form keeps P symmetric positive-definite
        A = np.eye(6) - K
        self.P = A @ self.P @ A.T + K @ R @ K.T

        self.nis = float(innovation @ S_inv @ innovation)
        self.hits += 1
        self.misses = 0
        return self.nis

    def gate_distance(self, measurement, R):
        """Mahalanobis distance on the position block."""
        z = np.asarray(measurement, dtype=float)
        innovation = z[0:3] - self.x[0:3]
        S = self.P[0:3, 0:3] + R[0:3, 0:3]
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        return float(innovation @ S_inv @ innovation)

    def as_row(self):
        return np.array([
            self.id,
            self.x[0], self.x[1], self.x[2],
            self.x[3], self.x[4], self.x[5],
            self.speed,
            self.age, self.hits, self.misses,
            self.nis,
        ], dtype=float)


class CVTracker:
    def __init__(self, params):
        self.params = dict(params)
        self.tracks = []
        self._next_id = 1
        self.last_nis = []

    def reset(self):
        self.tracks = []
        self._next_id = 1
        self.last_nis = []

    def step(self, stats, dt):
        """
        Advance one frame.

        Args:
            stats: Mx8 from cluster_stats
            dt: seconds since the previous frame

        Returns:
            confirmed tracks
        """
        stats = np.asarray(stats, dtype=float)
        if stats.ndim != 2 or stats.shape[0] == 0:
            stats = np.zeros((0, 8))

        if dt <= 0.0 or not np.isfinite(dt):
            dt = float(self.params.get('default_dt', 0.05))

        for track in self.tracks:
            track.predict(dt, self.params)

        measurements = stats[:, 1:7] if stats.shape[0] else np.zeros((0, 6))
        counts = stats[:, 7] if stats.shape[0] else np.zeros((0,))
        noises = [measurement_noise(measurements[i, 0:3], counts[i], self.params)
                  for i in range(measurements.shape[0])]

        # Greedy nearest, Mahalanobis-gated
        assigned_track = np.full(measurements.shape[0], -1, dtype=int)
        track_taken = np.zeros(len(self.tracks), dtype=bool)

        if len(self.tracks) and measurements.shape[0]:
            cost = np.full((len(self.tracks), measurements.shape[0]), np.inf)
            for t, track in enumerate(self.tracks):
                for m in range(measurements.shape[0]):
                    cost[t, m] = track.gate_distance(measurements[m], noises[m])

            gate = float(self.params['gate_mahalanobis'])
            while True:
                t, m = np.unravel_index(np.argmin(cost), cost.shape)
                if not np.isfinite(cost[t, m]) or cost[t, m] > gate:
                    break
                assigned_track[m] = t
                track_taken[t] = True
                cost[t, :] = np.inf
                cost[:, m] = np.inf

        self.last_nis = []
        for m in range(measurements.shape[0]):
            t = assigned_track[m]
            if t >= 0:
                self.last_nis.append(self.tracks[t].update(measurements[m], noises[m]))

        for t, track in enumerate(self.tracks):
            if not track_taken[t]:
                track.misses += 1
            if track.hits >= int(self.params['min_hits']):
                track.confirmed = True

        self.tracks = [tr for tr in self.tracks
                       if tr.misses <= int(self.params['max_misses'])]

        for m in range(measurements.shape[0]):
            if assigned_track[m] < 0:
                self.tracks.append(Track(self._next_id, measurements[m], noises[m]))
                self._next_id += 1

        return [tr for tr in self.tracks if tr.confirmed]

    def state_rows(self, tracks=None):
        """Mx12 array of KF_STATE_FIELDS."""
        tracks = self.tracks if tracks is None else tracks
        if not tracks:
            return np.zeros((0, len(KF_STATE_FIELDS)))
        return np.vstack([tr.as_row() for tr in tracks])

    def mean_nis(self):
        """Mean NIS of the last frame's updates; should sit near 6."""
        return float(np.mean(self.last_nis)) if self.last_nis else float('nan')
