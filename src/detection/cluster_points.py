# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import numpy as np
from sklearn.cluster import DBSCAN

CLUSTER_STATS_FIELDS = ('id', 'cx', 'cy', 'cz', 'vx', 'vy', 'vz', 'num_points')


def cluster_points(points, clusterer_params):
    """Cluster one frame on xyz. Returns 1-indexed labels; 0 means noise."""
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return np.zeros((0,), dtype=int)

    clusterer = DBSCAN(
        eps=clusterer_params['Epsilon'],
        min_samples=clusterer_params['MinNumPoints']
    )
    return clusterer.fit_predict(points[:, :3]) + 1


def cluster_and_drop_noise(points, clusterer_params):
    """Cluster and discard noise. Returns (points, labels, num_clusters)."""
    points = np.asarray(points)
    labels = cluster_points(points, clusterer_params)

    if labels.size == 0:
        return np.zeros((0, points.shape[1] if points.ndim == 2 else 6)), \
               np.zeros((0,), dtype=int), 0

    keep = labels > 0
    kept_points = points[keep]
    kept_labels = labels[keep]

    return kept_points, kept_labels, int(np.unique(kept_labels).size)


def cluster_stats(points, labels):
    """Aggregate to one row per cluster: CLUSTER_STATS_FIELDS."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)

    if points.ndim != 2 or points.shape[0] == 0 or labels.size != points.shape[0]:
        return np.zeros((0, len(CLUSTER_STATS_FIELDS)))

    unique_labels = np.unique(labels)
    stats = np.zeros((unique_labels.size, len(CLUSTER_STATS_FIELDS)))

    for i, label in enumerate(unique_labels):
        member = points[labels == label]
        velocity = np.mean(member[:, 3:6], axis=0) if member.shape[1] >= 6 else np.zeros(3)

        stats[i, 0] = float(label)
        stats[i, 1:4] = np.mean(member[:, :3], axis=0)
        stats[i, 4:7] = velocity
        stats[i, 7] = member.shape[0]

    return stats
