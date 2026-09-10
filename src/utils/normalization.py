import numpy as np


class Normalizer:
    def __init__(self):
        self.min_coords = None
        self.max_coords = None
        self.min_intensity = None
        self.max_intensity = None

    def fit(self, point_cloud_files):
        """
        Berechnet die Min- und Max-Werte für Punktkoordinaten und Intensität über alle Punktwolken.
        """
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])
        min_intensity, max_intensity = np.inf, -np.inf

        for point_file in point_cloud_files:
            points = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)
            min_coords = np.minimum(min_coords, points[:, :3].min(axis=0))
            max_coords = np.maximum(max_coords, points[:, :3].max(axis=0))
            min_intensity = min(min_intensity, points[:, 3].min())
            max_intensity = max(max_intensity, points[:, 3].max())

        self.min_coords = min_coords
        self.max_coords = max_coords
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def normalize(self, points):
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                f"Unerwartete Form der Punkte für die Normalisierung: {points.shape}. "
                f"Erwartet: (N, 4)"
            )

        points[:, :3] -= self.min_coords
        points[:, :3] /= self.max_coords - self.min_coords
        points[:, 3] = (points[:, 3] - self.min_intensity) / (
            self.max_intensity - self.min_intensity
        )
        return points
