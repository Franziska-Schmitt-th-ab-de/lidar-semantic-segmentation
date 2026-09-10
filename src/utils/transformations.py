import numpy as np


def point_cloud_to_range_image(points, labels, H=64, W=2048):
    """
    Wandelt eine Punktwolke in ein Range Image um und ordnet die Labels entsprechend zu.
    """
    # Berechnen der azimutalen und vertikalen Winkel
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)

    # Normalisieren der Winkel auf Indexbereiche
    azimuth_res = (azimuth + np.pi) / (2 * np.pi)  # Bereich [0, 1]
    elevation_res = (elevation + np.pi / 2) / np.pi  # Bereich [0, 1]

    azimuth_idx = (azimuth_res * W).astype(np.int32)
    elevation_idx = (elevation_res * H).astype(np.int32)

    # Clipping der Indizes
    azimuth_idx = np.clip(azimuth_idx, 0, W - 1)
    elevation_idx = np.clip(elevation_idx, 0, H - 1)

    # Initialisieren des Range Images und Label Images
    range_image = np.zeros(
        (H, W, 5), dtype=np.float32
    )  # x, y, z, Intensität, Entfernung
    label_image = np.zeros((H, W), dtype=np.int32)

    # Füllen des Range Images und Label Images
    range_image[elevation_idx, azimuth_idx, :3] = points[:, :3]
    range_image[elevation_idx, azimuth_idx, 3] = intensity
    range_image[elevation_idx, azimuth_idx, 4] = r
    label_image[elevation_idx, azimuth_idx] = labels

    return range_image.transpose(2, 0, 1), label_image  # Ausgabe in Form von [C, H, W]
