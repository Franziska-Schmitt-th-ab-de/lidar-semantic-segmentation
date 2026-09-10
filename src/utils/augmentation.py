import numpy as np


class DataAugmentation:
    def __init__(
        self,
        horizontal_flip=True,
        vertical_flip=False,
        random_rotation=True,
        max_rotation_angle=np.pi,
    ):
        """
        Initialisiert die Datenaugmentation mit Optionen für Spiegeln und Rotation.

        Args:
            horizontal_flip (bool): Ob horizontales Spiegeln angewendet werden soll.
            vertical_flip (bool): Ob vertikales Spiegeln angewendet werden soll.
            random_rotation (bool): Ob zufällige Rotation angewendet werden soll.
            max_rotation_angle (float): Maximale Rotationswinkel in Radiant (Standard: 180 Grad / Pi).
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_rotation = random_rotation
        self.max_rotation_angle = max_rotation_angle

    def apply(self, points, labels):
        """
        Führt Datenaugmentation auf Punktwolken und Labels aus.

        Args:
            points (np.ndarray): Array der Punktwolke in der Form [H, W, C].
            labels (np.ndarray): Array der Labels in der Form [H, W].

        Returns:
            np.ndarray: Augmentierte Punktwolke.
            np.ndarray: Augmentierte Labels.
        """
        # Horizontales Spiegeln
        if self.horizontal_flip and np.random.rand() > 0.5:
            points = points[
                :, ::-1, :
            ].copy()  # Spiegelt die W-Koordinate und entfernt negative Strides
            labels = labels[:, ::-1].copy()  # Spiegelt die W-Koordinate der Labels

        # Vertikales Spiegeln
        if self.vertical_flip and np.random.rand() > 0.5:
            points = points[
                ::-1, :, :
            ].copy()  # Spiegelt die H-Koordinate und entfernt negative Strides
            labels = labels[::-1, :].copy()  # Spiegelt die H-Koordinate der Labels

        # Zufällige Rotation
        if self.random_rotation:
            angle = np.random.uniform(-self.max_rotation_angle, self.max_rotation_angle)

            # Debug-Logging für Rotation
            # print(f"Applying rotation with angle: {angle:.2f} radians.")

            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            points[:, :, :3] = np.dot(
                points[:, :, :3], rotation_matrix.T
            )  # Rotiert die XYZ-Koordinaten

        return points, labels
