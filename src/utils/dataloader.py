import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentation import DataAugmentation


class LiDARDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scenes,
        sensor="both",
        apply_transform=True,
        apply_data_augmentation=True,
        transform=None,
    ):
        self.dataset_dir = dataset_dir
        self.scenes = scenes
        self.sensor_mode = sensor
        self.apply_transform = apply_transform
        self.apply_data_augmentation = apply_data_augmentation
        self.transform = transform

        self.point_cloud_files = []
        self.label_files = []

        for scene in self.scenes:
            velodyne_dir = os.path.join(self.dataset_dir, scene, "velodyne")
            labels_dir = os.path.join(self.dataset_dir, scene, "labels")

            velodyne_files = sorted(
                [
                    os.path.join(velodyne_dir, f)
                    for f in os.listdir(velodyne_dir)
                    if f.endswith(".bin")
                ]
            )
            label_files = sorted(
                [
                    os.path.join(labels_dir, f)
                    for f in os.listdir(labels_dir)
                    if f.endswith(".label")
                ]
            )

            assert len(velodyne_files) == len(
                label_files
            ), f"Anzahl der Punktwolken und Labels stimmt nicht überein in Szene {scene}"

            self.point_cloud_files.extend(velodyne_files)
            self.label_files.extend(label_files)

        # Transformationsmatrizen definieren
        self.T_left_inv = np.array(
            [
                [0.5408944, -0.70507801, 0.45858286, 0.0814],
                [0.54089373, 0.70912162, 0.45230576, 0.4954],
                [-0.64410187, 0.00339494, 0.76493219, 2.1656],
                [0, 0, 0, 1],
            ]
        )

        self.T_right_inv = np.array(
            [
                [5.41525802e-01, 7.07106499e-01, 4.54697927e-01, 8.08000000e-02],
                [-5.40733226e-01, 7.07106527e-01, -4.55640141e-01, -4.98900000e-01],
                [-6.43705977e-01, 8.70615596e-04, 7.65272407e-01, 2.16520000e00],
                [0, 0, 0, 1],
            ]
        )

        # Sensor-Positionen (aus SemanticKANIS)
        # (für das "Nahbereich auf Klasse 0 setzen")
        self.left_sensor_pos = np.array([0.0814, 0.4954, 2.1656])
        self.right_sensor_pos = np.array([0.0808, -0.4989, 2.1652])
        self.near_distance = 0.1  # Schwellwert, wann Punkte "zu nah" sind

        # Initialisiere Datenaugmentation
        if self.apply_data_augmentation:
            self.data_augmentation = DataAugmentation(
                horizontal_flip=True,
                vertical_flip=False,
                random_rotation=True,
                max_rotation_angle=np.pi / 4,
            )

        # Label-Mapping (aus SemanticKANIS)
        # Punkte, die nicht in diesem Dict sind -> vgl. unten 'np.vectorize(...)'
        self.label_mapping_less_classes = {
            0: 0,  # "unlabeled"
            1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 1,  # "car"
            11: 2,  # "truck"
            12: 3,  # "forklift"
            20: 4,  # "person"
            21: 5,  # "bicyclist"
            30: 6,  # "object"
            31: 6,  # "pallet" (7)
            40: 8,  # "terrain"
            41: 9,  # "drive able ground"
            42: 9,  # "other-ground" (10)
            43: 9,  # "lane marking" (11)
            50: 12,  # vegetation
            51: 6,  # trunk (13)
            60: 14,  # building
            61: 6,  # static object (15)
            62: 6,  # fence (16)
            63: 6,  # rack (17)
            254: 4,  # person
            258: 2,  # truck
            252: 1,  # car
            253: 5,  # bicyclist
        }

        self.label_mapping_all_classes = {
            0: 0,  # unlabled
            1: 0,  # outlier
            10: 1,  # car
            11: 2,  # truck
            12: 3,  # forklift
            20: 4,  # person
            21: 5,  # bicyclist
            30: 6,  # object
            31: 7,  # pallet
            40: 8,  # terrain
            41: 9,  # driveable ground
            42: 9,  # other ground
            43: 11,  # lane marking
            50: 12,  # vegetation
            51: 13,  # trunk
            60: 14,  # building
            61: 15,  # static object
            62: 16,  # fence
            63: 17,  # rack
            254: 4,  # person
            258: 2,  # truck
            252: 1,  # car
            253: 5,  # bicyclist
        }

        # Dictionary um Fehler zu zählen: {scene: {error_type: set(files)}}
        self.errors = {}

    def __len__(self):
        return len(self.point_cloud_files)

    def __getitem__(self, idx):
        H, W = 128, 2 * 2048
        expected_points_size = H * W * 4

        for attempt in range(len(self)):
            point_cloud_path = self.point_cloud_files[idx]
            label_path = self.label_files[idx]

            # Szene extrahieren
            scene = point_cloud_path.split(os.sep)[-3]  # .../dataset/0001/velodyne/...

            # --- Hilfsfunktion zum Fehlerloggen ---
            def log_error(error_type, detail=""):
                if scene not in self.errors:
                    self.errors[scene] = {}
                if error_type not in self.errors[scene]:
                    self.errors[scene][error_type] = set()
                self.errors[scene][error_type].add(
                    (point_cloud_path, label_path, detail)
                )

            # --- Laden der Binärdaten ---
            points = np.fromfile(point_cloud_path, dtype=np.float32)
            labels = np.fromfile(label_path, dtype=np.int32)

            # --- Plausibilitätschecks (Size) ---
            if points.size != expected_points_size:
                log_error("unexpected_point_size", f"points.size={points.size}")
                idx = (idx + 1) % len(self)
                continue

            if labels.size != (H * W):
                log_error("unexpected_label_size", f"labels.size={labels.size}")
                idx = (idx + 1) % len(self)
                continue

            # --- Aufteilen in (x, y, z, i) und sem_label ---
            points = points.reshape(H, W, 4)
            # Label-Bits: sem = label & 0xFFFF, inst = label >> 16
            sem_labels = (labels & 0xFFFF).reshape(H, W)

            # --- Links / Rechts aufsplitten ---
            points_left = points[:, :2048, :]  # [H, 2048, 4]
            points_right = points[:, 2048:, :]  # [H, 2048, 4]
            labels_left = sem_labels[:, :2048]
            labels_right = sem_labels[:, 2048:]

            # --- Mapping der sem. Klassen ---
            try:
                #! To use less classes switch to label_mapping_less_classes
                labels_left_remapped = np.vectorize(self.label_mapping_all_classes.get)(
                    labels_left
                )
                labels_right_remapped = np.vectorize(
                    self.label_mapping_all_classes.get
                )(labels_right)
                # Achtung: Falls ein Label nicht im Dict ist, wird None zurückgegeben.
                # In so einem Fall könntest du default=0 machen.
                # Oder du prüfst vorher, ob None vorkommt.
                # Hier mal ganz pragmatisch:
                labels_left_remapped[np.where(labels_left_remapped == None)] = 0
                labels_right_remapped[np.where(labels_right_remapped == None)] = 0

                labels_left_remapped = labels_left_remapped.astype(np.int32)
                labels_right_remapped = labels_right_remapped.astype(np.int32)

            except Exception as e:
                log_error("label_remapping_error", str(e))
                idx = (idx + 1) % len(self)
                continue

            # --- Sensor-Auswahl / Transformation ---
            if self.sensor_mode == "both":
                # Random Entscheidung, ob linke oder rechte Seite verwendet wird
                use_left = np.random.rand() < 0.5
                if use_left:
                    points_sensor = points_left
                    labels_sensor = labels_left_remapped
                    transformation_matrix = self.T_left_inv
                    sensor_pos = self.left_sensor_pos
                else:
                    points_sensor = points_right
                    labels_sensor = labels_right_remapped
                    transformation_matrix = self.T_right_inv
                    sensor_pos = self.right_sensor_pos

            elif self.sensor_mode == "left":
                points_sensor = points_left
                labels_sensor = labels_left_remapped
                transformation_matrix = self.T_left_inv
                sensor_pos = self.left_sensor_pos

            elif self.sensor_mode == "right":
                points_sensor = points_right
                labels_sensor = labels_right_remapped
                transformation_matrix = self.T_right_inv
                sensor_pos = self.right_sensor_pos

            else:
                raise ValueError("sensor_mode muss 'left', 'right' oder 'both' sein.")

            # --- Optional: Transformation (homogene Koordinaten) ---
            if self.apply_transform:
                points_sensor_flat = points_sensor.reshape(-1, 4)
                ones = np.ones((points_sensor_flat.shape[0], 1))
                points_hom = np.hstack((points_sensor_flat[:, :3], ones))
                points_transformed = (transformation_matrix @ points_hom.T).T
                points_transformed = np.hstack(
                    (points_transformed[:, :3], points_sensor_flat[:, 3:4])
                )
                points_sensor = points_transformed.reshape(H, W // 2, 4)

            # --- Setze Punkte nah am Sensor auf Label 0 ---
            #     Beispiel: L2-Distanz <= 0.1
            xyz_sensor = points_sensor[..., :3]  # Shape [H, W//2, 3]
            dist = np.linalg.norm(xyz_sensor - sensor_pos, axis=-1)  # [H, W//2]
            labels_sensor[dist <= self.near_distance] = 0

            # --- (Optionale) externe Transformation / Normalisierung ---
            if self.transform:
                points_sensor = self.transform(points_sensor)

            # --- Datenaugmentation (z. B. Flip / Rotation) ---
            if self.apply_data_augmentation:
                points_sensor, labels_sensor = self.data_augmentation.apply(
                    points_sensor, labels_sensor
                )

            # --- Als Torch-Format zurückgeben ---
            points_tensor = torch.from_numpy(points_sensor).float()  # [H, W//2, 4]
            labels_tensor = torch.from_numpy(labels_sensor).long()  # [H, W//2]

            # Permute => [4, H, W//2]
            points_tensor = points_tensor.permute(2, 0, 1)

            return points_tensor, labels_tensor

        # Falls wir nach x Versuchen nichts ausgeben konnten
        raise RuntimeError(
            "Keine gültigen Samples im Datensatz gefunden oder alle Samples sind fehlerhaft."
        )

    def save_errors(self):
        """
        Schreibt am Ende des Trainings einmalig eine Datei mit den gesammelten Fehlerinformationen.
        Außerdem kann hier die Prozentanzahl fehlerhafter Dateien pro Szene berechnet werden.
        """
        if not self.errors:
            print("Keine Fehler aufgezeichnet, alle Dateien waren OK.")
            return

        error_file = os.path.join(self.dataset_dir, "error_summary.csv")
        with open(error_file, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(
                ["Scene", "ErrorType", "NumFiles", "TotalFilesInScene", "Percentage"]
            )

            for scene in self.scenes:
                scene_velodyne_dir = os.path.join(self.dataset_dir, scene, "velodyne")
                total_files_in_scene = sum(
                    1 for ff in os.listdir(scene_velodyne_dir) if ff.endswith(".bin")
                )

                if scene in self.errors:
                    for error_type, files in self.errors[scene].items():
                        num_files = len(files)
                        percentage = (
                            (num_files / total_files_in_scene) * 100
                            if total_files_in_scene > 0
                            else 0.0
                        )
                        writer.writerow(
                            [
                                scene,
                                error_type,
                                num_files,
                                total_files_in_scene,
                                f"{percentage:.2f}%",
                            ]
                        )
                else:
                    writer.writerow(
                        [scene, "NoError", 0, total_files_in_scene, "0.00%"]
                    )

        print(f"Fehlerübersicht gespeichert unter: {error_file}")
