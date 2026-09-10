import os
import numpy as np
import cv2
import argparse
from utils.dataloader import LiDARDataset
from utils.colormap import create_custom_class_colormap


def main():
    parser = argparse.ArgumentParser(description="Visualisierung der LiDAR-Daten")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/workspace/dataset",
        help="Pfad zum Datensatzverzeichnis",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="0002",
        help="Liste der Szenen, getrennt durch Unterstriche",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Sensor auswählen (left oder right)",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Startindex der anzuzeigenden Proben"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=500,
        help="Endindex der anzuzeigenden Proben (inklusive)",
    )
    args = parser.parse_args()

    # Trennen der eingegebenen Szenen (auch bei nur 1 wichtig)
    scenes_list = args.scenes.split("_")

    # DataLoader initialisieren
    dataset = LiDARDataset(
        dataset_dir=args.dataset_dir,
        scenes=scenes_list,
        sensor=args.sensor,
        apply_transform=False,
    )

    # Verzeichnis zum Speichern der Bilder erstellen
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Sicherstellen, dass der Endindex innerhalb der Dataset-Länge liegt
    if args.end_index >= len(dataset):
        args.end_index = len(dataset) - 1

    # Überprüfen, ob der Startindex kleiner oder gleich dem Endindex ist
    if args.start_index > args.end_index:
        print("Fehler: Der Startindex muss kleiner oder gleich dem Endindex sein.")
        return

    # Über den angegebenen Indexbereich iterieren
    for idx in range(args.start_index, args.end_index + 1):
        # Laden der Daten
        points_tensor, labels_tensor = dataset[idx]

        # Konvertieren zu NumPy
        points_np = points_tensor.numpy()  # [4, H, W]
        labels_np = labels_tensor.numpy()  # [H, W]

        H, W = labels_np.shape

        # Visualisierung der Intensität
        intensity_image = points_np[3, :, :]  # Intensitätskanal
        intensity_image = (intensity_image - intensity_image.min()) / (
            intensity_image.max() - intensity_image.min() + 1e-8
        )  # Normalisierung
        intensity_image = (intensity_image * 255).astype(np.uint8)

        # Labels einfärben
        custom_colormap = create_custom_class_colormap()
        labels_colored = cv2.applyColorMap(labels_np.astype(np.uint8), custom_colormap)

        # Intensitätsbild in BGR konvertieren
        intensity_colored = cv2.cvtColor(intensity_image, cv2.COLOR_GRAY2BGR)

        # Labels überlagern
        overlay = cv2.addWeighted(intensity_colored, 0.7, labels_colored, 0.3, 0)

        # Bilder speichern
        intensity_path = os.path.join(
            output_dir, f"intensity_image_{idx}_{args.sensor}.png"
        )
        labels_path = os.path.join(output_dir, f"labels_image_{idx}_{args.sensor}.png")
        overlay_path = os.path.join(
            output_dir, f"labels_overlay_{idx}_{args.sensor}.png"
        )

        cv2.imwrite(intensity_path, intensity_image)
        cv2.imwrite(labels_path, labels_colored)
        cv2.imwrite(overlay_path, overlay)

        print(
            f"Bilder wurden gespeichert: {intensity_path}, {labels_path}, {overlay_path}"
        )


if __name__ == "__main__":
    main()
