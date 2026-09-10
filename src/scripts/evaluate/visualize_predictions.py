import os
import torch
import numpy as np
import cv2
import argparse
from utils.dataloader import LiDARDataset
from utils.colormap import create_custom_class_colormap
from models import get_model  # Deine Modell-Importfunktion


def main():
    parser = argparse.ArgumentParser(description="Visualisierung von Modellvorhersagen")
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
    parser.add_argument(
        "--model_path", type=str, required=True, help="Pfad zum trainierten Modell"
    )
    parser.add_argument(
        "--num_classes", type=int, default=17, help="Anzahl der Klassen im Modell"
    )
    args = parser.parse_args()

    # Modell laden
    model = get_model(
        "resnet18", num_classes=args.num_classes
    )  # Ändere den Modellnamen entsprechend
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # DataLoader initialisieren
    scenes_list = args.scenes.split("_")
    dataset = LiDARDataset(
        dataset_dir=args.dataset_dir,
        scenes=scenes_list,
        sensor=args.sensor,
        apply_transform=False,
    )

    # Verzeichnis zum Speichern der Bilder erstellen
    output_dir = "predicted_images"
    os.makedirs(output_dir, exist_ok=True)

    # Sicherstellen, dass der Endindex innerhalb der Dataset-Länge liegt
    if args.end_index >= len(dataset):
        args.end_index = len(dataset) - 1

    # Über den angegebenen Indexbereich iterieren
    for idx in range(args.start_index, args.end_index + 1):
        # Laden der Daten
        points_tensor, labels_tensor = dataset[idx]

        # Modellvorhersagen berechnen
        with torch.no_grad():
            predictions = model(points_tensor.unsqueeze(0))  # [1, num_classes, H, W]
            predictions = (
                torch.argmax(predictions, dim=1).squeeze(0).cpu().numpy()
            )  # [H, W]

        # Labels einfärben
        custom_colormap = create_custom_class_colormap()
        pred_colored = cv2.applyColorMap(predictions.astype(np.uint8), custom_colormap)

        # Originalbilder laden
        intensity_image = points_tensor[3].numpy()  # Intensitätskanal
        intensity_image = (intensity_image - intensity_image.min()) / (
            intensity_image.max() - intensity_image.min() + 1e-8
        )
        intensity_image = (intensity_image * 255).astype(np.uint8)
        intensity_colored = cv2.cvtColor(intensity_image, cv2.COLOR_GRAY2BGR)

        # Prädiktionen überlagern
        overlay = cv2.addWeighted(intensity_colored, 0.7, pred_colored, 0.3, 0)

        # Bilder speichern
        pred_path = os.path.join(output_dir, f"prediction_{idx}_{args.sensor}.png")
        overlay_path = os.path.join(
            output_dir, f"prediction_overlay_{idx}_{args.sensor}.png"
        )

        cv2.imwrite(pred_path, pred_colored)
        cv2.imwrite(overlay_path, overlay)

        print(f"Bilder gespeichert: {pred_path}, {overlay_path}")


if __name__ == "__main__":
    main()
