import torch
from utils.dataloader import LiDARDataset


def main():
    # Pfade und Szenen definieren
    dataset_dir = "/workspace/dataset"
    scenes = ["0005"]  # Passen Sie dies an Ihre verfügbaren Szenen an

    # Sensor auswählen ('left', 'right' oder 'both')
    sensor = "right"

    # DataLoader instanziieren
    dataset = LiDARDataset(
        dataset_dir=dataset_dir,
        scenes=scenes,
        sensor=sensor,
        apply_transform=False,
        apply_data_augmentation=False,
    )

    # Überprüfen der Länge des Datasets
    print(f"Anzahl der Samples im Dataset: {len(dataset)}")

    # Testweise alle Samples durchlaufen (optional)
    for sample_idx in range(len(dataset)):
        print(f"\n--- Sample Index: {sample_idx} ---")

        try:
            # Laden einer Datenprobe
            points_tensor, labels_tensor = dataset[sample_idx]

            # Form und Datentypen überprüfen
            print(f"Form des Punkte-Tensors: {points_tensor.shape}")  # [C, H, W]
            print(f"Form des Label-Tensors: {labels_tensor.shape}")  # [H, W]
            print(f"Datentyp des Punkte-Tensors: {points_tensor.dtype}")
            print(f"Datentyp des Label-Tensors: {labels_tensor.dtype}")

            # Einzigartige Labels überprüfen
            unique_labels = torch.unique(labels_tensor)
            print(f"Einzigartige Labels: {unique_labels.tolist()}")

            # Minimal- und Maximalwerte der Punkte überprüfen
            print(f"Minimalwert im Punkte-Tensor: {points_tensor.min().item()}")
            print(f"Maximalwert im Punkte-Tensor: {points_tensor.max().item()}")

            # Fehlerbedingungen prüfen
            if labels_tensor.max() > 17:  # Maximalwert der Klassen-ID im Mapping
                print("[WARNUNG] Labels außerhalb der erwarteten Klassen-ID!")
            if labels_tensor.min() < 0:
                print("[WARNUNG] Negative Labels gefunden!")

        except RuntimeError as e:
            print(f"Fehler bei Sample {sample_idx}: {e}")
            continue

    print("\n--- Test abgeschlossen ---")


if __name__ == "__main__":
    main()
