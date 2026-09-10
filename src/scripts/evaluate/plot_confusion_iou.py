import os
import matplotlib.pyplot as plt
import numpy as np
import json
from utils.compare_eval_helpers import get_latest_run

# Deine bestehenden Farbcodes in (R,G,B), jeweils 0..255
class_idx_to_color = {
    0: [0, 0, 0],
    1: [100, 150, 245],
    2: [80, 30, 180],
    3: [102, 178, 255],
    4: [255, 30, 30],
    5: [255, 153, 255],
    6: [255, 178, 100],
    7: [153, 76, 0],
    8: [102, 102, 0],
    9: [0, 255, 255],
    10: [175, 0, 75],
    11: [245, 255, 0],
    12: [0, 175, 0],
    13: [102, 51, 0],
    14: [255, 200, 0],
    15: [252, 102, 38],
    16: [255, 0, 0],
    17: [255, 153, 153],
}

# Namen für die Klassen (wie schon zuvor)
CLASS_NAMES = {
    0: "unlabeled",
    1: "car",
    2: "truck",
    3: "forklift",
    4: "person",
    5: "bicyclist",
    6: "object",
    7: "pallet",
    8: "terrain",
    9: "driveable_ground",
    10: "unused",  # nicht belegt, Platzhalter
    11: "lane_marking",
    12: "vegetation",
    13: "trunk",
    14: "building",
    15: "static_object",
    16: "fence",
    17: "rack",
}


def plot_confusion_and_iou(models_dir):
    """
    Visualisiert die Konfusionsmatrix und IoU pro Klasse.
    - Konfusionsmatrix: als Standard-Heatmap
    - IoU-Linien: Klassenfarbe + Sortierung nach höchster IoU (z.B. letzte Epoche)
    """

    cross_val_outdir = os.path.join(models_dir, "evaluation_output")
    os.makedirs(cross_val_outdir, exist_ok=True)

    from os import listdir
    from os.path import isdir, join

    model_names = [d for d in listdir(models_dir) if isdir(join(models_dir, d))]

    for model_name in model_names:
        print(f"Analyse von Konfusionsmatrix und IoU für Modell: {model_name}")

        model_dir = os.path.join(models_dir, model_name)
        latest_run = get_latest_run(model_dir)
        if not latest_run:
            print(f"Kein gültiger Run gefunden für Modell: {model_name}")
            continue

        metrics_path = os.path.join(model_dir, latest_run, "detailed_metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Keine detailed_metrics.json gefunden für {model_name}")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        confusion_matrices = metrics.get("confusion_matrices", {})
        iou_per_class = metrics.get("iou_per_class", {})

        if not confusion_matrices or not iou_per_class:
            print(f"Keine Konfusionsmatrix- oder IoU-Daten für {model_name}")
            continue

        # -----------------------------
        # 1) Konfusionsmatrix (Heatmap)
        # -----------------------------
        final_epoch_cm = max(map(int, confusion_matrices.keys()))
        final_cm = np.array(confusion_matrices[str(final_epoch_cm)])

        plt.figure(figsize=(12, 10))
        plt.imshow(final_cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(
            f"Final Confusion Matrix (Model: {model_name}, Epoch: {final_epoch_cm})"
        )
        plt.colorbar()

        num_classes_cm = final_cm.shape[0]
        x_labels = [CLASS_NAMES.get(i, f"Class {i}") for i in range(num_classes_cm)]

        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        plt.xticks(
            ticks=np.arange(num_classes_cm), labels=x_labels, rotation=45, ha="right"
        )
        plt.yticks(ticks=np.arange(num_classes_cm), labels=x_labels)

        plt.tight_layout()

        heatmap_path = os.path.join(
            cross_val_outdir, f"{model_name}_final_confusion_matrix.png"
        )
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Konfusionsmatrix gespeichert unter: {heatmap_path}")

        # -----------------------------
        # 2) IoU-Kurven pro Klasse
        # -----------------------------
        epochs = sorted(map(int, iou_per_class.keys()))
        final_epoch_iou = max(epochs)  # Letzte Epoche für Sortierung
        num_classes_iou = len(iou_per_class[str(epochs[0])])

        # a) Ermitteln der IoU in der finalen Epoche -> Sortierung nach absteigendem IoU
        iou_in_final_epoch = iou_per_class[str(final_epoch_iou)]  # Liste pro Klasse
        # -> [(Klasse, IoU-Wert), ...], danach sortieren
        class_indices_sorted = sorted(
            range(num_classes_iou),
            key=lambda c: (
                iou_in_final_epoch[c] if not np.isnan(iou_in_final_epoch[c]) else 0.0
            ),
            reverse=True,
        )

        plt.figure(figsize=(14, 8))

        # b) Schleife in absteigender IoU-Reihenfolge
        for c in class_indices_sorted:
            iou_values = []
            for epoch in epochs:
                val = iou_per_class[str(epoch)][c]
                if np.isnan(val):
                    val = 0.0
                iou_values.append(val)

            class_label = CLASS_NAMES.get(c, f"Class {c}")
            # Farbe aus deinem Dict, umgewandelt auf [0..1] (matplotlib erwartet float 0..1)
            rgb = np.array(class_idx_to_color.get(c, [0, 0, 0]), dtype=float) / 255.0

            plt.plot(epochs, iou_values, label=class_label, color=rgb)

        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title(f"IoU per Class Over Epochs (Model: {model_name})")
        plt.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()

        iou_path = os.path.join(cross_val_outdir, f"{model_name}_iou_per_class.png")
        plt.savefig(iou_path)
        plt.close()
        print(f"IoU-Kurven gespeichert unter: {iou_path}")


def safe_float(value):
    """Konvertiert 'value' sicher in Float. None, str oder NaN/Inf => 0.0"""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val


def plot_iou_per_class_for_all_runs(models_dir):
    """
    Durchsucht alle Modelle/Run-Ordner nach detailed_metrics.json,
    lädt IoU-Daten und erstellt einen Plot pro Run.
    Filtert Klassen mit reinem 0.0 IoU über alle Epochen hinweg.
    """
    model_names = [
        d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))
    ]

    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name)

        run_names = [
            r
            for r in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, r))
        ]

        for run_name in run_names:
            run_path = os.path.join(model_path, run_name)
            metrics_path = os.path.join(run_path, "detailed_metrics.json")

            if not os.path.exists(metrics_path):
                continue

            # JSON lesen und "NaN"/"Infinity" -> "null"
            with open(metrics_path, "r") as f:
                content = f.read()
            content = content.replace("NaN", "null")
            content = content.replace("Infinity", "null")
            content = content.replace("-Infinity", "null")

            try:
                metrics = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Fehler beim Laden von {metrics_path}: {e}")
                continue

            iou_per_class = metrics.get("iou_per_class", {})
            if not iou_per_class:
                print(f"Keine IoU-Daten in {metrics_path}, überspringe.")
                continue

            epochs = sorted(map(int, iou_per_class.keys()))
            if not epochs:
                print(f"Keine Epochen in den IoU-Daten, überspringe {metrics_path}.")
                continue

            final_epoch = max(epochs)
            num_classes = len(iou_per_class[str(epochs[0])])

            # IoU in der finalen Epoche laden
            iou_in_final = iou_per_class[str(final_epoch)]

            # 1) Klassen sortieren (absteigend nach IoU) via safe_float
            class_indices_sorted = sorted(
                range(num_classes),
                key=lambda c: safe_float(iou_in_final[c]),
                reverse=True,
            )

            # 2) Plot
            plt.figure(figsize=(14, 8))

            # Nur Klassen plotten, deren IoU != 0 (über alle Epochen)
            for c in class_indices_sorted:
                # IoU-Werte über alle Epochen hinweg sammeln
                iou_values = [
                    safe_float(iou_per_class[str(epoch)][c]) for epoch in epochs
                ]

                # Prüfen, ob *alle* Epochen 0.0 sind
                if all(val == 0.0 for val in iou_values):
                    # -> Klasse überspringen, sie taucht nicht auf
                    continue

                class_label = CLASS_NAMES.get(c, f"Class {c}")
                rgb = np.array(class_idx_to_color.get(c, [0, 0, 0])) / 255.0
                plt.plot(epochs, iou_values, label=class_label, color=rgb)

            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.title(f"IoU per Class (Model: {model_name}, Run: {run_name})")
            plt.grid(True)
            plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            out_plot_path = os.path.join(run_path, "iou_per_class.png")
            plt.savefig(out_plot_path)
            plt.close()

            print(f"IoU-Plot gespeichert unter: {out_plot_path}")
