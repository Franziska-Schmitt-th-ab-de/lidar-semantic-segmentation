import os
import yaml
import torch
import numpy as np


def load_config(config_path):
    """Lädt die YAML-Konfigurationsdatei."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_iou(preds, labels, num_classes, ignore_index=255):
    """Berechnet IoU pro Klasse."""
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


def get_run_directory(base_dir, model_name):
    """
    Bestimmt das Verzeichnis für den nächsten Run und erstellt es.
    """
    import datetime

    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Bestehende Runs finden und sortieren
    existing_runs = [
        d
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("run_")
    ]
    existing_runs.sort()

    # Nächsten Run-Index bestimmen
    if existing_runs:
        last_run = existing_runs[-1]
        last_index = int(last_run.split("_")[1])
        next_index = last_index + 1
    else:
        next_index = 1

    now = datetime.datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    new_run_dir_name = f"run_{next_index:03d}_{date_str}"
    new_run_dir = os.path.join(model_dir, new_run_dir_name)
    os.makedirs(new_run_dir, exist_ok=True)

    return new_run_dir
