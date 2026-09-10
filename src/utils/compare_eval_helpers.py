import os
import pandas as pd
import numpy as np
import json
from functools import reduce
import matplotlib.colors as mcolors


def get_model_color_map(model_names):
    """
    Gibt ein Dict zurück, das jedem Modellnamen (als Key)
    eine Farbe (als Wert) zuordnet.
    """
    # Beispiel: wir nutzen die TABLEAU_COLORS von matplotlib
    all_colors = list(mcolors.TABLEAU_COLORS.values())

    # Falls mehr Modelle als Farben, wiederholen wir das Farbset,
    # bis wir genug Farben haben:
    needed = len(model_names) - len(all_colors)
    if needed > 0:
        # Sofern man das möchte, könnte man die Farben einfach wiederholen
        # oder eine andere Colormap nutzen
        all_colors = all_colors * (needed // len(all_colors) + 2)

    # Sortierte Modellnamen, damit die Reihenfolge immer konsistent bleibt
    model_names_sorted = sorted(model_names)

    # Farbzuordnung erstellen
    model_color_map = {name: all_colors[i] for i, name in enumerate(model_names_sorted)}
    return model_color_map


def get_latest_run(model_dir):
    """
    Findet den neuesten Run in einem Modellverzeichnis.
    """
    runs = [
        d
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("run_")
    ]
    if not runs:
        return None
    runs.sort()
    return runs[-1]  # Neuester Run


def load_metrics(model_name, models_dir):
    """
    Lädt die Metriken des neuesten Runs eines Modells (ohne Cross-Val).
    Gibt (DataFrame, run_ordnername) zurück oder (None, None) bei Fehler.
    """
    model_dir = os.path.join(models_dir, model_name)
    latest_run = get_latest_run(model_dir)
    if not latest_run:
        print(f"Kein Run gefunden für Modell: {model_name}")
        return None, None

    csv_file_path = os.path.join(model_dir, latest_run, "metrics.csv")
    if not os.path.exists(csv_file_path):
        print(f"Keine metrics.csv gefunden für {model_name}, Run: {latest_run}")
        return None, None

    df = pd.read_csv(csv_file_path)
    return df, latest_run


def load_cross_val_metrics(model_name, models_dir):
    """
    Lädt die Metriken für alle Folds des neuesten Runs eines Modells.
    Gibt ein Dict zurück:
    {
      'fold_1': pd.DataFrame(...),
      'fold_2': pd.DataFrame(...),
      ...
    }
    und zusätzlich den Pfad zum Run-Ordner.

    Falls keine Folds gefunden werden, wird ein leeres Dict zurückgegeben.
    """
    model_dir = os.path.join(models_dir, model_name)
    latest_run = get_latest_run(model_dir)
    if not latest_run:
        print(f"Kein Run gefunden für Modell (CrossVal): {model_name}")
        return {}, None

    run_path = os.path.join(model_dir, latest_run)
    # Alle fold_x Ordner
    fold_dirs = [
        d
        for d in os.listdir(run_path)
        if os.path.isdir(os.path.join(run_path, d)) and d.startswith("fold_")
    ]
    fold_dirs.sort()

    folds_data = {}
    for fdir in fold_dirs:
        csv_file_path = os.path.join(run_path, fdir, "metrics.csv")
        if os.path.exists(csv_file_path):
            try:
                df = pd.read_csv(csv_file_path)
                folds_data[fdir] = df
            except Exception as e:
                print(f"Fehler beim Einlesen {csv_file_path}: {e}")
        else:
            print(f"Keine metrics.csv in {fdir} gefunden.")

    return folds_data, run_path


def load_kfold_summary(run_path):
    """
    Lädt das k_fold_summary.csv aus dem Run-Ordner (falls vorhanden).
    Gibt ein DataFrame zurück oder None, wenn nicht vorhanden oder Fehler.
    """
    summary_path = os.path.join(run_path, "k_fold_summary.csv")
    if not os.path.exists(summary_path):
        return None

    try:
        df = pd.read_csv(summary_path)
        return df
    except Exception as e:
        print(f"Fehler beim Einlesen der k_fold_summary.csv: {e}")
        return None


def compute_fold_averages(folds_data):
    """
    Nimmt einen Dict { 'fold_1': df1, 'fold_2': df2, ... }
    und berechnet für jede Epoch den Mittelwert und die Standardabweichung
    aller Metriken (außer 'Epoch') über die Folds hinweg.

    Rückgabe:
      df_mean, df_std
    Beide haben dieselbe Form wie eine gewöhnliche Metrics-Tabelle,
    mit den Spalten: ['Epoch', 'Metric1', 'Metric2', ...]
    """
    if not folds_data:
        return None, None

    # Statt reduce(... pd.merge) verwenden wir concat:
    df_concat = pd.concat(
        [df.set_index("Epoch") for df in folds_data.values()],
        axis=1,
        keys=folds_data.keys(),
    )

    # Mittelwert pro "Spalten-Ebene 1" (also die Metrik) bilden
    df_mean = df_concat.groupby(level=1, axis=1).mean()
    df_std = df_concat.groupby(level=1, axis=1).std()

    # Index (= Epochen) wieder zur normalen Spalte machen
    df_mean.reset_index(inplace=True)
    df_std.reset_index(inplace=True)

    return df_mean, df_std
