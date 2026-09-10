import os
import matplotlib.pyplot as plt
from utils.compare_eval_helpers import load_metrics, get_model_color_map


def plot_normal_case(models_dir):
    """
    Originale Logik ohne Cross-Validation:
    Lädt von jedem Modell den neuesten Run und plottet metrics.csv.
    """
    # Alle Modelle identifizieren
    from os import listdir
    from os.path import isdir, join

    model_names = [d for d in listdir(models_dir) if isdir(join(models_dir, d))]

    # Metriken sammeln
    metrics = {}
    for model_name in model_names:
        df, latest_run = load_metrics(model_name, models_dir)
        if df is not None:
            metrics[model_name] = {"data": df, "run": latest_run}

    if not metrics:
        print("Keine Metriken verfügbar (ohne Cross-Validation).")
        return

    # Farbzuordnung holen
    model_color_map = get_model_color_map(metrics.keys())

    # Alle Metrik-Spalten (außer 'Epoch') ermitteln
    all_metric_columns = set()
    for model_name, content in metrics.items():
        df = content["data"]
        columns_without_epoch = set(df.columns) - {"Epoch"}
        all_metric_columns = all_metric_columns.union(columns_without_epoch)

    # Für jede Metrik einen Plot erstellen
    for metric_col in all_metric_columns:
        plt.figure(figsize=(12, 8))
        for model_name, content in metrics.items():
            df = content["data"]
            if metric_col in df.columns:
                plt.plot(
                    df["Epoch"],
                    df[metric_col],
                    label=f'{model_name} (Run: {content["run"]})',
                    color=model_color_map[model_name],
                )

        plt.xlabel("Epoch")
        plt.ylabel(metric_col)
        plt.title(f"{metric_col} Across Models (ohne Cross-Val)")
        plt.legend()
        plt.grid(True)

        # Dateiname für den Plot
        metric_filename = metric_col.lower().replace(" ", "_")
        output_path = os.path.join(
            models_dir, f"comparison_{metric_filename}_curve.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"{metric_col} Plot gespeichert unter: {output_path}")
