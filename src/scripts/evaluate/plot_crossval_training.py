import os
import matplotlib.pyplot as plt
import numpy as np
from utils.compare_eval_helpers import load_cross_val_metrics, load_kfold_summary, compute_fold_averages


def plot_cross_val_case(models_dir):
    """
    Cross-Validation-Logik:
    - Sucht für jedes Modell den neuesten Run.
    - Lädt alle Folds, sammelt Metrik-Spalten.
    - Plottet für jeden Fold (vergleichbar zwischen den Modellen).
    - Liest außerdem k_fold_summary.csv ein, um pro Modell den "besten Fold" zu identifizieren.
    - Bildet außerdem den Durchschnitt + StdAbw. aller Folds (pro Epoch) und plottet diese.
    """
    cross_val_outdir = os.path.join(models_dir, "evaluation_output")
    os.makedirs(cross_val_outdir, exist_ok=True)

    from os import listdir
    from os.path import isdir, join

    # Alle Modelle identifizieren
    model_names = [d for d in listdir(models_dir) if isdir(join(models_dir, d))]

    cross_val_info = {}
    for model_name in model_names:
        folds_data, run_path = load_cross_val_metrics(model_name, models_dir)
        if run_path is None or not folds_data:
            # Entweder kein Run oder keine folds -> ignorieren
            continue

        summary_df = load_kfold_summary(run_path)
        df_mean, df_std = compute_fold_averages(folds_data)

        cross_val_info[model_name] = {
            "run_path": run_path,
            "folds_data": folds_data,
            "summary": summary_df,
            "df_mean": df_mean,
            "df_std": df_std,
        }

    if not cross_val_info:
        print("Keine Cross-Validation-Daten verfügbar.")
        return

    # ----------------------------------------
    # 1) Vergleich: Gleiche Fold-Nummern über alle Modelle hinweg
    # ----------------------------------------
    all_folds = set()
    for model_name, info in cross_val_info.items():
        all_folds = all_folds.union(info["folds_data"].keys())
    all_folds = sorted(list(all_folds))

    # Sammle alle Metrik-Spalten (außer 'Epoch')
    all_metric_columns = set()
    for model_name, info in cross_val_info.items():
        for fold, df_fold in info["folds_data"].items():
            columns_without_epoch = set(df_fold.columns) - {"Epoch"}
            all_metric_columns = all_metric_columns.union(columns_without_epoch)

    # Pro fold_x ein Plot pro Metrik
    for fold in all_folds:
        for metric_col in all_metric_columns:
            plt.figure(figsize=(12, 8))
            legend_entries = False
            for model_name, info in cross_val_info.items():
                df_fold = info["folds_data"].get(fold, None)
                if df_fold is not None and metric_col in df_fold.columns:
                    plt.plot(
                        df_fold["Epoch"],
                        df_fold[metric_col],
                        label=f"{model_name} - {fold}",
                    )
                    legend_entries = True

            if not legend_entries:
                plt.close()
                continue

            plt.xlabel("Epoch")
            plt.ylabel(metric_col)
            plt.title(f"CrossVal: {metric_col} - {fold} (Modellvergleich)")
            plt.legend()
            plt.grid(True)

            fold_str = fold.replace("_", "")  # z.B. fold1 statt fold_1
            metric_filename = metric_col.lower().replace(" ", "_")
            outname = f"comparison_{fold_str}_{metric_filename}.png"
            output_path = os.path.join(cross_val_outdir, outname)
            plt.savefig(output_path)
            plt.close()
            print(f"Plot gespeichert: {output_path}")

    # ----------------------------------------
    # 2) Vergleich: Bester Fold pro Modell (z.B. nach höchster Best Val Acc)
    # ----------------------------------------
    best_fold_per_model = {}
    for model_name, info in cross_val_info.items():
        summary = info["summary"]
        if summary is None:
            continue
        if "Best Val Acc" not in summary.columns:
            continue
        idx_best = summary["Best Val Acc"].idxmax()
        fold_best = summary.loc[idx_best, "Fold"]  # z.B. 1, 2, 3, ...
        fold_name = f"fold_{int(fold_best)}"
        best_fold_per_model[model_name] = fold_name

    for metric_col in all_metric_columns:
        plt.figure(figsize=(12, 8))
        legend_entries = False
        for model_name, fold_name in best_fold_per_model.items():
            df_fold = cross_val_info[model_name]["folds_data"].get(fold_name, None)
            if df_fold is not None and metric_col in df_fold.columns:
                plt.plot(
                    df_fold["Epoch"],
                    df_fold[metric_col],
                    label=f"{model_name} - {fold_name}",
                )
                legend_entries = True

        if not legend_entries:
            plt.close()
            continue

        plt.xlabel("Epoch")
        plt.ylabel(metric_col)
        plt.title(f"CrossVal: {metric_col} - Beste Folds pro Modell")
        plt.legend()
        plt.grid(True)

        metric_filename = metric_col.lower().replace(" ", "_")
        outname = f"comparison_bestfold_{metric_filename}.png"
        output_path = os.path.join(cross_val_outdir, outname)
        plt.savefig(output_path)
        plt.close()
        print(f"Bester-Fold-Plot gespeichert: {output_path}")

    # ----------------------------------------
    # 3) Durchschnittlicher Verlauf über alle Folds (Mean & Std)
    # ----------------------------------------
    for metric_col in all_metric_columns:
        plt.figure(figsize=(12, 8))
        legend_entries = False

        for model_name, info in cross_val_info.items():
            df_mean = info["df_mean"]
            df_std = info["df_std"]

            if df_mean is None or df_std is None:
                continue
            if metric_col not in df_mean.columns:
                continue

            x = df_mean["Epoch"]
            y_mean = df_mean[metric_col]
            y_std = df_std[metric_col]

            plt.plot(x, y_mean, label=f"{model_name} (mean over folds)")
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

            legend_entries = True

        if not legend_entries:
            plt.close()
            continue

        plt.xlabel("Epoch")
        plt.ylabel(metric_col)
        plt.title(f"CrossVal Average: {metric_col} (Mean ± Std)")
        plt.legend()
        plt.grid(True)

        metric_filename = metric_col.lower().replace(" ", "_")
        outname = f"comparison_average_{metric_filename}.png"
        output_path = os.path.join(cross_val_outdir, outname)
        plt.savefig(output_path)
        plt.close()
        print(f"Durchschnitts-Plot gespeichert: {output_path}")
