import os
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(model_name, run_name):
    # run_name ist nun der vollständige Ordnername, z. B. "run_001_12-12-2024"
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(os.path.dirname(src_dir), "output")
    selected_model_dir = os.path.join(models_dir, model_name, run_name)

    csv_file_path = os.path.join(selected_model_dir, "metrics.csv")
    if not os.path.exists(csv_file_path):
        print(
            f"Keine metrics.csv für Modell {model_name} im Verzeichnis {selected_model_dir} gefunden."
        )
        return

    df = pd.read_csv(csv_file_path)
    print("Verfügbare Spalten:", df.columns)
    print(df.head())

    epochs = df["Epoch"]

    # Plot Loss
    if "Train Loss" in df.columns and "Val Loss" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df["Train Loss"], label="Train Loss")
        plt.plot(epochs, df["Val Loss"], label="Val Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {model_name} ({run_name})")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(selected_model_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss-Plot gespeichert unter: {plot_path}")

    # Plot Accuracy
    if "Train Acc" in df.columns and "Val Acc" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df["Train Acc"], label="Train Acc")
        plt.plot(
            epochs,
            df["Val Acc"],
            label="Val Acc",
            linestyle="--",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Training and Validation Accuracy - {model_name} ({run_name})")
        plt.legend()
        plt.grid(True)
        acc_plot_path = os.path.join(selected_model_dir, "accuracy_curve.png")
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"Accuracy-Plot gespeichert unter: {acc_plot_path}")
    else:
        print("Keine Accuracy-Daten gefunden, überspringe Accuracy-Plot.")

    # Plot mIoU
    if "Val mIoU" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df["Val mIoU"], label="Validation mIoU", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Mean IoU")
        plt.title(f"Validation Mean IoU - {model_name} ({run_name})")
        plt.legend()
        plt.grid(True)
        miou_plot_path = os.path.join(selected_model_dir, "miou_curve.png")
        plt.savefig(miou_plot_path)
        plt.close()
        print(f"mIoU-Plot gespeichert unter: {miou_plot_path}")
    else:
        print("Keine mIoU-Daten gefunden, überspringe mIoU-Plot.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluationsskript für Modelle.")
    parser.add_argument("--model", type=str, default="resnet34", help="Modellname")
    parser.add_argument(
        "--run",
        type=str,
        default="run_024_15-01-2025",
        #required=True,
        help="Run-Verzeichnis, z. B. run_001_12-12-2024",
    )
    args = parser.parse_args()
    evaluate_model(args.model, args.run)
