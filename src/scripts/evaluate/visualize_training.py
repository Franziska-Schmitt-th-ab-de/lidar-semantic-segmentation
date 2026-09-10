import os
import json
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Visualisierung von Trainings- und Validierungsverlusten."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Pfad zum Modellverzeichnis mit den gespeicherten Verlusten.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name des Modells (z.B. resnet34, sn5).",
    )
    args = parser.parse_args()

    # Pfad zur Verluste-Datei konstruieren
    losses_file = os.path.join(args.model_dir, f"losses_{args.model_name}.json")
    if not os.path.exists(losses_file):
        print(f"Verluste-Datei nicht gefunden: {losses_file}")
        return

    # Verluste laden
    with open(losses_file, "r") as f:
        losses = json.load(f)

    train_loss = losses["train_loss"]
    val_loss = losses["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, "b-", label="Trainingsverlust")
    plt.plot(epochs, val_loss, "r-", label="Validierungsverlust")
    plt.title("Trainings- und Validierungsverlust")
    plt.xlabel("Epochen")
    plt.ylabel("Verlust")
    plt.legend()
    plt.grid(True)

    # Plot speichern
    plot_file = os.path.join(args.model_dir, f"loss_plot_{args.model_name}.png")
    plt.savefig(plot_file)
    print(f"Plot gespeichert unter {plot_file}")

    # Plot anzeigen (optional)
    plt.show()


if __name__ == "__main__":
    main()
