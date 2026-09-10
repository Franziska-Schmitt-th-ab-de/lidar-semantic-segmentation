import argparse
import os
import subprocess
import sys
import torch

from utils.seed_utils import seed_everything
from scripts.train.train_module import load_config, train_single_model
from scripts.train.crossval_module import k_fold_cross_validation


def query_or_default(prompt, default_value, cast_func):
    """
    Hilfsfunktion: Fragt den Benutzer nach einem Wert und gibt entweder
    die Benutzereingabe oder den Default-Wert zurück.
    """
    user_input = input(f"{prompt} (Leer => {default_value}): ").strip()
    if user_input:
        try:
            return cast_func(user_input)
        except ValueError:
            print(f"[WARN] Ungültige Eingabe. Nutze Default-Wert {default_value}.")
    return default_value


def main():
    """
    Haupt-Einstiegspunkt für das Training bzw. Cross-Validation.
    """

    # -------------------------------------------------------
    # 1) CLI-Parser
    # -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Main entry point for semantic segmentation training."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modellname (z.B. 'sn5', 'resnet18' oder 'all'). Leer => Interaktiver Modus.",
    )
    parser.add_argument(
        "--val_scene",
        type=str,
        default=None,
        help="Val-Szene (z.B. '0001'). 'random' => zufällige. Leer => Interaktiv.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Anzahl der Epochen. Wenn nicht gesetzt, nutze Config oder Interaktiv.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch Size. Wenn nicht gesetzt, nutze Config oder Interaktiv.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning Rate. Wenn nicht gesetzt, nutze Config oder Interaktiv.",
    )
    parser.add_argument(
        "--cross_val",
        action="store_true",
        help="Aktiviere K-Fold Cross Validation (ignoriert val_scene!).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionaler Seed für Reproduzierbarkeit (sonst kein Seed).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------
    # 2) Device bestimmen (CUDA oder CPU) und ggf. bestätigen
    # -------------------------------------------------------
    # Zuerst herausfinden, ob eine CUDA-GPU verfügbar ist ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Verwende Device: {device}")

    # Falls nur CPU verfügbar ist: Noch einmal vom User bestätigen lassen
    if device.type == "cpu":
        confirm_cpu = (
            input(
                "GPU ist nicht verfügbar. Möchtest du wirklich auf der CPU trainieren? (j/n, Leer => j): "
            )
            .strip()
            .lower()
        )
        if confirm_cpu == "n":
            print("[INFO] Abbruch: Training auf CPU wurde vom User abgelehnt.")
            sys.exit(0)

    # -------------------------------------------------------
    # 3) Seed abfragen (falls nicht per CLI)
    # -------------------------------------------------------
    seed_value = args.seed
    if seed_value is None:
        # Interaktiv fragen, ob ein Seed gesetzt werden soll
        seed_input = input(
            "Möchtest du einen Seed setzen? (Enter für kein Seed): "
        ).strip()
        if seed_input:
            try:
                seed_value = int(seed_input)
            except ValueError:
                print("[WARN] Ungültige Seed-Eingabe. Kein Seed gesetzt.")
                seed_value = None

    if seed_value is not None:
        print(f"[INFO] Verwende Seed {seed_value} für Reproduzierbarkeit.")
        seed_everything(seed_value)

    # -------------------------------------------------------
    # 4) Modelle definieren (mit 'all' als 6. Option)
    # -------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_dir = os.path.join(script_dir, "evaluate")
    config_dir = os.path.join(script_dir, "../configs")

    available_models = {
        "regnet_y_800_mf": os.path.join(config_dir, "config_regnet_800_mf.yaml"),
        "resnet18": os.path.join(config_dir, "config_resnet18.yaml"),
        "resnet34": os.path.join(config_dir, "config_resnet34.yaml"),
        "sn0": os.path.join(config_dir, "config_shufflenet_v2_x1_0.yaml"),
        "sn5": os.path.join(config_dir, "config_shufflenet_v2_x1_5.yaml"),
        "all": None,  # <- Option "all"
    }

    # -------------------------------------------------------
    # 5) Modell abfragen (interaktiv oder CLI)
    # -------------------------------------------------------
    if not args.model:
        print("Wähle ein Modell aus (Zahl eingeben):")
        model_keys = list(available_models.keys())  # enthält "all" als Option
        for i, mk in enumerate(model_keys, start=1):
            print(f"{i}) {mk}")

        choice = input("Deine Wahl: ").strip()
        try:
            chosen_model = model_keys[int(choice) - 1]
        except (ValueError, IndexError):
            print("[WARN] Ungültige Auswahl, fallback auf erstes Modell.")
            chosen_model = model_keys[0]
    else:
        chosen_model = args.model
        if chosen_model not in available_models:
            print(
                f"[WARN] Unbekanntes Modell '{chosen_model}'. Nutze 'sn5' als Fallback."
            )
            chosen_model = "sn5"

    # -------------------------------------------------------
    # 6) Interaktiv abfragen, ob CrossVal (falls nicht per CLI)
    # -------------------------------------------------------
    if not args.cross_val:
        user_input_cv = input("Cross Validation? (j/n, Leer => n): ").strip().lower()
        use_cross_val = user_input_cv == "j"
    else:
        use_cross_val = True

    # -------------------------------------------------------
    # 7) Frage nach Compare Evaluations (mit Default = "ja")
    # -------------------------------------------------------
    compare_input = (
        input("compare_evaluations.py am Ende ausführen? (j/n, Leer => j): ")
        .strip()
        .lower()
    )
    compare_now = compare_input != "n"  # 'n' => False, sonst True

    # -------------------------------------------------------
    # 8) Val-Szene nur fragen, wenn KEIN CrossVal
    # -------------------------------------------------------
    val_scene_input = None
    if not use_cross_val:
        # Nur bei CrossVal = False macht val_scene Sinn
        if args.val_scene is not None:
            val_scene_input = args.val_scene
        else:
            val_scene_input = query_or_default(
                "Welche Validierungsszene möchtest du verwenden? ('random' => zufällig)",
                "random",
                str,
            )
    else:
        print("[INFO] K-Fold aktiv: Die Angabe einer Validierungsszene wird ignoriert.")

    # -------------------------------------------------------
    # 9) Falls chosen_model != "all": Config laden & Training
    # -------------------------------------------------------
    if chosen_model != "all":
        config_path = available_models[chosen_model]
        config = load_config(config_path)

        # Epochen
        config["num_epochs"] = args.epochs or query_or_default(
            "Anzahl Epochen", config["num_epochs"], int
        )
        # Batch Size
        config["batch_size"] = args.batch_size or query_or_default(
            "Batch Size", config["batch_size"], int
        )
        # Learning Rate
        config["learning_rate"] = args.learning_rate or query_or_default(
            "Learning Rate", config["learning_rate"], float
        )
        config["model"] = chosen_model  # sicherheitshalber setzen

        # Dataloader importieren
        from utils.dataloader import LiDARDataset

        if use_cross_val:
            print(f"[INFO] Starte Cross Validation für Modell: {chosen_model}")
            k_fold_cross_validation(config, LiDARDataset, device)
        else:
            print(f"[INFO] Starte Training für Modell: {chosen_model}")
            train_single_model(chosen_model, config, val_scene_input, device=device)

    else:
        # -------------------------------------------------------
        # 10) "all": Wir fragen EINMAL nach Epochen, BS, LR und nutzen
        #     dieselben Werte für alle Modelle.
        # -------------------------------------------------------
        model_list = [m for m in available_models.keys() if m != "all"]

        # => Lade exemplarisch eine Config (z.B. die erste), nur um Defaults zu holen
        first_model = model_list[0]
        first_config = load_config(available_models[first_model])

        # Interaktiv / CLI:
        common_epochs = args.epochs or query_or_default(
            "[ALL] Anzahl Epochen", first_config["num_epochs"], int
        )
        common_batch_size = args.batch_size or query_or_default(
            "[ALL] Batch Size", first_config["batch_size"], int
        )
        common_lr = args.learning_rate or query_or_default(
            "[ALL] Learning Rate", first_config["learning_rate"], float
        )

        from utils.dataloader import LiDARDataset

        for model_name in model_list:
            config_path = available_models[model_name]
            config = load_config(config_path)

            # Setze überall dieselben Werte
            config["num_epochs"] = common_epochs
            config["batch_size"] = common_batch_size
            config["learning_rate"] = common_lr
            config["model"] = model_name

            print(f"\n=== Starte Training für Modell: {model_name} ===")

            if use_cross_val:
                k_fold_cross_validation(config, LiDARDataset, device)
            else:
                train_single_model(model_name, config, val_scene_input, device=device)

        print("\n[INFO] Fertig mit Training für alle Modelle.")

    # -------------------------------------------------------
    # 11) compare_evaluations.py ausführen (falls gewünscht)
    # -------------------------------------------------------
    if compare_now:
        print("\n[INFO] Starte Vergleichsauswertung (compare_evaluations.py).")
        compare_script = os.path.join(evaluate_dir, "compare_evaluations.py")
        try:
            subprocess.run(["python", compare_script], check=True)
            print("[INFO] Vergleichsauswertung abgeschlossen.")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Fehler bei der Ausführung von compare_evaluations.py: {e}")
    else:
        print("\n[INFO] Keine automatische Vergleichsauswertung angefordert.")

    print("\n[INFO] Alles fertig - bis zum nächsten Mal!")


if __name__ == "__main__":
    main()
