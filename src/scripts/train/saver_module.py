# src/scripts/saver_module.py

import os
import csv
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class GenericMetricsSaver:
    """
    Generischer Metrik-Saver, der CSV und JSON schreibt.
    Optional kann er auch einen TensorBoard-Writer verwalten.
    """

    def __init__(self, save_dir, enable_tensorboard=True):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.enable_tensorboard = enable_tensorboard
        self.writer = None
        if self.enable_tensorboard:
            log_dir = os.path.join(self.save_dir, "logs")
            self.writer = SummaryWriter(log_dir=log_dir)

        # Interne Listen für Standardmetriken
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_miou = []
        self.confusions = {}  # z.B. {epoch: [[...], [...], ...]}
        self.iou_per_class = {}  # z.B. {epoch: [iou0, iou1, ...]}

        # CSV-Datei vorbereiten (nur Header)
        self.csv_path = os.path.join(self.save_dir, "metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Epoch",
                        "Train Loss",
                        "Val Loss",
                        "Train Acc",
                        "Val Acc",
                        "Val mIoU",
                    ]
                )

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, val_miou):
        """
        Speichert die Metriken für eine Epoche:
          - CSV (Append)
          - Interne Listen
          - TensorBoard (falls aktiv)
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.val_miou.append(val_miou)

        # CSV-Append
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, val_miou])

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            self.writer.add_scalar("IoU/Validation", val_miou, epoch)

    def log_confusion_iou(self, epoch, confusion_matrix, iou_per_class):
        """
        Speichert (im Speicher) die Konfusionsmatrix und IoU pro Klasse.
        """
        # Wandeln in (möglichst) Python-Listen, damit JSON-serialisierbar
        self.confusions[epoch] = confusion_matrix.tolist()
        self.iou_per_class[epoch] = list(map(float, iou_per_class))

        # Optional: Auch in TensorBoard, z. B. iou pro Klasse als "histogram"
        if self.writer:
            for c, iou in enumerate(iou_per_class):
                self.writer.add_scalar(f"IoU_per_class/Class_{c}", iou, epoch)

    def save_json(self):
        """
        Speichert den Verlauf in 'metrics.json' und
        die Konfusions- & IoU-Daten in 'detailed_metrics.json'.
        """
        # Erstmal Standard-Metriken
        json_path = os.path.join(self.save_dir, "metrics.json")
        data = {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_acc,
            "val_accuracy": self.val_acc,
            "val_miou": self.val_miou,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        # Dann Detail-Metriken
        detailed_path = os.path.join(self.save_dir, "detailed_metrics.json")
        data_detailed = {
            "confusion_matrices": self.confusions,
            "iou_per_class": self.iou_per_class,
        }
        with open(detailed_path, "w") as f:
            json.dump(data_detailed, f, indent=4)

    def close(self):
        """
        Schließt ggf. den TensorBoard-Writer.
        """
        if self.writer:
            self.writer.close()
