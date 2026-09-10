#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Eigene Importe
from models import get_model
from utils.cross_entropy_loss import SemanticSegmentationLoss
from utils.lr_scheduler import get_scheduler
from utils.training_utils import get_run_directory
from scripts.evaluate.validate_module import validate
from train.saver_module import GenericMetricsSaver


def k_fold_cross_validation(config, LiDARDataset, device):
    """
    Führt K-Fold-Cross-Validation auf allen in config['scenes'] enthaltenen Szenen durch.
    Pro Fold wird jeweils eine Szene als Validation verwendet.

    Änderungen:
    - Speichert nun alle Epoche-Checkpoints (model_epoch_{epoch}.pth) in fold_x/
    - Bewahrt zusätzlich den besten Checkpoint als best_model.pth auf.
    """
    dataset_dir = config["dataset_dir"]
    scenes = config["scenes"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    selected_model = config["model"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(os.path.dirname(src_dir), "output")
    selected_model_dir = get_run_directory(models_dir, selected_model)

    fold_final_losses = []
    fold_final_val_accuracies = []
    fold_final_val_losses = []
    fold_final_train_accuracies = []
    fold_final_val_mious = []
    best_fold_metrics = []

    for fold_idx, val_scene in enumerate(scenes):
        print(f"\n=== Fold {fold_idx+1}/{len(scenes)}: Validation on '{val_scene}' ===")

        train_scenes = [sc for sc in scenes if sc != val_scene]
        print(f"Training scenes: {train_scenes}, Validation scene: {val_scene}")

        # Datasets
        train_dataset = LiDARDataset(dataset_dir, train_scenes)
        val_dataset = LiDARDataset(dataset_dir, [val_scene])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

        # Modell
        model = get_model(selected_model, num_classes=num_classes).to(device)
        criterion = SemanticSegmentationLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scaler = GradScaler()
        scheduler = get_scheduler(
            optimizer, scheduler_type="StepLR", step_size=10, gamma=0.1
        )

        # Ordner für diesen Fold
        fold_save_dir = os.path.join(selected_model_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # Neuer Saver für diesen Fold
        saver = GenericMetricsSaver(save_dir=fold_save_dir, enable_tensorboard=True)

        best_val_accuracy = 0.0
        best_val_miou = 0.0
        best_val_loss = float("inf")
        best_epoch_idx = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            correct_predictions = 0
            total_pixels = 0

            for data, target in tqdm(
                train_loader, desc=f"Fold {fold_idx+1} - Epoch {epoch+1}/{num_epochs}"
            ):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast():
                    output = model(data)
                    if output.size()[2:] != target.size()[1:]:
                        output = F.interpolate(
                            output,
                            size=target.shape[1:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    loss = criterion(output, target, num_classes=num_classes)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                # Accuracy
                mask = target != 255
                preds = output.argmax(dim=1)
                correct_predictions += (preds[mask] == target[mask]).sum().item()
                total_pixels += mask.sum().item()

            avg_train_loss = (
                epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            )
            train_accuracy = (
                correct_predictions / total_pixels if total_pixels > 0 else 0
            )

            # Validation
            val_loss, val_accuracy, val_mean_iou = validate(
                model, val_loader, criterion, device, num_classes
            )

            # Scheduler aktualisieren
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Logging
            print(
                f"[Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, mIoU: {val_mean_iou:.4f}"
            )

            # Metriken in Saver
            saver.log_epoch(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                train_acc=train_accuracy,
                val_acc=val_accuracy,
                val_miou=val_mean_iou,
            )

            # == SPEICHERN ALLER EPOCHEN ==
            epoch_model_path = os.path.join(fold_save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_model_path)

            # Best Model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_miou = val_mean_iou
                best_val_loss = val_loss
                best_epoch_idx = epoch + 1

                best_model_path = os.path.join(fold_save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)

        # Epoche(n) fertig => Speichere CSV/JSON, schließe Writer
        saver.save_json()
        saver.close()

        # Letzte Epoche: Metriken in Aggregationslisten
        fold_final_losses.append(saver.train_loss[-1])
        fold_final_val_losses.append(saver.val_loss[-1])
        fold_final_train_accuracies.append(saver.train_acc[-1])
        fold_final_val_accuracies.append(saver.val_acc[-1])
        fold_final_val_mious.append(saver.val_miou[-1])

        # Best-Fold-Metriken speichern
        best_fold_metrics.append(
            {
                "fold_idx": fold_idx + 1,
                "val_scene": val_scene,
                "best_epoch": best_epoch_idx,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "best_val_miou": best_val_miou,
            }
        )

    # Am Ende: Aggregierte Metriken (auf Basis der letzten Epoche pro Fold)
    avg_final_train_loss = sum(fold_final_losses) / len(fold_final_losses)
    avg_final_train_acc = sum(fold_final_train_accuracies) / len(
        fold_final_train_accuracies
    )
    avg_final_val_loss = sum(fold_final_val_losses) / len(fold_final_val_losses)
    avg_final_val_acc = sum(fold_final_val_accuracies) / len(fold_final_val_accuracies)
    avg_final_val_miou = sum(fold_final_val_mious) / len(fold_final_val_mious)

    print("\n=== Summary (Final Epoch per Fold) ===")
    print(f"Avg Train Loss  : {avg_final_train_loss:.4f}")
    print(f"Avg Train Acc   : {avg_final_train_acc:.4f}")
    print(f"Avg Val Loss    : {avg_final_val_loss:.4f}")
    print(f"Avg Val Acc     : {avg_final_val_acc:.4f}")
    print(f"Avg Val mIoU    : {avg_final_val_miou:.4f}")

    # Optional: best_fold_metrics in JSON
    """
    best_json_path = os.path.join(selected_model_dir, "best_folds.json")
    with open(best_json_path, "w") as f:
        import json
        json.dump(best_fold_metrics, f, indent=4)
    """
