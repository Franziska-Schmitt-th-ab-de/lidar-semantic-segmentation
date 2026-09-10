# src/scripts/evaluate/validate_module.py

import torch
import torch.nn.functional as F
import numpy as np


def validate(model, val_loader, criterion, device, num_classes):
    """
    Führt eine Validierung auf dem Validierungs-Dataloader durch.
    Gibt den Durchschnitts-Loss, Accuracy und mIoU zurück.
    """
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_pixels = 0

    # Konfusionsmatrix anlegen: shape [num_classes, num_classes]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)

            # ggf. Resize
            if output.size()[2:] != target.size()[1:]:
                output = F.interpolate(
                    output, size=target.shape[1:], mode="bilinear", align_corners=False
                )

            # Loss
            loss = criterion(output, target, num_classes=num_classes)
            val_loss += loss.item()

            # Predictions
            preds = output.argmax(dim=1)  # shape [B, H, W]
            # Maske für gültige Pixel (z. B. 255 ist Ignore-Label)
            mask = target != 255
            valid_preds = preds[mask]  # 1D-Tensor
            valid_targets = target[mask]  # 1D-Tensor

            # Accuracy-Teil
            total_correct += (valid_preds == valid_targets).sum().item()
            total_pixels += valid_targets.numel()

            # Update Konfusionsmatrix
            # Achtung: valid_targets und valid_preds müssen in CPU-Form vorliegen (np.array)
            for vt, vp in zip(valid_targets.cpu().numpy(), valid_preds.cpu().numpy()):
                if vt < num_classes:
                    confusion_matrix[vt, vp] += 1

    # Durchschnittsverlust
    val_loss /= len(val_loader) if len(val_loader) > 0 else 1
    val_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

    # -------------------------------------------------------------------
    # IoU pro Klasse aus Konfusionsmatrix berechnen
    # -------------------------------------------------------------------
    # confusion_matrix[c, c] -> True Positives für Klasse c
    # Summe über Zeile c -> total actual = TP + FN
    # Summe über Spalte c -> total predicted = TP + FP
    # IoU_c = TP / (TP + FP + FN)
    iou_per_class = []
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fn = confusion_matrix[c, :].sum() - tp
        fp = confusion_matrix[:, c].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            iou_per_class.append(float("nan"))  # oder 0.0
        else:
            iou_per_class.append(tp / denom)

    # Mittelwert der gültigen Klassen (mIoU)
    valid_iou = [x for x in iou_per_class if not np.isnan(x)]
    if valid_iou:
        mean_iou = float(np.mean(valid_iou))
    else:
        mean_iou = 0.0

    return val_loss, val_accuracy, mean_iou, confusion_matrix, iou_per_class
