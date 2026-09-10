import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.cross_entropy_loss import SemanticSegmentationLoss
from utils.dataloader import LiDARDataset
from utils.lr_scheduler import get_scheduler
from utils.training_utils import load_config, get_run_directory
from scripts.evaluate.validate_module import validate
from scripts.evaluate.evaluate import evaluate_model
from models import get_model

from train.saver_module import GenericMetricsSaver


def train_single_model(selected_model, config, val_scene, device="cuda"):
    """
    Führt das Training für ein einzelnes Modell durch.
    :param selected_model: z.B. "sn5"
    :param config: Dictionary aus load_config (Keys: dataset_dir, scenes, batch_size, ...)
    :param val_scene: z.B. "0001" oder "random"
    :param device: "cuda" oder "cpu"
    """

    # Check GPU
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but device=cuda was selected.")

    dataset_dir = config["dataset_dir"]
    scenes = config["scenes"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]

    # Val-Szene ggf. random wählen
    if val_scene == "random":
        val_scene = random.choice(scenes)

    if val_scene not in scenes:
        raise ValueError(
            f"Die angegebene Validierungsszene {val_scene} ist nicht in den verfügbaren Szenen: {scenes}"
        )
    val_scenes = [val_scene]
    train_scenes = [s for s in scenes if s not in val_scenes]

    print(f"Trainingsszenen: {train_scenes}")
    print(f"Validierungsszenen: {val_scenes}")

    # Datasets & DataLoader
    train_dataset = LiDARDataset(dataset_dir, train_scenes)
    val_dataset = LiDARDataset(dataset_dir, val_scenes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # Modell + Optimizer
    model = get_model(selected_model, num_classes=num_classes).to(device)
    criterion = SemanticSegmentationLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = get_scheduler(
        optimizer, scheduler_type="StepLR", step_size=10, gamma=0.1
    )

    # Ordnerstruktur für Saves
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(os.path.dirname(src_dir), "output")
    selected_model_dir = get_run_directory(models_dir, selected_model)

    # Erstelle Saver für Metriken
    # (TensorBoard standardmäßig an, kannst du sonst via Param steuern)
    saver = GenericMetricsSaver(save_dir=selected_model_dir, enable_tensorboard=True)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_pixels = 0

        for batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Evtl. NaN/Inf check
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"[WARN] Batch {batch_idx} hat NaN/Inf in Eingabedaten!")
                continue
            if torch.isnan(target).any() or torch.isinf(target).any():
                print(f"[WARN] Batch {batch_idx} hat NaN/Inf in Labels!")
                continue

            optimizer.zero_grad()
            output = model(data)

            # ggf. Resize
            if output.size()[2:] != target.size()[1:]:
                output = F.interpolate(
                    output, size=target.shape[1:], mode="bilinear", align_corners=False
                )

            # Loss
            loss = criterion(output, target, num_classes=num_classes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Accuracy
            mask = target != 255
            predictions = output.argmax(dim=1)
            correct_predictions += (predictions[mask] == target[mask]).sum().item()
            total_pixels += mask.sum().item()

        # Metriken pro Epoche
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_accuracy = correct_predictions / total_pixels if total_pixels > 0 else 0

        results = validate(
            model, val_loader, criterion, device, num_classes
        )
        val_loss, val_accuracy, val_mean_iou, confusion_matrix, iou_per_class = results

        # Logging
        print(
            f"Epoch [{epoch+1}/{num_epochs}]: "
            f"TrainLoss={avg_loss:.4f}, TrainAcc={train_accuracy:.4f}, "
            f"ValLoss={val_loss:.4f}, ValAcc={val_accuracy:.4f}, "
            f"mIoU={val_mean_iou:.4f}"
        )

        # NEU: Konfusionsmatrix + Per-Class-IoU an Saver übergeben
        saver.log_confusion_iou(
            epoch=epoch + 1,
            confusion_matrix=confusion_matrix,
            iou_per_class=iou_per_class,
        )

        # Scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # -> Saver statt manuelle CSV/JSON/TensorBoard
        saver.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=val_loss,
            train_acc=train_accuracy,
            val_acc=val_accuracy,
            val_miou=val_mean_iou,
        )

        # Modell pro Epoche speichern
        model_save_path = os.path.join(
            selected_model_dir, f"model_{selected_model}_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), model_save_path)

    # Nach allen Epochen -> JSON, Writer zu
    saver.save_json()
    saver.close()

    # Auswertung
    run_name = os.path.basename(selected_model_dir)
    evaluate_model(selected_model, run_name)
    print("[INFO] Automatische Auswertung abgeschlossen.")

    # Fehlerprotokolle
    print("[INFO] Speichere Fehlerprotokolle (train & val)...")
    train_dataset.save_errors()
    val_dataset.save_errors()
    print("[INFO] Fehlerprotokolle gespeichert.")
