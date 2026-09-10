import os
import time
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models import get_model  # Deine Model-Factory
from utils.dataloader import LiDARDataset  # Dein Datensatz


def measure_inference_time(model, dataset, device="cuda", num_samples=50):
    """
    Misst die durchschnittliche Inferenzzeit (ms/Frame) für 'num_samples' Frames.
    Gibt (avg_time_in_ms, std_time_in_ms) zurück.
    """
    model.eval()  # wichtig: eval-Modus
    model.to(device)

    # Falls du einen Warm-Up brauchst:
    warmup_runs = 5

    times = []

    with torch.no_grad():
        # Warm-Up
        for i in range(warmup_runs):
            _ = model_forward_one_sample(model, dataset[i % len(dataset)], device)

        # Echte Messung
        for i in range(num_samples):
            start_t = time.perf_counter()
            _ = model_forward_one_sample(model, dataset[i % len(dataset)], device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            end_t = time.perf_counter()

            times.append((end_t - start_t) * 1000.0)  # ms

    avg_t = sum(times) / len(times)
    var_t = sum((t - avg_t)**2 for t in times) / len(times)
    std_t = var_t**0.5
    return avg_t, std_t


def model_forward_one_sample(model, sample, device):
    points_tensor, _ = sample
    points_tensor = points_tensor.unsqueeze(0).to(device)
    logits = model(points_tensor)
    # Optionales Interpolate falls nötig:
    # H, W = sample[1].shape
    # if logits.size()[2:] != (H, W):
    #     logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Wo soll die Plot-Datei gespeichert werden?
    output_dir = "../../output/evaluation_output"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = "inference_time_comparison.png"
    save_path = os.path.join(output_dir, plot_filename)

    model_names = ["regnet_y_800_mf", "resnet18", "resnet34", "sn0", "sn5"]
    model_paths = {
        "regnet_y_800_mf": "../../output/regnet_y_800_mf/run_030_15-01-2025/model_regnet_y_800_mf_epoch_60.pth",
        "resnet18": "../../output/resnet18/run_034_15-01-2025/model_resnet18_epoch_60.pth",
        "resnet34": "../../output/resnet34/run_025_16-01-2025/model_resnet34_epoch_60.pth",
        "sn0": "../../output/sn0/run_012_16-01-2025/model_sn0_epoch_60.pth",
        "sn5": "../../output/sn5/run_029_16-01-2025/model_sn5_epoch_60.pth",
    }
    num_classes = 18

    dataset = LiDARDataset(
        dataset_dir="/workspace/dataset",
        scenes=["0004"],
        sensor="left",
        apply_transform=False,
        apply_data_augmentation=False,
    )

    avg_times = []
    std_times = []

    for m_name in model_names:
        print(f"\nLade Modell {m_name} ...")
        model = get_model(m_name, num_classes=num_classes)
        model.load_state_dict(torch.load(model_paths[m_name], map_location=device))

        print("Messe Inferenzzeit ...")
        avg_t, std_t = measure_inference_time(
            model, dataset, device=device, num_samples=50
        )

        avg_times.append(avg_t)
        std_times.append(std_t)
        print(f"{m_name}: {avg_t:.2f} ms ± {std_t:.2f} ms")

    # Plot
    plt.figure(figsize=(8, 5))
    x_positions = range(len(model_names))
    plt.bar(x_positions, avg_times, yerr=std_times, capsize=5, alpha=0.7)
    plt.xticks(x_positions, model_names, rotation=45)
    plt.ylabel("Durchschn. Inferenzzeit (ms/Frame)")
    plt.title("Vergleich verschiedener Modelle")

    plt.tight_layout()
    plt.savefig(save_path)  # <-- Speichert das Diagramm
    # plt.show()  # Optional; in einem Container meist nicht nötig

    print(f"Plot wurde gespeichert unter: {save_path}")


if __name__ == "__main__":
    main()
