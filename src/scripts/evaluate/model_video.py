import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dataloader import LiDARDataset
from utils.training_utils import get_run_directory
from models import get_model
from utils.colormap import create_custom_class_colormap


def find_latest_model(base_dir, model_name):
    """
    Findet den neuesten Run für das angegebene Modell und gibt den Pfad der neuesten Modelldatei zurück.
    """
    model_dir = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    runs = sorted(
        [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("run_")
        ],
        key=lambda x: os.path.getmtime(x),
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No runs found in '{model_dir}'.")

    latest_run = runs[0]
    model_files = sorted(
        [
            os.path.join(latest_run, f)
            for f in os.listdir(latest_run)
            if f.endswith(".pth")
        ],
        key=lambda x: os.path.getmtime(x),
        reverse=True,
    )
    if not model_files:
        raise FileNotFoundError(f"No model files found in '{latest_run}'.")

    return model_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video with top=GT labels, bottom=model predictions."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/workspace/dataset",
        help="Path to the LiDAR dataset root directory.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="0003",
        help='One or more scene IDs, separated by underscores (e.g. "0002_0003").',
    )
    parser.add_argument(
        "--sensor",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Which sensor to visualize (left or right).",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start index of frames to visualize."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=500,
        help="End index (inclusive) of frames to visualize.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sn5",
        help="Model architecture name (must match your get_model() factory).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=18,
        help="Number of semantic classes (same as training).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the saved .pth file (state_dict) from training. If not provided, the latest run is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../output/model_videos",
        help="Directory to save the output video.",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for the output video."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the latest model if no path is provided
    if not args.model_path:
        args.model_path = find_latest_model("../../output", args.model_name)
        print(f"Using latest model: {args.model_path}")

    # Prepare the dataset
    scenes_list = args.scenes.split("_")
    dataset = LiDARDataset(
        dataset_dir=args.dataset_dir,
        scenes=scenes_list,
        sensor=args.sensor,
        apply_transform=False,
        apply_data_augmentation=False,
    )

    # Clamp end_index if it exceeds dataset length
    if args.end_index >= len(dataset):
        args.end_index = len(dataset) - 1

    # Load the model
    model = get_model(args.model_name, num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Generate output file name
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = f"{args.model_name}_{os.path.basename(args.model_path).replace('.pth', '')}_{args.scenes}.mp4"
    output_video_path = os.path.join(args.output_dir, video_name)

    sample_points, sample_labels = dataset[args.start_index]
    H, W = sample_labels.shape
    frame_width = W
    frame_height = 2 * H  # top=GT labels, bottom=predicted labels

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, args.fps, (frame_width, frame_height)
    )

    # Custom colormap for classes
    custom_colormap = create_custom_class_colormap()

    print(
        f"Generating video '{output_video_path}' from frame indices [{args.start_index}, {args.end_index}]..."
    )
    for idx in tqdm(range(args.start_index, args.end_index + 1)):
        points_tensor, labels_tensor = dataset[idx]
        gt_np = labels_tensor.numpy().astype(np.uint8)
        gt_color = cv2.applyColorMap(gt_np, custom_colormap)

        points_tensor = points_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(points_tensor)
            if logits.size()[2:] != (H, W):
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=False
                )
            pred = torch.argmax(logits, dim=1).squeeze(0)

        pred_np = pred.cpu().numpy().astype(np.uint8)
        pred_color = cv2.applyColorMap(pred_np, custom_colormap)

        combined_frame = np.vstack((gt_color, pred_color))
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"Video saved as {output_video_path}.")


if __name__ == "__main__":
    main()
