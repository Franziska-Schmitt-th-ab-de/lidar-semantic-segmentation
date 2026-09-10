import cv2
import os
import argparse
import re


def main():
    parser = argparse.ArgumentParser(description="Erstellung eines Videos aus Bildern")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="output_images",
        help="Verzeichnis mit den Bildern",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="output_video.avi",
        help="Name der Ausgabedatei für das Video",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Bilder pro Sekunde im Video"
    )
    parser.add_argument(
        "--sensor",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Sensor auswählen (left oder right)",
    )
    args = parser.parse_args()

    images = []
    pattern = re.compile(r"labels_image_(\d+)_" + re.escape(args.sensor) + r"\.png")
    for filename in os.listdir(args.images_dir):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            images.append((index, os.path.join(args.images_dir, filename)))

    if not images:
        print("Keine passenden Bilder gefunden.")
        return

    # Bilder nach Index sortieren
    images.sort(key=lambda x: x[0])

    # Lesen des ersten Bildes, um die Größe zu ermitteln
    frame = cv2.imread(images[0][1])
    height, width, layers = frame.shape

    # VideoWriter initialisieren
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))

    for index, image_path in images:
        video.write(cv2.imread(image_path))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video wurde gespeichert als {args.output_video}")


if __name__ == "__main__":
    main()
