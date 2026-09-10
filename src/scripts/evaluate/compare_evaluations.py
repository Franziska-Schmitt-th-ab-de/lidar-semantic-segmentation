import os
import sys
from plot_single_training import plot_normal_case
from plot_crossval_training import plot_cross_val_case
from plot_confusion_iou import plot_confusion_and_iou, plot_iou_per_class_for_all_runs


def main():
    #! Wichtige Notiz bzw.:
    # TODO Aktuell werden alle Dateien, welche in diesem Skript erstellt werden im evaluation_output Ordner gespeichert, allerdings kann das sehr verwirrend sein, da 1. nicht alle der Dateien überhaupt aus der --cross_val Evaluierung stammen, und es wird einfach alles in den Ordner geworfen :
    # Gehe davon aus, dass dieses Skript auf derselben Ebene wie die anderen liegt
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # Beispiel: ../output
    models_dir = os.path.join(os.path.dirname(src_dir), "../output")

    # Prüfen, ob --cross_val in den Argumenten enthalten ist
    cross_val = "--cross_val" in sys.argv

    if cross_val:
        print("Starte Auswertung mit Cross-Validation...")
        plot_cross_val_case(models_dir)
    else:
        print("Starte Auswertung ohne Cross-Validation...")
        plot_normal_case(models_dir)

    # Analyse und Visualisierung der Konfusionsmatrix und IoU
    print("Starte Auswertung von Konfusionsmatrix und IoU...")
    plot_confusion_and_iou(models_dir)
    
    #! This following function needs to be run only once (would be better to save the plots directly after the training (i know))
    #plot_iou_per_class_for_all_runs(models_dir)


if __name__ == "__main__":
    main()
