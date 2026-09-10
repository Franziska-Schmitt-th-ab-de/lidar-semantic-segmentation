import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentationLoss(nn.Module):
    def __init__(self):
        super(SemanticSegmentationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, predicted_logits, target, num_classes=18):
        # predicted_logits: [B, num_classes, H, W]
        # target: [B, H, W]

        # Passe die Größe an: [B, H, W, num_classes] -> [B*H*W, num_classes]
        predicted_logits_flat = (
            predicted_logits.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        )
        # Passe die Zielgröße an: [B, H, W] -> [B*H*W]
        target_flat = target.view(-1)

        # Berechne den Cross-Entropy Verlust
        loss = self.criterion(predicted_logits_flat, target_flat)
        return loss
