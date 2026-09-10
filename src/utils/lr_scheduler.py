import torch


# gibt typabhängigen Lernratenplaner zurück
# Typen StepLR, ExponentialLR, ReduceLROnPLateau
def get_scheduler(optimizer, scheduler_type="ReduceLROnPlateau", **kwargs):

    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_type == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95)
        )
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 3),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
