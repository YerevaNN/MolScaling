import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score


def _set_seed(seed: int = 42):
    """
    Sets deterministic random seeds across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mae_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, weights: torch.Tensor = None):
    """
    Computes masked Mean Absolute Error (MAE) with optional sample weights.
    """
    if weights is not None:
        denom = (masks * weights).sum()
        if denom.item() <= 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True), 0
        loss = (torch.abs(preds - targets) * masks * weights).sum() / denom
    else:
        denom = masks.sum()
        if denom.item() <= 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True), 0
        loss = (torch.abs(preds - targets) * masks).sum() / denom
    return loss, denom.item()


def masked_mse_loss(preds_dict: dict[str, torch.Tensor], targets: torch.Tensor, masks: torch.Tensor, tasks: list[str]):
    """
    Computes multi-task masked Mean Squared Error (MSE) loss.
    """
    task_losses = []
    total_valid = 0
    for idx, t in enumerate(tasks):
        m = masks[:, idx].bool()
        if m.sum().item() > 0:
            p = preds_dict[t][m]
            y = targets[:, idx][m]
            task_losses.append(torch.mean((p - y) ** 2))
            total_valid += 1
    if len(task_losses) == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True), 0
    return torch.mean(torch.stack(task_losses)), total_valid


@torch.no_grad()
def evaluate_regression(model, loader, device) -> dict[str, float]:
    """
    Evaluates single-target continuous regression (e.g. PXR pEC50),
    returning RAE (Relative Absolute Error) and R2.
    """
    model.eval()
    all_preds = []
    all_y = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            preds = model(input_ids=input_ids, attention_mask=attention_mask)

        m = masks.bool()
        if m.sum().item() > 0:
            all_preds.append(preds[m].detach().cpu())
            all_y.append(targets[m].detach().cpu())

    if len(all_preds) == 0:
        return {"rae": float("nan"), "r2": float("nan"), "n": 0}

    p = torch.cat(all_preds).numpy()
    y = torch.cat(all_y).numpy()

    y_mean = np.mean(y)
    denom_sum = np.sum(np.abs(y - y_mean))
    numer_sum = np.sum(np.abs(y - p))
    rae = float(numer_sum / denom_sum) if denom_sum > 0 else float("nan")
    r2 = float(r2_score(y, p)) if len(y) >= 2 else float("nan")

    return {"rae": rae, "r2": r2, "n": int(len(y))}


@torch.no_grad()
def evaluate_multitask(model, loader, device, tasks: list[str]) -> dict:
    """
    Evaluates multi-task regression (e.g. Polaris 6 ADME targets),
    returning per-task MSE, per-task R2, and mean R2.
    """
    model.eval()
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    all_preds = {t: [] for t in tasks}
    all_y = {t: [] for t in tasks}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            preds = model(input_ids, attention_mask)

        for t in tasks:
            idx = task_to_idx[t]
            m = masks[:, idx].bool()
            if m.sum().item() == 0:
                continue
            all_preds[t].append(preds[t][m].detach().cpu())
            all_y[t].append(targets[:, idx][m].detach().cpu())

    metrics = {}
    for t in tasks:
        if len(all_preds[t]) == 0:
            metrics[t] = {"mse": float("nan"), "r2": float("nan"), "n": 0}
            continue
        p = torch.cat(all_preds[t]).numpy()
        y = torch.cat(all_y[t]).numpy()
        mse = float(mean_squared_error(y, p))
        r2 = float(r2_score(y, p)) if len(y) >= 2 else float("nan")
        metrics[t] = {"mse": mse, "r2": r2, "n": int(len(y))}

    r2s = [metrics[t]["r2"] for t in tasks if not np.isnan(metrics[t]["r2"])]
    mean_r2 = float(np.mean(r2s)) if len(r2s) else float("nan")
    metrics["_mean_r2"] = mean_r2
    return metrics
