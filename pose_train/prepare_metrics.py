import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .dataset import MAVPoseDataset
from .model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet


def train_and_cache(
    data_root: Path,
    cache_dir: Path,
    model_type: str = 'simple',
    backbone: str = "small",
    use_imagenet_norm: bool = False,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)

    dataset = MAVPoseDataset(
        data_root=str(data_root),
        cams=["cam0"],
        img_size=(240, 320),
        use_imagenet_norm=use_imagenet_norm,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    in_ch = 3 if use_imagenet_norm else 1
    # choose model by type
    if model_type == 'simple':
        model = SimplePoseNet(in_channels=in_ch, backbone=backbone, pretrained=False).to(device_t)
    elif model_type == 'advanced':
        model = AdvancedPoseNet(in_channels=in_ch, backbone=backbone, pretrained=False, use_attention=True).to(device_t)
    elif model_type == 'multi_agent':
        model = MultiAgentPoseNet(in_channels=in_ch, backbone=backbone, pretrained=False, max_agents=3, use_attention=True).to(device_t)
    else:
        model = SimplePoseNet(in_channels=in_ch, backbone=backbone, pretrained=False).to(device_t)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, poses in loader:
            imgs = imgs.to(device_t)
            poses = poses.to(device_t)

            optimizer.zero_grad()
            preds = model(imgs)
            # handle multi-agent output shape
            if model_type == 'multi_agent':
                # preds is (B, max_agents, 7), take first agent for training target
                if isinstance(preds, (tuple, list)):
                    pred_poses, pred_presence = preds
                    pred_first = pred_poses[:, 0, :]
                else:
                    # fallback
                    pred_first = preds[:, 0, :]
                loss_pose = criterion(pred_first, poses)
                # presence supervision: encourage first agent presence
                presence_target = torch.ones(pred_presence.shape[0], pred_presence.shape[1], device=device_t)
                bce = nn.BCEWithLogitsLoss()
                loss_presence = bce(pred_presence, presence_target)
                loss = loss_pose + 0.1 * loss_presence
            else:
                loss = criterion(preds, poses)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    # save model
    model_path = cache_dir / "best_model.pt"
    torch.save({"model_state": model.state_dict()}, model_path)

    # Evaluate on the same dataset for a quick cached metric
    model.eval()
    preds_list = []
    gts_list = []
    with torch.no_grad():
        for imgs, poses in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            imgs = imgs.to(device_t)
            preds = model(imgs).cpu().numpy()
            preds_list.append(preds)
            gts_list.append(poses.numpy())

    pred_arr = np.concatenate(preds_list, axis=0)
    gt_arr = np.concatenate(gts_list, axis=0)

    # Compute simple ATE and RPE similar to eval.py utilities
    def compute_ate(gt, pred):
        err = gt[:, :3] - pred[:, :3]
        return float(np.sqrt((err ** 2).sum(axis=1)).mean())

    def compute_rpe(gt, pred, delta=1):
        errs = []
        for i in range(len(gt) - delta):
            gt_rel = gt[i + delta, :3] - gt[i, :3]
            pr_rel = pred[i + delta, :3] - pred[i, :3]
            errs.append(np.linalg.norm(gt_rel - pr_rel))
        return float(np.mean(errs)) if errs else 0.0

    ate = compute_ate(gt_arr, pred_arr)
    rpe = compute_rpe(gt_arr, pred_arr, delta=1)

    metrics = {
        "backbone": backbone,
        "use_imagenet_norm": use_imagenet_norm,
        "epochs": epochs,
        "batch_size": batch_size,
        "ate_m": ate,
        "rpe_m": rpe,
        "num_samples": int(len(gt_arr)),
    }

    with open(cache_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trajectory plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(gt_arr[:, 0], gt_arr[:, 1], label="GT")
    ax.plot(pred_arr[:, 0], pred_arr[:, 1], label="Pred")
    ax.set_title("XY Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cache_dir / "trajectory.png")
    plt.close(fig)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {cache_dir / 'metrics.json'}")
    print(f"Saved plot to: {cache_dir / 'trajectory.png'}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root  # expects cam0/, state_groundtruth_estimate0/ under this
    base_cache = project_root / "work_ui_cache"
    # Define multiple configurations to compare
    configs = [
        {"name": "small_gray", "backbone": "small", "use_imagenet_norm": False},
        {"name": "resnet18_gray", "backbone": "resnet18", "use_imagenet_norm": False},
        {"name": "resnet18_rgb_norm", "backbone": "resnet18", "use_imagenet_norm": True},
    ]
    # Use CPU by default to be broadly compatible; set modest epochs for quick prep
    default_epochs = int(os.environ.get("UI_PREP_EPOCHS", "5"))
    for cfg in configs:
        out_dir = base_cache / cfg["name"]
        train_and_cache(
            data_root=data_root,
            cache_dir=out_dir,
            backbone=cfg["backbone"],
            use_imagenet_norm=cfg["use_imagenet_norm"],
            epochs=default_epochs,
            batch_size=32,
            lr=1e-3,
            device="cpu",
        )


