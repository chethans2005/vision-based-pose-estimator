import os
import time
import argparse
import math

import torch
from torch.utils.data import DataLoader, random_split

from pose_train.dataset import MAVPoseDataset
from pose_train.model import SimplePoseNet
from pose_train.train import pose_loss


def rotation_error_deg(pred_q, gt_q):
    # pred_q, gt_q: [N,4] in w,x,y,z order
    p = pred_q / (pred_q.norm(dim=-1, keepdim=True) + 1e-12)
    g = gt_q / (gt_q.norm(dim=-1, keepdim=True) + 1e-12)
    dot = (p * g).sum(dim=-1).abs().clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    return (angle * 180.0 / math.pi).mean().item()


def translation_rmse(pred_t, gt_t):
    err = (pred_t - gt_t).norm(dim=-1)
    return torch.sqrt((err ** 2).mean()).item()


def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    for imgs, poses in loader:
        imgs = imgs.to(device)
        poses = poses.to(device)
        optim.zero_grad()
        out = model(imgs)
        loss, _, _ = pose_loss(out, poses)
        loss.backward()
        optim.step()
        total_loss += loss.item() * imgs.shape[0]
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, poses in loader:
            imgs = imgs.to(device)
            out = model(imgs).cpu()
            preds.append(out)
            gts.append(poses)
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    t_rmse = translation_rmse(preds[:, :3], gts[:, :3])
    r_deg = rotation_error_deg(preds[:, 3:], gts[:, 3:])
    return t_rmse, r_deg


def run_experiment(data_root, cams, backbone, pretrained, epochs, batch_size, lr, work_dir, augment=False, use_imagenet_norm=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = MAVPoseDataset(data_root, cams=cams, augment=augment, use_imagenet_norm=use_imagenet_norm)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # if dataset uses ImageNet normalization, inputs will be 3-channel
    in_ch = 3 if use_imagenet_norm else len(cams)
    model = SimplePoseNet(in_channels=in_ch, backbone=backbone, pretrained=pretrained).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(work_dir, exist_ok=True)
    best_val = float('inf')
    # Standard training path
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optim, device)
        val_t, val_r = evaluate(model, val_loader, device)
        print(f"{backbone} pretrained={pretrained} Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_t_rmse={val_t:.4f} val_r_deg={val_r:.3f} time={time.time()-t0:.1f}s")
        # save model
        torch.save({'model_state': model.state_dict(), 'epoch': epoch}, os.path.join(work_dir, f"{backbone}_pre{int(pretrained)}_epoch{epoch}.pt"))

    # final metrics
    val_t, val_r = evaluate(model, val_loader, device)
    return {'backbone': backbone, 'pretrained': pretrained, 'val_t_rmse': val_t, 'val_r_deg': val_r}


def fast_eval(data_root, cams, backbone, pretrained, samples=50, use_imagenet_norm=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = MAVPoseDataset(data_root, cams=cams, augment=False, use_imagenet_norm=use_imagenet_norm)
    # take a small subset
    n = min(len(ds), samples)
    idx = list(range(n))
    in_ch = 3 if use_imagenet_norm else len(cams)
    model = SimplePoseNet(in_channels=in_ch, backbone=backbone, pretrained=pretrained).to(device)
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for i in idx:
            img, pose = ds[i]
            x = img.unsqueeze(0).to(device)
            out = model(x).cpu()
            preds.append(out)
            gts.append(pose.unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    t_rmse = translation_rmse(preds[:, :3], gts[:, :3])
    r_deg = rotation_error_deg(preds[:, 3:], gts[:, 3:])
    return {'backbone': backbone, 'pretrained': pretrained, 'val_t_rmse': t_rmse, 'val_r_deg': r_deg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='.', help='dataset root')
    p.add_argument('--cams', nargs='+', default=['cam0'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--work-dir', type=str, default='work_compare')
    p.add_argument('--fast', action='store_true', help='fast inference-only mode')
    p.add_argument('--fast-samples', type=int, default=50, help='number of samples for fast mode')
    p.add_argument('--augment', action='store_true', help='apply simple augmentations during training')
    p.add_argument('--use-imagenet-norm', action='store_true', help='apply ImageNet normalization to inputs')
    args = p.parse_args()

    configs = [
        ('small', False),
        ('resnet18', False),
        ('resnet18', True),
    ]

    results = []
    for backbone, pre in configs:
        print('Running', backbone, 'pretrained=', pre)
        if args.fast:
            res = fast_eval(args.data_root, args.cams, backbone, pre, samples=args.fast_samples, use_imagenet_norm=args.use_imagenet_norm)
        else:
            res = run_experiment(args.data_root, args.cams, backbone, pre, args.epochs, args.batch_size, args.lr, args.work_dir, augment=args.augment, use_imagenet_norm=args.use_imagenet_norm)
        results.append(res)

    print('\nComparison results:')
    for r in results:
        print(r)

    # save CSV and simple bar plot
    import csv
    import matplotlib.pyplot as plt
    csv_path = os.path.join(args.work_dir, 'compare_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['backbone', 'pretrained', 'val_t_rmse', 'val_r_deg'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # bar plots
    names = [f"{r['backbone']}_pre{int(r['pretrained'])}" for r in results]
    tvals = [r['val_t_rmse'] for r in results]
    rvals = [r['val_r_deg'] for r in results]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.bar(names, tvals)
    plt.title('Translation RMSE (m)')
    plt.ylabel('RMSE (m)')
    plt.subplot(1,2,2)
    plt.bar(names, rvals)
    plt.title('Rotation error (deg)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, 'compare_plots.png'))


if __name__ == '__main__':
    main()
