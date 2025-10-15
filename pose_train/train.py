import os
import argparse
import time

import torch
from torch.utils.data import DataLoader, random_split
import csv

try:
    # when running from project root and using package imports
    from pose_train.dataset import MAVPoseDataset
    from pose_train.model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet
except Exception:
    # fallback for direct script execution (same-folder imports)
    from dataset import MAVPoseDataset
    from model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet


def _quat_normalize(q: torch.Tensor):
    # q: (...,4) in order [qw,qx,qy,qz]
    return q / (q.norm(dim=-1, keepdim=True) + 1e-12)


def quat_geodesic_loss(pred_q: torch.Tensor, target_q: torch.Tensor):
    """Compute geodesic loss between quaternions.

    pred_q, target_q: [B,4] in (w,x,y,z) order. Returns scalar tensor (mean squared angle).
    """
    p = _quat_normalize(pred_q)
    t = _quat_normalize(target_q)
    # dot product along last dim
    dot = (p * t).sum(dim=-1)
    # handle double-cover: use absolute value
    dot_abs = dot.abs().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = 2.0 * torch.acos(dot_abs)
    return (angle ** 2).mean()


def pose_loss(pred, target, w_trans=1.0, w_rot=1.0):
    # pred: [B,7] target: [B,7], where quaternion order is [qw,qx,qy,qz]
    trans_pred = pred[:, :3]
    rot_pred = pred[:, 3:]
    trans_t = target[:, :3]
    rot_t = target[:, 3:]
    loss_trans = torch.nn.functional.mse_loss(trans_pred, trans_t)
    loss_rot = quat_geodesic_loss(rot_pred, rot_t)
    loss = w_trans * loss_trans + w_rot * loss_rot
    return loss, loss_trans.item(), loss_rot.item()


def multi_agent_pose_loss(pred_poses, pred_presence, target_poses, target_presence, 
                         w_trans=1.0, w_rot=1.0, w_presence=0.1):
    """
    Loss function for multi-agent pose estimation.
    
    Args:
        pred_poses: [B, max_agents, 7] predicted poses for each agent
        pred_presence: [B, max_agents] predicted presence probabilities
        target_poses: [B, max_agents, 7] target poses for each agent
        target_presence: [B, max_agents] target presence (binary)
    """
    batch_size, max_agents = pred_poses.shape[:2]
    
    # Pose loss (only for present agents)
    total_pose_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0
    valid_agents = 0
    
    for i in range(max_agents):
        # Only compute loss for agents that are present in ground truth
        present_mask = target_presence[:, i] > 0.5
        if present_mask.sum() > 0:
            pred_pose_i = pred_poses[present_mask, i]  # [N, 7]
            target_pose_i = target_poses[present_mask, i]  # [N, 7]
            
            trans_pred = pred_pose_i[:, :3]
            rot_pred = pred_pose_i[:, 3:]
            trans_t = target_pose_i[:, :3]
            rot_t = target_pose_i[:, 3:]
            
            loss_trans = torch.nn.functional.mse_loss(trans_pred, trans_t)
            loss_rot = quat_geodesic_loss(rot_pred, rot_t)
            loss_pose = w_trans * loss_trans + w_rot * loss_rot
            
            total_pose_loss += loss_pose
            total_trans_loss += loss_trans.item()
            total_rot_loss += loss_rot.item()
            valid_agents += 1
    
    if valid_agents > 0:
        total_pose_loss /= valid_agents
        total_trans_loss /= valid_agents
        total_rot_loss /= valid_agents
    
    # Presence loss (binary cross-entropy)
    presence_loss = torch.nn.functional.binary_cross_entropy(pred_presence, target_presence.float())
    
    total_loss = total_pose_loss + w_presence * presence_loss
    
    return total_loss, total_trans_loss, total_rot_loss, presence_loss.item()


def train_epoch(model, loader, optim, device, is_multi_agent=False):
    model.train()
    total_loss = 0.0
    for imgs, poses in loader:
        imgs = imgs.to(device)
        poses = poses.to(device)
        optim.zero_grad()
        
        if is_multi_agent:
            # For multi-agent models, we need to create dummy multi-agent targets
            # In a real scenario, you would have multi-agent ground truth data
            batch_size = poses.shape[0]
            max_agents = model.max_agents
            
            # Create dummy multi-agent targets (single agent in first position)
            target_poses = torch.zeros(batch_size, max_agents, 7, device=device)
            target_presence = torch.zeros(batch_size, max_agents, device=device)
            target_poses[:, 0] = poses  # Put single agent pose in first position
            target_presence[:, 0] = 1.0  # Mark first agent as present
            
            pred_poses, pred_presence = model(imgs)
            loss, _, _, _ = multi_agent_pose_loss(pred_poses, pred_presence, target_poses, target_presence)
        else:
            out = model(imgs)
            loss, _, _ = pose_loss(out, poses)
        
        loss.backward()
        optim.step()
        total_loss += loss.item() * imgs.shape[0]
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, is_multi_agent=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, poses in loader:
            imgs = imgs.to(device)
            poses = poses.to(device)
            
            if is_multi_agent:
                # For multi-agent models, we need to create dummy multi-agent targets
                batch_size = poses.shape[0]
                max_agents = model.max_agents
                
                target_poses = torch.zeros(batch_size, max_agents, 7, device=device)
                target_presence = torch.zeros(batch_size, max_agents, device=device)
                target_poses[:, 0] = poses
                target_presence[:, 0] = 1.0
                
                pred_poses, pred_presence = model(imgs)
                loss, _, _, _ = multi_agent_pose_loss(pred_poses, pred_presence, target_poses, target_presence)
            else:
                out = model(imgs)
                loss, _, _ = pose_loss(out, poses)
            
            total_loss += loss.item() * imgs.shape[0]
    return total_loss / len(loader.dataset)


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MAVPoseDataset(args.data_root, cams=args.cams, augment=args.augment, use_imagenet_norm=args.use_imagenet_norm)

    # split train/val
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model based on type
    if args.model_type == 'simple':
        model = SimplePoseNet(in_channels=len(args.cams), backbone=args.backbone, pretrained=args.pretrained).to(device)
        is_multi_agent = False
    elif args.model_type == 'advanced':
        in_channels = 3 if args.use_imagenet_norm else len(args.cams)
        model = AdvancedPoseNet(in_channels=in_channels, backbone=args.backbone, pretrained=args.pretrained, use_attention=args.use_attention).to(device)
        is_multi_agent = False
    elif args.model_type == 'multi_agent':
        in_channels = 3 if args.use_imagenet_norm else len(args.cams)
        model = MultiAgentPoseNet(in_channels=in_channels, backbone=args.backbone, pretrained=args.pretrained, 
                                 max_agents=args.max_agents, use_attention=args.use_attention).to(device)
        is_multi_agent = True
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    os.makedirs(args.work_dir, exist_ok=True)
    epoch_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optim, device, is_multi_agent)
        val_loss = evaluate(model, val_loader, device, is_multi_agent)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.6f} val_loss: {val_loss:.6f} - time: {time.time()-t0:.1f}s")

        # save checkpoint
        ckpt = os.path.join(args.work_dir, f"ckpt_epoch_{epoch}.pt")
        torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'optimizer': optim.state_dict()}, ckpt)
        # save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.work_dir, "best_ckpt.pt")
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'optimizer': optim.state_dict()}, best_path)

        # record epoch stats
        epoch_records.append({'epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(val_loss)})

    # Write results.csv to work_dir
    results_csv = os.path.join(args.work_dir, 'results.csv')
    try:
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
            writer.writeheader()
            for r in epoch_records:
                writer.writerow(r)
    except Exception as e:
        print('Warning: failed to write results.csv:', e)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='.', help='path to dataset root')
    p.add_argument('--cams', nargs='+', default=['cam0'], help='cameras to use')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--work-dir', type=str, default='work')
    p.add_argument('--val-frac', type=float, default=0.1, help='fraction for validation split')
    p.add_argument('--backbone', type=str, default='small', help="backbone: 'small', 'resnet18', or 'resnet50'")
    p.add_argument('--pretrained', action='store_true', help='use pretrained weights for backbone if available')
    p.add_argument('--model-type', type=str, default='simple', choices=['simple', 'advanced', 'multi_agent'], 
                   help='type of model to use')
    p.add_argument('--use-attention', action='store_true', help='use attention mechanisms in advanced models')
    p.add_argument('--max-agents', type=int, default=5, help='maximum number of agents for multi-agent model')
    p.add_argument('--augment', action='store_true', help='use data augmentation during training')
    p.add_argument('--use-imagenet-norm', action='store_true', help='use ImageNet normalization')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    run(args)
