"""
Generate predicted poses CSV from trained model for evaluation.
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from pose_train.model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet
from pose_train.dataset import MAVPoseDataset


def generate_predictions(args):
    """Load model, run inference, save predictions to CSV."""
    
    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    dataset = MAVPoseDataset(
        data_root=args.data_root,
        cams=args.cams,
        use_imagenet_norm=args.use_imagenet_norm,
        img_size=(240, 320)
    )
    
    # Split into train/val (same as training)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"Creating {args.model_type} model with backbone {args.backbone}...")
    in_ch = 3 if args.use_imagenet_norm else 1
    
    if args.model_type == 'simple':
        model = SimplePoseNet(in_channels=in_ch, backbone=args.backbone, pretrained=False)
    elif args.model_type == 'advanced':
        model = AdvancedPoseNet(in_channels=in_ch, backbone=args.backbone, pretrained=False, use_attention=args.use_attention)
    elif args.model_type == 'multi_agent':
        model = MultiAgentPoseNet(in_channels=in_ch, backbone=args.backbone, pretrained=False, 
                                 max_agents=args.max_agents, use_attention=args.use_attention)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Run inference
    print("Running inference on validation set...")
    all_gt_poses = []
    all_pred_poses = []
    all_timestamps = []
    
    with torch.no_grad():
        for imgs, gt_poses in tqdm(val_loader, desc="Inference"):
            imgs = imgs.to(device)
            
            # Forward pass
            if args.model_type == 'multi_agent':
                pred_poses, presence = model(imgs)
                # For single-agent eval, take first agent
                pred_poses = pred_poses[:, 0, :]  # [batch, 7]
            else:
                pred_poses = model(imgs)
            
            all_gt_poses.append(gt_poses.cpu().numpy())
            all_pred_poses.append(pred_poses.cpu().numpy())
    
    # Concatenate all batches
    all_gt_poses = np.concatenate(all_gt_poses, axis=0)
    all_pred_poses = np.concatenate(all_pred_poses, axis=0)
    
    print(f"Generated {len(all_gt_poses)} predictions")
    
    # Create timestamps (use sequential indices as pseudo-timestamps)
    timestamps = np.arange(len(all_gt_poses)) * 1000000  # Increment by 1ms in nanoseconds
    
    # Save ground truth CSV
    gt_df = pd.DataFrame({
        '#timestamp [ns]': timestamps,
        'p_RS_R_x [m]': all_gt_poses[:, 0],
        'p_RS_R_y [m]': all_gt_poses[:, 1],
        'p_RS_R_z [m]': all_gt_poses[:, 2],
        'q_RS_w []': all_gt_poses[:, 3],
        'q_RS_x []': all_gt_poses[:, 4],
        'q_RS_y []': all_gt_poses[:, 5],
        'q_RS_z []': all_gt_poses[:, 6],
    })
    
    # Save predicted CSV
    pred_df = pd.DataFrame({
        '#timestamp [ns]': timestamps,
        'p_RS_R_x [m]': all_pred_poses[:, 0],
        'p_RS_R_y [m]': all_pred_poses[:, 1],
        'p_RS_R_z [m]': all_pred_poses[:, 2],
        'q_RS_w []': all_pred_poses[:, 3],
        'q_RS_x []': all_pred_poses[:, 4],
        'q_RS_y []': all_pred_poses[:, 5],
        'q_RS_z []': all_pred_poses[:, 6],
    })
    
    # Save to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, 'ground_truth.csv')
    pred_path = os.path.join(args.output_dir, 'predicted.csv')
    
    gt_df.to_csv(gt_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Ground truth: {gt_path}")
    print(f"  Predictions:  {pred_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python pose_train/eval.py --gt {gt_path} --pred {pred_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions from trained model')
    parser.add_argument('--data-root', type=str, default='.',
                       help='Dataset root directory')
    parser.add_argument('--cams', nargs='+', default=['cam0'],
                       help='Camera folders to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['simple', 'advanced', 'multi_agent'],
                       help='Model type')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['small', 'resnet18', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--use-imagenet-norm', action='store_true',
                       help='Use ImageNet normalization')
    parser.add_argument('--use-attention', action='store_true',
                       help='Use attention modules (for advanced/multi-agent models)')
    parser.add_argument('--max-agents', type=int, default=3,
                       help='Max agents for multi-agent model')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save output CSVs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_predictions(args)
