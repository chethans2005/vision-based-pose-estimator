"""
Advanced training script with comprehensive features for multi-agent pose estimation.
Supports data augmentation, advanced models, and comprehensive evaluation.
"""
import os
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

from pose_train.dataset import MAVPoseDataset
from pose_train.model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet
from pose_train.train import pose_loss, multi_agent_pose_loss
from pose_train.evaluation import evaluate_model_performance, create_evaluation_report
from pose_train.visualization import create_evaluation_dashboard


class AdvancedTrainer:
    """
    Advanced trainer with comprehensive features.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'translation_rmse': [],
            'rotation_rmse': []
        }
        
    def setup_data(self):
        """Setup data loaders with augmentation."""
        print("Setting up data loaders...")
        
        # Create dataset with augmentation
        dataset = MAVPoseDataset(
            self.args.data_root, 
            cams=self.args.cams,
            augment=self.args.augment,
            use_imagenet_norm=self.args.use_imagenet_norm,
            img_size=self.args.img_size
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Split dataset
        val_size = max(1, int(len(dataset) * self.args.val_frac))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    def setup_model(self):
        """Setup model and optimizer."""
        print(f"Setting up {self.args.model_type} model...")
        
        # Determine input channels
        in_channels = 3 if self.args.use_imagenet_norm else len(self.args.cams)
        
        # Create model
        if self.args.model_type == 'simple':
            self.model = SimplePoseNet(
                in_channels=in_channels,
                backbone=self.args.backbone,
                pretrained=self.args.pretrained
            )
            self.is_multi_agent = False
            
        elif self.args.model_type == 'advanced':
            self.model = AdvancedPoseNet(
                in_channels=in_channels,
                backbone=self.args.backbone,
                pretrained=self.args.pretrained,
                use_attention=self.args.use_attention
            )
            self.is_multi_agent = False
            
        elif self.args.model_type == 'multi_agent':
            self.model = MultiAgentPoseNet(
                in_channels=in_channels,
                backbone=self.args.backbone,
                pretrained=self.args.pretrained,
                max_agents=self.args.max_agents,
                use_attention=self.args.use_attention
            )
            self.is_multi_agent = True
            
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        
        # Setup scheduler
        if self.args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs
            )
        elif self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_trans_loss = 0.0
        total_rot_loss = 0.0
        
        for batch_idx, (imgs, poses) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            poses = poses.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.is_multi_agent:
                # Multi-agent training
                batch_size = poses.shape[0]
                max_agents = self.model.max_agents
                
                # Create dummy multi-agent targets
                target_poses = torch.zeros(batch_size, max_agents, 7, device=self.device)
                target_presence = torch.zeros(batch_size, max_agents, device=self.device)
                target_poses[:, 0] = poses
                target_presence[:, 0] = 1.0
                
                pred_poses, pred_presence = self.model(imgs)
                loss, trans_loss, rot_loss, presence_loss = multi_agent_pose_loss(
                    pred_poses, pred_presence, target_poses, target_presence
                )
            else:
                # Single-agent training
                pred_poses = self.model(imgs)
                loss, trans_loss, rot_loss = pose_loss(pred_poses, poses)
            
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item() * imgs.shape[0]
            total_trans_loss += trans_loss * imgs.shape[0]
            total_rot_loss += rot_loss * imgs.shape[0]
            
            if batch_idx % self.args.log_interval == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.6f}, '
                      f'Trans: {trans_loss:.6f}, '
                      f'Rot: {rot_loss:.6f}')
        
        return (total_loss / len(self.train_loader.dataset),
                total_trans_loss / len(self.train_loader.dataset),
                total_rot_loss / len(self.train_loader.dataset))
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_trans_loss = 0.0
        total_rot_loss = 0.0
        
        with torch.no_grad():
            for imgs, poses in self.val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                poses = poses.to(self.device, non_blocking=True)
                
                if self.is_multi_agent:
                    # Multi-agent validation
                    batch_size = poses.shape[0]
                    max_agents = self.model.max_agents
                    
                    target_poses = torch.zeros(batch_size, max_agents, 7, device=self.device)
                    target_presence = torch.zeros(batch_size, max_agents, device=self.device)
                    target_poses[:, 0] = poses
                    target_presence[:, 0] = 1.0
                    
                    pred_poses, pred_presence = self.model(imgs)
                    loss, trans_loss, rot_loss, presence_loss = multi_agent_pose_loss(
                        pred_poses, pred_presence, target_poses, target_presence
                    )
                else:
                    # Single-agent validation
                    pred_poses = self.model(imgs)
                    loss, trans_loss, rot_loss = pose_loss(pred_poses, poses)
                
                total_loss += loss.item() * imgs.shape[0]
                total_trans_loss += trans_loss * imgs.shape[0]
                total_rot_loss += rot_loss * imgs.shape[0]
        
        return (total_loss / len(self.val_loader.dataset),
                total_trans_loss / len(self.val_loader.dataset),
                total_rot_loss / len(self.val_loader.dataset))
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'args': self.args,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.work_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.work_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss, train_trans_loss, train_rot_loss = self.train_epoch()
            
            # Validation
            val_loss, val_trans_loss, val_rot_loss = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['translation_rmse'].append(val_trans_loss)
            self.training_history['rotation_rmse'].append(val_rot_loss)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch}/{self.args.epochs} - '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'Time: {epoch_time:.1f}s')
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.args.early_stopping > 0:
                if epoch > self.args.early_stopping:
                    recent_losses = self.training_history['val_loss'][-self.args.early_stopping:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint(self.args.epochs, False)
        
        # Generate evaluation report
        self.evaluate_and_report()
    
    def evaluate_and_report(self):
        """Evaluate model and generate comprehensive report."""
        print("Generating evaluation report...")
        
        # Load best model
        best_model_path = os.path.join(self.args.work_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
        
        # Evaluate on validation set
        self.model.eval()
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for imgs, poses in self.val_loader:
                imgs = imgs.to(self.device)
                poses = poses.to(self.device)
                
                if self.is_multi_agent:
                    pred_poses, pred_presence = self.model(imgs)
                    # For evaluation, we'll use the first agent's pose
                    preds = pred_poses[:, 0].cpu().numpy()
                else:
                    preds = self.model(imgs).cpu().numpy()
                
                all_preds.append(preds)
                all_gts.append(poses.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_gts = np.vstack(all_gts)
        
        # Compute metrics
        metrics = evaluate_model_performance(all_gts, all_preds)
        
        # Create evaluation report
        report = create_evaluation_report(metrics, f"{self.args.model_type}_{self.args.backbone}")
        report_path = os.path.join(self.args.work_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {report_path}")
        
        # Create visualizations
        try:
            create_evaluation_dashboard(
                all_gts, all_preds,
                save_dir=os.path.join(self.args.work_dir, 'evaluation_plots')
            )
            print("Evaluation plots saved to evaluation_plots/")
        except Exception as e:
            print(f"Could not generate evaluation plots: {e}")
        
        # Plot training history
        self.plot_training_history()
        
        # Save metrics
        metrics_path = os.path.join(self.args.work_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Translation RMSE
        axes[0, 1].plot(self.training_history['translation_rmse'], label='Translation RMSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE (m)')
        axes[0, 1].set_title('Translation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Rotation RMSE
        axes[1, 0].plot(self.training_history['rotation_rmse'], label='Rotation RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE (deg)')
        axes[1, 0].set_title('Rotation RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if hasattr(self, 'scheduler') and self.scheduler:
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lr_history, label='Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.work_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced Multi-Agent Pose Estimation Training')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='.', help='Path to dataset root')
    parser.add_argument('--cams', nargs='+', default=['cam0'], help='Cameras to use')
    parser.add_argument('--img-size', type=int, nargs=2, default=[240, 320], help='Image size (height, width)')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='advanced', 
                       choices=['simple', 'advanced', 'multi_agent'], help='Model type')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                       choices=['small', 'resnet18', 'resnet50'], help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanisms')
    parser.add_argument('--max-agents', type=int, default=5, help='Maximum number of agents')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'cosine', 'step'], help='Learning rate scheduler')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--use-imagenet-norm', action='store_true', help='Use ImageNet normalization')
    
    # Logging and saving
    parser.add_argument('--work-dir', type=str, default='work_advanced', help='Work directory')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=5, help='Save interval')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create trainer
    trainer = AdvancedTrainer(args)
    
    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
