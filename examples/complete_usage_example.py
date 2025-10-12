#!/usr/bin/env python3
"""
Complete Usage Example for Vision-Based Multi-Agent Pose Estimator

This script demonstrates all the key features of the system:
1. Data loading and preprocessing
2. Model training and evaluation
3. Visualization and analysis
4. Real-time inference simulation
5. Multi-agent formation analysis

Usage:
    python examples/complete_usage_example.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from pose_train.dataset import MAVPoseDataset
from pose_train.model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet
from pose_train.train import pose_loss, multi_agent_pose_loss
from pose_train.evaluation import evaluate_model_performance, create_evaluation_report
from pose_train.visualization import (
    plot_trajectory_2d, plot_trajectory_3d, plot_error_analysis,
    plot_multi_agent_trajectory, plot_formation_analysis,
    create_evaluation_dashboard, plot_model_comparison
)


def setup_environment():
    """Setup the environment and check dependencies."""
    print("Setting up environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Project root: {project_root}")
    
    # Set up plotting
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_explore_data(data_root, cams):
    """Load dataset and explore its structure."""
    print("\n" + "="*50)
    print("1. DATA LOADING AND EXPLORATION")
    print("="*50)
    
    # Load dataset with augmentation
    dataset = MAVPoseDataset(
        data_root=data_root,
        cams=cams,
        augment=True,
        use_imagenet_norm=True,
        img_size=(240, 320)
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image shape: {dataset[0][0].shape}")
    print(f"Pose shape: {dataset[0][1].shape}")
    
    # Show sample data
    img, pose = dataset[0]
    print(f"\nSample pose: {pose.numpy()}")
    print(f"Translation: {pose[:3].numpy()}")
    print(f"Quaternion: {pose[3:].numpy()}")
    
    return dataset


def create_and_compare_models(device):
    """Create different model architectures and compare them."""
    print("\n" + "="*50)
    print("2. MODEL ARCHITECTURE COMPARISON")
    print("="*50)
    
    # Create different model architectures
    models = {}
    
    # Simple model
    models['Simple'] = SimplePoseNet(
        in_channels=3,  # RGB due to ImageNet normalization
        backbone='small',
        pretrained=False
    ).to(device)
    
    # Advanced model
    models['Advanced'] = AdvancedPoseNet(
        in_channels=3,
        backbone='resnet18',
        pretrained=True,
        use_attention=True
    ).to(device)
    
    # Multi-agent model
    models['Multi-Agent'] = MultiAgentPoseNet(
        in_channels=3,
        backbone='resnet18',
        pretrained=True,
        max_agents=5,
        use_attention=True
    ).to(device)
    
    # Compare model sizes
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Comparison:")
    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"{name} Model: {param_count:,} parameters")
    
    return models


def train_models(models, dataset, device, epochs=3):
    """Train all models and return training history."""
    print("\n" + "="*50)
    print("3. MODEL TRAINING")
    print("="*50)
    
    # Split dataset
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training function
    def train_model(model, train_loader, val_loader, epochs, model_name):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for imgs, poses in train_loader:
                imgs = imgs.to(device)
                poses = poses.to(device)
                
                optimizer.zero_grad()
                
                if isinstance(model, MultiAgentPoseNet):
                    # Multi-agent training
                    batch_size = poses.shape[0]
                    target_poses = torch.zeros(batch_size, model.max_agents, 7, device=device)
                    target_presence = torch.zeros(batch_size, model.max_agents, device=device)
                    target_poses[:, 0] = poses
                    target_presence[:, 0] = 1.0
                    
                    pred_poses, pred_presence = model(imgs)
                    loss, _, _, _ = multi_agent_pose_loss(pred_poses, pred_presence, target_poses, target_presence)
                else:
                    # Single-agent training
                    pred_poses = model(imgs)
                    loss, _, _ = pose_loss(pred_poses, poses)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.shape[0]
            
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for imgs, poses in val_loader:
                    imgs = imgs.to(device)
                    poses = poses.to(device)
                    
                    if isinstance(model, MultiAgentPoseNet):
                        batch_size = poses.shape[0]
                        target_poses = torch.zeros(batch_size, model.max_agents, 7, device=device)
                        target_presence = torch.zeros(batch_size, model.max_agents, device=device)
                        target_poses[:, 0] = poses
                        target_presence[:, 0] = 1.0
                        
                        pred_poses, pred_presence = model(imgs)
                        loss, _, _, _ = multi_agent_pose_loss(pred_poses, pred_presence, target_poses, target_presence)
                    else:
                        pred_poses = model(imgs)
                        loss, _, _ = pose_loss(pred_poses, poses)
                    
                    val_loss += loss.item() * imgs.shape[0]
            
            val_loss /= len(val_loader.dataset)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return train_losses, val_losses
    
    # Train all models
    training_history = {}
    for name, model in models.items():
        print(f"\nTraining {name} Model...")
        train_loss, val_loss = train_model(model, train_loader, val_loader, epochs, name)
        training_history[name] = {'train': train_loss, 'val': val_loss}
    
    return training_history, val_loader


def evaluate_models(models, val_loader, device):
    """Evaluate all models and return results."""
    print("\n" + "="*50)
    print("4. MODEL EVALUATION")
    print("="*50)
    
    def evaluate_model(model, val_loader, model_name):
        model.eval()
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for imgs, poses in val_loader:
                imgs = imgs.to(device)
                poses = poses.to(device)
                
                if isinstance(model, MultiAgentPoseNet):
                    agent_poses, agent_presence = model(imgs)
                    preds = agent_poses[:, 0].cpu().numpy()  # Use first agent
                else:
                    preds = model(imgs).cpu().numpy()
                
                all_preds.append(preds)
                all_gts.append(poses.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_gts = np.vstack(all_gts)
        
        # Compute metrics
        metrics = evaluate_model_performance(all_gts, all_preds)
        
        print(f"\n{model_name} Model Evaluation:")
        print(f"Translation RMSE: {metrics['translation_rmse']:.4f} m")
        print(f"Translation MAE: {metrics['translation_mae']:.4f} m")
        print(f"Rotation RMSE: {metrics['rotation_rmse']:.2f}°")
        print(f"Rotation MAE: {metrics['rotation_mae']:.2f}°")
        
        return all_gts, all_preds, metrics
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        gt, pred, metrics = evaluate_model(model, val_loader, name)
        results[name] = {'gt': gt, 'pred': pred, 'metrics': metrics}
    
    return results


def visualize_results(results, training_history):
    """Create comprehensive visualizations."""
    print("\n" + "="*50)
    print("5. VISUALIZATION AND ANALYSIS")
    print("="*50)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for name, history in training_history.items():
        plt.plot(history['train'], label=f'{name} Train', marker='o')
        plt.plot(history['val'], label=f'{name} Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    for name, history in training_history.items():
        plt.plot(history['val'], label=name, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    final_losses = [history['val'][-1] for history in training_history.values()]
    model_names = list(training_history.keys())
    bars = plt.bar(model_names, final_losses, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Final Validation Loss')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                 f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot trajectory comparisons
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = ['red', 'blue', 'magenta']
    
    for i, (name, result) in enumerate(results.items()):
        gt = result['gt']
        pred = result['pred']
        color = colors[i]
        
        # XY trajectory
        axes[0, i].plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
        axes[0, i].plot(pred[:, 0], pred[:, 1], f'{color}--', linewidth=2, label=f'{name} Pred', alpha=0.8)
        axes[0, i].set_xlabel('X (m)')
        axes[0, i].set_ylabel('Y (m)')
        axes[0, i].set_title(f'{name} Model - XY Trajectory')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')
        
        # XZ trajectory
        axes[1, i].plot(gt[:, 0], gt[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
        axes[1, i].plot(pred[:, 0], pred[:, 2], f'{color}--', linewidth=2, label=f'{name} Pred', alpha=0.8)
        axes[1, i].set_xlabel('X (m)')
        axes[1, i].set_ylabel('Z (m)')
        axes[1, i].set_title(f'{name} Model - XZ Trajectory')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axis('equal')
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis for best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['translation_rmse'])
    best_result = results[best_model_name]
    
    print(f"\nPerforming detailed error analysis for {best_model_name} model...")
    plot_error_analysis(best_result['gt'], best_result['pred'], 
                       title=f"{best_model_name} Model - Error Analysis")
    
    # Create comprehensive evaluation dashboard
    print("Creating comprehensive evaluation dashboard...")
    create_evaluation_dashboard(
        best_result['gt'], best_result['pred'],
        save_dir="evaluation_results"
    )
    print("Evaluation dashboard saved to 'evaluation_results/' directory")


def demonstrate_multi_agent_capabilities():
    """Demonstrate multi-agent formation analysis capabilities."""
    print("\n" + "="*50)
    print("6. MULTI-AGENT FORMATION ANALYSIS")
    print("="*50)
    
    # Simulate multi-agent data
    def simulate_multi_agent_data(n_samples=100, n_agents=3):
        """Simulate multi-agent formation data."""
        agent_poses = {}
        agent_presence = {}
        
        # Base trajectory (leader)
        t = np.linspace(0, 4*np.pi, n_samples)
        base_x = t * np.cos(t) * 0.1
        base_y = t * np.sin(t) * 0.1
        base_z = t * 0.05
        
        for agent_id in range(n_agents):
            if agent_id == 0:
                # Leader agent
                x = base_x
                y = base_y
                z = base_z
                presence = np.ones(n_samples)
            else:
                # Follower agents with formation offsets
                offset_x = 2.0 * np.cos(2 * np.pi * agent_id / n_agents)
                offset_y = 2.0 * np.sin(2 * np.pi * agent_id / n_agents)
                offset_z = 0.5 * agent_id
                
                x = base_x + offset_x
                y = base_y + offset_y
                z = base_z + offset_z
                
                # Simulate some missing data
                presence = np.ones(n_samples)
                if agent_id > 1:
                    # Agent 2 is missing for some frames
                    missing_start = n_samples // 3
                    missing_end = 2 * n_samples // 3
                    presence[missing_start:missing_end] = 0.0
            
            # Create pose array [tx, ty, tz, qw, qx, qy, qz]
            poses = np.zeros((n_samples, 7))
            poses[:, 0] = x
            poses[:, 1] = y
            poses[:, 2] = z
            poses[:, 3] = 1.0  # qw
            poses[:, 4:7] = 0.0  # qx, qy, qz
            
            agent_poses[agent_id] = poses
            agent_presence[agent_id] = presence
        
        return agent_poses, agent_presence
    
    # Generate simulated multi-agent data
    sim_poses, sim_presence = simulate_multi_agent_data(n_samples=200, n_agents=4)
    
    print("Simulated Multi-Agent Formation Data:")
    for agent_id in sim_poses.keys():
        presence_rate = np.mean(sim_presence[agent_id])
        print(f"Agent {agent_id}: {len(sim_poses[agent_id])} poses, {presence_rate:.1%} presence")
    
    # Visualize multi-agent trajectories
    plot_multi_agent_trajectory(
        sim_poses, sim_presence,
        title="Simulated Multi-Agent Formation Trajectory"
    )
    
    # Analyze formation patterns
    plot_formation_analysis(
        sim_poses, sim_presence,
        title="Formation Analysis - Inter-Agent Relationships"
    )


def simulate_realtime_inference(model, dataset, device, n_frames=50):
    """Simulate real-time inference capabilities."""
    print("\n" + "="*50)
    print("7. REAL-TIME INFERENCE SIMULATION")
    print("="*50)
    
    model.eval()
    
    poses_history = []
    timestamps = []
    inference_times = []
    
    print(f"Simulating real-time inference for {n_frames} frames...")
    
    with torch.no_grad():
        for i in range(min(n_frames, len(dataset))):
            img, gt_pose = dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            
            start_time = time.time()
            
            if isinstance(model, MultiAgentPoseNet):
                agent_poses, agent_presence = model(img_tensor)
                pred_pose = agent_poses[0, 0].cpu().numpy()  # First agent
            else:
                pred_pose = model(img_tensor).cpu().numpy()[0]
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            poses_history.append({
                'translation': pred_pose[:3],
                'rotation': pred_pose[3:7],
                'quaternion': pred_pose[3:7]
            })
            timestamps.append(time.time())
            
            if i % 10 == 0:
                print(f"Frame {i}: Inference time: {inference_time*1000:.1f}ms")
    
    # Convert to numpy arrays for visualization
    rt_pred_poses = np.array([[pose['translation'][0], pose['translation'][1], pose['translation'][2],
                               pose['quaternion'][0], pose['quaternion'][1], pose['quaternion'][2], pose['quaternion'][3]]
                              for pose in poses_history])
    
    # Get corresponding ground truth
    rt_gt_poses = np.array([dataset[i][1].numpy() for i in range(len(poses_history))])
    
    # Visualize real-time trajectory
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rt_gt_poses[:, 0], rt_gt_poses[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    plt.plot(rt_pred_poses[:, 0], rt_pred_poses[:, 1], 'r--', linewidth=2, label='Real-time Pred', alpha=0.8)
    plt.scatter(rt_gt_poses[0, 0], rt_gt_poses[0, 1], color='green', s=100, marker='o', label='Start')
    plt.scatter(rt_gt_poses[-1, 0], rt_gt_poses[-1, 1], color='green', s=100, marker='s', label='End')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Real-Time Trajectory Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    trans_errors = np.linalg.norm(rt_pred_poses[:, :3] - rt_gt_poses[:, :3], axis=1)
    plt.plot(trans_errors, 'b-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Translation Error (m)')
    plt.title('Real-Time Error Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realtime_inference.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nReal-time Performance:")
    print(f"Average translation error: {np.mean(trans_errors):.4f} m")
    print(f"Max translation error: {np.max(trans_errors):.4f} m")
    print(f"Std translation error: {np.std(trans_errors):.4f} m")
    print(f"Average inference time: {np.mean(inference_times)*1000:.1f} ms")
    print(f"Max inference time: {np.max(inference_times)*1000:.1f} ms")


def create_final_comparison(results, training_history):
    """Create final comprehensive comparison."""
    print("\n" + "="*50)
    print("8. FINAL MODEL COMPARISON")
    print("="*50)
    
    # Prepare comparison data
    model_results = {name: result['metrics'] for name, result in results.items()}
    
    # Plot comparison
    plot_model_comparison(model_results, save_path='model_comparison.png')
    
    # Create detailed comparison table
    detailed_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Translation RMSE (m)': [results[name]['metrics']['translation_rmse'] for name in results.keys()],
        'Translation MAE (m)': [results[name]['metrics']['translation_mae'] for name in results.keys()],
        'Rotation RMSE (deg)': [results[name]['metrics']['rotation_rmse'] for name in results.keys()],
        'Rotation MAE (deg)': [results[name]['metrics']['rotation_mae'] for name in results.keys()],
        'Final Val Loss': [training_history[name]['val'][-1] for name in results.keys()]
    })
    
    print("\nDetailed Model Comparison:")
    print(detailed_comparison.to_string(index=False, float_format='%.4f'))
    
    # Save results
    detailed_comparison.to_csv('model_comparison_detailed.csv', index=False)
    print("\nDetailed comparison saved to 'model_comparison_detailed.csv'")
    
    # Generate evaluation reports
    for model_name, result in results.items():
        report = create_evaluation_report(result['metrics'], model_name)
        with open(f'{model_name.lower().replace("-", "_")}_evaluation_report.txt', 'w') as f:
            f.write(report)
        print(f"{model_name} evaluation report saved")
    
    print("\nAll evaluation reports and visualizations have been generated!")


def main():
    """Main function to run the complete example."""
    print("Vision-Based Multi-Agent Pose Estimator - Complete Usage Example")
    print("="*70)
    
    # Setup
    device = setup_environment()
    data_root = project_root
    cams = ['cam0']
    
    try:
        # 1. Load and explore data
        dataset = load_and_explore_data(data_root, cams)
        
        # 2. Create and compare models
        models = create_and_compare_models(device)
        
        # 3. Train models
        training_history, val_loader = train_models(models, dataset, device, epochs=3)
        
        # 4. Evaluate models
        results = evaluate_models(models, val_loader, device)
        
        # 5. Visualize results
        visualize_results(results, training_history)
        
        # 6. Demonstrate multi-agent capabilities
        demonstrate_multi_agent_capabilities()
        
        # 7. Simulate real-time inference
        best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['translation_rmse'])
        best_model = models[best_model_name]
        simulate_realtime_inference(best_model, dataset, device, n_frames=30)
        
        # 8. Create final comparison
        create_final_comparison(results, training_history)
        
        print("\n" + "="*70)
        print("COMPLETE USAGE EXAMPLE FINISHED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("- training_comparison.png")
        print("- trajectory_comparison.png")
        print("- realtime_inference.png")
        print("- model_comparison.png")
        print("- model_comparison_detailed.csv")
        print("- evaluation_results/ (directory)")
        print("- *_evaluation_report.txt files")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
