"""
Comprehensive visualization tools for multi-agent pose estimation.
Includes trajectory plotting, error analysis, and real-time monitoring.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import torch


def plot_trajectory_2d(poses_gt: np.ndarray, poses_pred: np.ndarray, 
                      title: str = "Trajectory Comparison", 
                      save_path: Optional[str] = None,
                      show_errors: bool = True):
    """
    Plot 2D trajectory comparison between ground truth and predictions.
    
    Args:
        poses_gt: Ground truth poses [N, 7] (tx, ty, tz, qw, qx, qy, qz)
        poses_pred: Predicted poses [N, 7]
        title: Plot title
        save_path: Path to save the plot
        show_errors: Whether to show error vectors
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # XY trajectory
    axes[0].plot(poses_gt[:, 0], poses_gt[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[0].plot(poses_pred[:, 0], poses_pred[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    
    if show_errors:
        # Show error vectors every 10th point
        step = max(1, len(poses_gt) // 20)
        for i in range(0, len(poses_gt), step):
            dx = poses_pred[i, 0] - poses_gt[i, 0]
            dy = poses_pred[i, 1] - poses_gt[i, 1]
            axes[0].arrow(poses_gt[i, 0], poses_gt[i, 1], dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
    
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('XY Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # XZ trajectory
    axes[1].plot(poses_gt[:, 0], poses_gt[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[1].plot(poses_pred[:, 0], poses_pred[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    
    if show_errors:
        for i in range(0, len(poses_gt), step):
            dx = poses_pred[i, 0] - poses_gt[i, 0]
            dz = poses_pred[i, 2] - poses_gt[i, 2]
            axes[1].arrow(poses_gt[i, 0], poses_gt[i, 2], dx, dz, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
    
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('XZ Trajectory')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_trajectory_3d(poses_gt: np.ndarray, poses_pred: np.ndarray,
                      title: str = "3D Trajectory Comparison",
                      save_path: Optional[str] = None):
    """
    Plot 3D trajectory comparison.
    
    Args:
        poses_gt: Ground truth poses [N, 7]
        poses_pred: Predicted poses [N, 7]
        title: Plot title
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(poses_gt[:, 0], poses_gt[:, 1], poses_gt[:, 2], 
           'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(poses_pred[:, 0], poses_pred[:, 1], poses_pred[:, 2], 
           'r--', linewidth=2, label='Predicted', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(*poses_gt[0, :3], color='green', s=100, marker='o', label='Start (GT)')
    ax.scatter(*poses_gt[-1, :3], color='green', s=100, marker='s', label='End (GT)')
    ax.scatter(*poses_pred[0, :3], color='red', s=100, marker='o', label='Start (Pred)')
    ax.scatter(*poses_pred[-1, :3], color='red', s=100, marker='s', label='End (Pred)')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_analysis(poses_gt: np.ndarray, poses_pred: np.ndarray,
                       title: str = "Error Analysis",
                       save_path: Optional[str] = None):
    """
    Plot comprehensive error analysis.
    
    Args:
        poses_gt: Ground truth poses [N, 7]
        poses_pred: Predicted poses [N, 7]
        title: Plot title
        save_path: Path to save the plot
    """
    # Calculate errors
    trans_errors = np.linalg.norm(poses_pred[:, :3] - poses_gt[:, :3], axis=1)
    
    # Quaternion errors (angular distance)
    q_gt = poses_gt[:, 3:7]
    q_pred = poses_pred[:, 3:7]
    # Normalize quaternions
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-12)
    q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-12)
    
    # Angular distance between quaternions
    dot_products = np.abs(np.sum(q_gt * q_pred, axis=1))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    rot_errors = 2 * np.arccos(dot_products) * 180 / np.pi  # Convert to degrees
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Translation errors over time
    axes[0, 0].plot(trans_errors, 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Translation Error (m)')
    axes[0, 0].set_title('Translation Error Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rotation errors over time
    axes[0, 1].plot(rot_errors, 'r-', linewidth=1)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Rotation Error (deg)')
    axes[0, 1].set_title('Rotation Error Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Translation error histogram
    axes[0, 2].hist(trans_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_xlabel('Translation Error (m)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Translation Error Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Rotation error histogram
    axes[1, 0].hist(rot_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Rotation Error (deg)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Rotation Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error vs trajectory length
    trajectory_length = np.cumsum(np.linalg.norm(np.diff(poses_gt[:, :3], axis=0), axis=1))
    trajectory_length = np.concatenate([[0], trajectory_length])
    axes[1, 1].scatter(trajectory_length, trans_errors, alpha=0.6, s=10)
    axes[1, 1].set_xlabel('Trajectory Length (m)')
    axes[1, 1].set_ylabel('Translation Error (m)')
    axes[1, 1].set_title('Error vs Trajectory Length')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error statistics
    stats_text = f"""Error Statistics:
    Translation RMSE: {np.sqrt(np.mean(trans_errors**2)):.4f} m
    Translation MAE: {np.mean(trans_errors):.4f} m
    Translation Max: {np.max(trans_errors):.4f} m
    
    Rotation RMSE: {np.sqrt(np.mean(rot_errors**2)):.2f}°
    Rotation MAE: {np.mean(rot_errors):.2f}°
    Rotation Max: {np.max(rot_errors):.2f}°"""
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_agent_trajectory(agent_poses: Dict[int, np.ndarray],
                               agent_presence: Dict[int, np.ndarray],
                               title: str = "Multi-Agent Trajectory",
                               save_path: Optional[str] = None):
    """
    Plot trajectories for multiple agents.
    
    Args:
        agent_poses: Dictionary mapping agent_id to poses [N, 7]
        agent_presence: Dictionary mapping agent_id to presence [N]
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_poses)))
    
    for i, (agent_id, poses) in enumerate(agent_poses.items()):
        presence = agent_presence[agent_id]
        color = colors[i]
        
        # Only plot where agent is present
        present_mask = presence > 0.5
        if np.any(present_mask):
            poses_present = poses[present_mask]
            
            # XY trajectory
            axes[0].plot(poses_present[:, 0], poses_present[:, 1], 
                        color=color, linewidth=2, label=f'Agent {agent_id}', alpha=0.8)
            axes[0].scatter(poses_present[0, 0], poses_present[0, 1], 
                           color=color, s=100, marker='o', zorder=5)
            axes[0].scatter(poses_present[-1, 0], poses_present[-1, 1], 
                           color=color, s=100, marker='s', zorder=5)
            
            # XZ trajectory
            axes[1].plot(poses_present[:, 0], poses_present[:, 2], 
                        color=color, linewidth=2, label=f'Agent {agent_id}', alpha=0.8)
            axes[1].scatter(poses_present[0, 0], poses_present[0, 2], 
                           color=color, s=100, marker='o', zorder=5)
            axes[1].scatter(poses_present[-1, 0], poses_present[-1, 2], 
                           color=color, s=100, marker='s', zorder=5)
    
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('XY Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('XZ Trajectory')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_formation_analysis(agent_poses: Dict[int, np.ndarray],
                          agent_presence: Dict[int, np.ndarray],
                          title: str = "Formation Analysis",
                          save_path: Optional[str] = None):
    """
    Analyze and visualize formation flying patterns.
    
    Args:
        agent_poses: Dictionary mapping agent_id to poses [N, 7]
        agent_presence: Dictionary mapping agent_id to presence [N]
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate formation metrics
    agent_ids = list(agent_poses.keys())
    n_agents = len(agent_ids)
    
    if n_agents < 2:
        axes[0, 0].text(0.5, 0.5, 'Need at least 2 agents for formation analysis', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    # Find common time steps where all agents are present
    common_present = np.ones(len(agent_presence[agent_ids[0]]), dtype=bool)
    for agent_id in agent_ids:
        common_present &= (agent_presence[agent_id] > 0.5)
    
    if not np.any(common_present):
        axes[0, 0].text(0.5, 0.5, 'No common time steps with all agents present', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    # Extract poses for common time steps
    common_poses = {agent_id: poses[common_present] for agent_id, poses in agent_poses.items()}
    
    # Calculate inter-agent distances
    distances = {}
    for i, agent1 in enumerate(agent_ids):
        for j, agent2 in enumerate(agent_ids[i+1:], i+1):
            dist = np.linalg.norm(common_poses[agent1][:, :3] - common_poses[agent2][:, :3], axis=1)
            distances[f'{agent1}-{agent2}'] = dist
    
    # Plot inter-agent distances over time
    for pair, dist in distances.items():
        axes[0, 0].plot(dist, label=f'Agents {pair}', linewidth=2)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_title('Inter-Agent Distances Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot formation center trajectory
    center_trajectory = np.mean([poses[:, :3] for poses in common_poses.values()], axis=0)
    axes[0, 1].plot(center_trajectory[:, 0], center_trajectory[:, 1], 
                   'k-', linewidth=3, label='Formation Center', alpha=0.8)
    
    # Plot individual agent trajectories relative to center
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    for i, (agent_id, poses) in enumerate(common_poses.items()):
        relative_pos = poses[:, :3] - center_trajectory
        axes[0, 1].plot(relative_pos[:, 0], relative_pos[:, 1], 
                       color=colors[i], linewidth=1, label=f'Agent {agent_id}', alpha=0.6)
    
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title('Formation Center and Relative Positions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Plot formation spread (standard deviation of distances from center)
    spread = []
    for poses in common_poses.values():
        dist_from_center = np.linalg.norm(poses[:, :3] - center_trajectory, axis=1)
        spread.append(np.std(dist_from_center))
    
    axes[1, 0].bar(range(n_agents), spread, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Agent ID')
    axes[1, 0].set_ylabel('Position Spread (m)')
    axes[1, 0].set_title('Formation Spread by Agent')
    axes[1, 0].set_xticks(range(n_agents))
    axes[1, 0].set_xticklabels([f'Agent {i}' for i in agent_ids])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot formation stability (variance of inter-agent distances)
    stability_metrics = []
    for pair, dist in distances.items():
        stability_metrics.append(np.var(dist))
    
    axes[1, 1].bar(range(len(stability_metrics)), stability_metrics, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(stability_metrics))), alpha=0.7)
    axes[1, 1].set_xlabel('Agent Pair')
    axes[1, 1].set_ylabel('Distance Variance (m²)')
    axes[1, 1].set_title('Formation Stability (Lower = More Stable)')
    axes[1, 1].set_xticks(range(len(stability_metrics)))
    axes[1, 1].set_xticklabels(list(distances.keys()), rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_evaluation_dashboard(poses_gt: np.ndarray, poses_pred: np.ndarray,
                              agent_poses: Optional[Dict[int, np.ndarray]] = None,
                              agent_presence: Optional[Dict[int, np.ndarray]] = None,
                              save_dir: str = "evaluation_results"):
    """
    Create a comprehensive evaluation dashboard with all visualizations.
    
    Args:
        poses_gt: Ground truth poses [N, 7]
        poses_pred: Predicted poses [N, 7]
        agent_poses: Optional multi-agent poses
        agent_presence: Optional multi-agent presence
        save_dir: Directory to save all plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Single agent visualizations
    plot_trajectory_2d(poses_gt, poses_pred, 
                      title="2D Trajectory Comparison",
                      save_path=os.path.join(save_dir, "trajectory_2d.png"))
    
    plot_trajectory_3d(poses_gt, poses_pred,
                      title="3D Trajectory Comparison", 
                      save_path=os.path.join(save_dir, "trajectory_3d.png"))
    
    plot_error_analysis(poses_gt, poses_pred,
                       title="Error Analysis",
                       save_path=os.path.join(save_dir, "error_analysis.png"))
    
    # Multi-agent visualizations if available
    if agent_poses is not None and agent_presence is not None:
        plot_multi_agent_trajectory(agent_poses, agent_presence,
                                  title="Multi-Agent Trajectory",
                                  save_path=os.path.join(save_dir, "multi_agent_trajectory.png"))
        
        plot_formation_analysis(agent_poses, agent_presence,
                              title="Formation Analysis",
                              save_path=os.path.join(save_dir, "formation_analysis.png"))
    
    print(f"Evaluation dashboard saved to {save_dir}/")


def plot_model_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Plot comparison between different models.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        save_path: Path to save the plot
    """
    models = list(results.keys())
    metrics = ['translation_rmse', 'rotation_rmse', 'translation_mae', 'rotation_mae']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = [results[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, alpha=0.7, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Error')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
