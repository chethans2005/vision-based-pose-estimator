"""
Comprehensive evaluation metrics for pose estimation.
Includes ATE, RPE, and specialized multi-agent metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import math


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute angular distance between quaternions.
    
    Args:
        q1, q2: Quaternions [N, 4] in (w, x, y, z) format
        
    Returns:
        Angular distances in degrees [N]
    """
    # Normalize quaternions
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12)
    
    # Compute dot product
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angular distance
    angle = 2 * np.arccos(dot)
    return angle * 180 / np.pi


def compute_ate(poses_gt: np.ndarray, poses_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        poses_gt: Ground truth poses [N, 7] (tx, ty, tz, qw, qx, qy, qz)
        poses_pred: Predicted poses [N, 7]
        
    Returns:
        Dictionary with ATE metrics
    """
    # Translation errors
    trans_gt = poses_gt[:, :3]
    trans_pred = poses_pred[:, :3]
    trans_errors = np.linalg.norm(trans_gt - trans_pred, axis=1)
    
    # Rotation errors
    rot_gt = poses_gt[:, 3:7]
    rot_pred = poses_pred[:, 3:7]
    rot_errors = quaternion_angular_distance(rot_gt, rot_pred)
    
    return {
        'translation_rmse': np.sqrt(np.mean(trans_errors**2)),
        'translation_mae': np.mean(trans_errors),
        'translation_max': np.max(trans_errors),
        'translation_std': np.std(trans_errors),
        'rotation_rmse': np.sqrt(np.mean(rot_errors**2)),
        'rotation_mae': np.mean(rot_errors),
        'rotation_max': np.max(rot_errors),
        'rotation_std': np.std(rot_errors)
    }


def compute_rpe(poses_gt: np.ndarray, poses_pred: np.ndarray, 
                delta: int = 1) -> Dict[str, float]:
    """
    Compute Relative Pose Error (RPE).
    
    Args:
        poses_gt: Ground truth poses [N, 7]
        poses_pred: Predicted poses [N, 7]
        delta: Time step for relative pose computation
        
    Returns:
        Dictionary with RPE metrics
    """
    n = len(poses_gt)
    if n <= delta:
        return {'translation_rpe': 0.0, 'rotation_rpe': 0.0}
    
    # Translation RPE
    trans_gt = poses_gt[:, :3]
    trans_pred = poses_pred[:, :3]
    
    gt_rel = trans_gt[delta:] - trans_gt[:-delta]
    pred_rel = trans_pred[delta:] - trans_pred[:-delta]
    trans_rpe = np.linalg.norm(gt_rel - pred_rel, axis=1)
    
    # Rotation RPE
    rot_gt = poses_gt[:, 3:7]
    rot_pred = poses_pred[:, 3:7]
    
    # Compute relative rotations
    gt_rel_rot = quaternion_multiply(quaternion_inverse(rot_gt[:-delta]), rot_gt[delta:])
    pred_rel_rot = quaternion_multiply(quaternion_inverse(rot_pred[:-delta]), rot_pred[delta:])
    
    rot_rpe = quaternion_angular_distance(gt_rel_rot, pred_rel_rot)
    
    return {
        'translation_rpe': np.mean(trans_rpe),
        'translation_rpe_std': np.std(trans_rpe),
        'rotation_rpe': np.mean(rot_rpe),
        'rotation_rpe_std': np.std(rot_rpe)
    }


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([w, x, y, z], axis=-1)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse."""
    return np.array([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]]).T


def compute_multi_agent_metrics(agent_poses_gt: Dict[int, np.ndarray],
                               agent_poses_pred: Dict[int, np.ndarray],
                               agent_presence_gt: Dict[int, np.ndarray],
                               agent_presence_pred: Dict[int, np.ndarray]) -> Dict[str, float]:
    """
    Compute specialized metrics for multi-agent pose estimation.
    
    Args:
        agent_poses_gt: Ground truth poses for each agent
        agent_poses_pred: Predicted poses for each agent
        agent_presence_gt: Ground truth presence for each agent
        agent_presence_pred: Predicted presence for each agent
        
    Returns:
        Dictionary with multi-agent metrics
    """
    metrics = {}
    
    # Agent detection metrics
    all_gt_presence = []
    all_pred_presence = []
    
    for agent_id in agent_poses_gt.keys():
        if agent_id in agent_presence_gt and agent_id in agent_presence_pred:
            all_gt_presence.extend(agent_presence_gt[agent_id])
            all_pred_presence.extend(agent_presence_pred[agent_id])
    
    if all_gt_presence and all_pred_presence:
        # Convert to binary predictions
        pred_binary = np.array(all_pred_presence) > 0.5
        gt_binary = np.array(all_gt_presence) > 0.5
        
        # Detection metrics
        tp = np.sum(pred_binary & gt_binary)
        fp = np.sum(pred_binary & ~gt_binary)
        fn = np.sum(~pred_binary & gt_binary)
        tn = np.sum(~pred_binary & ~gt_binary)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        metrics.update({
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'detection_accuracy': accuracy
        })
    
    # Formation metrics
    formation_metrics = compute_formation_metrics(agent_poses_gt, agent_poses_pred, 
                                                agent_presence_gt, agent_presence_pred)
    metrics.update(formation_metrics)
    
    return metrics


def compute_formation_metrics(agent_poses_gt: Dict[int, np.ndarray],
                            agent_poses_pred: Dict[int, np.ndarray],
                            agent_presence_gt: Dict[int, np.ndarray],
                            agent_presence_pred: Dict[int, np.ndarray]) -> Dict[str, float]:
    """
    Compute formation-specific metrics.
    
    Args:
        agent_poses_gt: Ground truth poses for each agent
        agent_poses_pred: Predicted poses for each agent
        agent_presence_gt: Ground truth presence for each agent
        agent_presence_pred: Predicted presence for each agent
        
    Returns:
        Dictionary with formation metrics
    """
    metrics = {}
    
    # Find common time steps where all agents are present
    agent_ids = list(agent_poses_gt.keys())
    if len(agent_ids) < 2:
        return metrics
    
    # Find time steps where all agents are present in ground truth
    common_present_gt = np.ones(len(agent_presence_gt[agent_ids[0]]), dtype=bool)
    for agent_id in agent_ids:
        if agent_id in agent_presence_gt:
            common_present_gt &= (agent_presence_gt[agent_id] > 0.5)
    
    if not np.any(common_present_gt):
        return metrics
    
    # Extract poses for common time steps
    gt_poses = {agent_id: poses[common_present_gt] for agent_id, poses in agent_poses_gt.items()}
    pred_poses = {agent_id: poses[common_present_gt] for agent_id, poses in agent_poses_pred.items()}
    
    # Compute formation center
    gt_center = np.mean([poses[:, :3] for poses in gt_poses.values()], axis=0)
    pred_center = np.mean([poses[:, :3] for poses in pred_poses.values()], axis=0)
    
    # Formation center error
    center_error = np.linalg.norm(gt_center - pred_center, axis=1)
    metrics['formation_center_rmse'] = np.sqrt(np.mean(center_error**2))
    metrics['formation_center_mae'] = np.mean(center_error)
    
    # Formation shape preservation
    gt_distances = []
    pred_distances = []
    
    for i, agent1 in enumerate(agent_ids):
        for j, agent2 in enumerate(agent_ids[i+1:], i+1):
            if agent1 in gt_poses and agent2 in gt_poses:
                gt_dist = np.linalg.norm(gt_poses[agent1][:, :3] - gt_poses[agent2][:, :3], axis=1)
                pred_dist = np.linalg.norm(pred_poses[agent1][:, :3] - pred_poses[agent2][:, :3], axis=1)
                gt_distances.append(gt_dist)
                pred_distances.append(pred_dist)
    
    if gt_distances and pred_distances:
        gt_distances = np.array(gt_distances).T  # [N, n_pairs]
        pred_distances = np.array(pred_distances).T  # [N, n_pairs]
        
        # Shape preservation error
        shape_error = np.linalg.norm(gt_distances - pred_distances, axis=1)
        metrics['formation_shape_rmse'] = np.sqrt(np.mean(shape_error**2))
        metrics['formation_shape_mae'] = np.mean(shape_error)
        
        # Formation stability (variance of inter-agent distances)
        gt_stability = np.mean(np.var(gt_distances, axis=0))
        pred_stability = np.mean(np.var(pred_distances, axis=0))
        metrics['formation_stability_gt'] = gt_stability
        metrics['formation_stability_pred'] = pred_stability
        metrics['formation_stability_error'] = abs(gt_stability - pred_stability)
    
    return metrics


def evaluate_model_performance(poses_gt: np.ndarray, poses_pred: np.ndarray,
                             agent_poses_gt: Optional[Dict[int, np.ndarray]] = None,
                             agent_poses_pred: Optional[Dict[int, np.ndarray]] = None,
                             agent_presence_gt: Optional[Dict[int, np.ndarray]] = None,
                             agent_presence_pred: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        poses_gt: Ground truth poses [N, 7]
        poses_pred: Predicted poses [N, 7]
        agent_poses_gt: Optional multi-agent ground truth poses
        agent_poses_pred: Optional multi-agent predicted poses
        agent_presence_gt: Optional multi-agent ground truth presence
        agent_presence_pred: Optional multi-agent predicted presence
        
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {}
    
    # Single agent metrics
    ate_metrics = compute_ate(poses_gt, poses_pred)
    rpe_metrics = compute_rpe(poses_gt, poses_pred)
    
    metrics.update(ate_metrics)
    metrics.update(rpe_metrics)
    
    # Multi-agent metrics if available
    if (agent_poses_gt is not None and agent_poses_pred is not None and
        agent_presence_gt is not None and agent_presence_pred is not None):
        multi_agent_metrics = compute_multi_agent_metrics(
            agent_poses_gt, agent_poses_pred, agent_presence_gt, agent_presence_pred
        )
        metrics.update(multi_agent_metrics)
    
    return metrics


def create_evaluation_report(metrics: Dict[str, float], 
                           model_name: str = "Model",
                           save_path: Optional[str] = None) -> str:
    """
    Create a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        save_path: Optional path to save the report
        
    Returns:
        Formatted report string
    """
    report = f"""
# Evaluation Report: {model_name}

## Single Agent Metrics

### Translation Errors
- RMSE: {metrics.get('translation_rmse', 0.0):.4f} m
- MAE: {metrics.get('translation_mae', 0.0):.4f} m
- Max Error: {metrics.get('translation_max', 0.0):.4f} m
- Std Dev: {metrics.get('translation_std', 0.0):.4f} m

### Rotation Errors
- RMSE: {metrics.get('rotation_rmse', 0.0):.2f}°
- MAE: {metrics.get('rotation_mae', 0.0):.2f}°
- Max Error: {metrics.get('rotation_max', 0.0):.2f}°
- Std Dev: {metrics.get('rotation_std', 0.0):.2f}°

### Relative Pose Errors
- Translation RPE: {metrics.get('translation_rpe', 0.0):.4f} m
- Rotation RPE: {metrics.get('rotation_rpe', 0.0):.2f}°
"""

    # Multi-agent metrics if available
    if 'detection_accuracy' in metrics:
        report += f"""
## Multi-Agent Metrics

### Agent Detection
- Precision: {metrics.get('detection_precision', 0.0):.3f}
- Recall: {metrics.get('detection_recall', 0.0):.3f}
- F1 Score: {metrics.get('detection_f1', 0.0):.3f}
- Accuracy: {metrics.get('detection_accuracy', 0.0):.3f}

### Formation Metrics
- Center RMSE: {metrics.get('formation_center_rmse', 0.0):.4f} m
- Center MAE: {metrics.get('formation_center_mae', 0.0):.4f} m
- Shape RMSE: {metrics.get('formation_shape_rmse', 0.0):.4f} m
- Shape MAE: {metrics.get('formation_shape_mae', 0.0):.4f} m
- Stability Error: {metrics.get('formation_stability_error', 0.0):.4f} m²
"""

    report += "\n---\n*Generated by Vision-Based Multi-Agent Pose Estimator*"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def compare_models(model_results: Dict[str, Dict[str, float]], 
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare multiple models and create a comparison table.
    
    Args:
        model_results: Dictionary mapping model names to their metrics
        save_path: Optional path to save the comparison table
        
    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(model_results).T
    
    # Round numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(4)
    
    if save_path:
        df.to_csv(save_path)
    
    return df
