"""Evaluation utilities: compute ATE and RPE between predicted poses and ground-truth.

Expect CSVs with timestamps and pose columns similar to ground truth file.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_poses(csv_path):
    df = pd.read_csv(csv_path, header=0)
    cols = [c.strip().lstrip('#') for c in df.columns]
    df.columns = cols
    ts = df.iloc[:,0].astype(np.int64).to_numpy()
    poses = df[['p_RS_R_x [m]','p_RS_R_y [m]','p_RS_R_z [m]','q_RS_w []','q_RS_x []','q_RS_y []','q_RS_z []']].to_numpy()
    return ts, poses


def align_by_timestamp(ts_ref, poses_ref, ts_pred, poses_pred, max_dt=50000000):
    # For each pred timestamp, find nearest ref timestamp within max_dt (ns)
    matched = []
    for i, t in enumerate(ts_pred):
        j = np.argmin(np.abs(ts_ref - t))
        if abs(int(ts_ref[j]) - int(t)) <= max_dt:
            matched.append((j, i))
    if not matched:
        raise RuntimeError('No matched timestamps')
    r_idx, p_idx = zip(*matched)
    return poses_ref[list(r_idx)], poses_pred[list(p_idx)]


def compute_ate(gt, pred):
    # gt and pred are Nx7 arrays [tx,ty,tz, qw,qx,qy,qz]
    trans_gt = gt[:, :3]
    trans_pred = pred[:, :3]
    err = trans_gt - trans_pred
    ate = np.sqrt((err**2).sum(axis=1)).mean()
    return ate


def compute_rpe(gt, pred, delta=1):
    # use translation part of relative pose over delta frames
    trans_gt = gt[:, :3]
    trans_pred = pred[:, :3]
    errs = []
    for i in range(len(gt)-delta):
        gt_rel = trans_gt[i+delta] - trans_gt[i]
        pred_rel = trans_pred[i+delta] - trans_pred[i]
        errs.append(np.linalg.norm(gt_rel - pred_rel))
    return float(np.mean(errs)) if errs else 0.0


def plot_trajs(gt, pred):
    plt.plot(gt[:,0], gt[:,1], label='gt')
    plt.plot(pred[:,0], pred[:,1], label='pred')
    plt.legend()
    plt.title('XY Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', required=True)
    p.add_argument('--pred', required=True)
    p.add_argument('--max-dt', type=int, default=50000000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ts_gt, poses_gt = read_poses(args.gt)
    ts_pred, poses_pred = read_poses(args.pred)
    gt_m, pred_m = align_by_timestamp(ts_gt, poses_gt, ts_pred, poses_pred, max_dt=args.max_dt)
    print('Matched samples:', len(gt_m))
    ate = compute_ate(gt_m, pred_m)
    rpe = compute_rpe(gt_m, pred_m, delta=1)
    print('ATE (m):', ate)
    print('RPE (m):', rpe)
    plot_trajs(gt_m, pred_m)
