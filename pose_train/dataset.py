import os
import glob
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_image_gray(path: str, size: Tuple[int, int] = None) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    if size is not None:
        im = cv2.resize(im, (size[1], size[0]))
    im = im.astype(np.float32) / 255.0
    return im


def _parse_timestamp_from_filename(p: str) -> int:
    name = os.path.basename(p)
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except Exception:
        # fallback: try parsing leading number
        import re
        m = re.search(r"(\d+)", stem)
        return int(m.group(1)) if m else 0


class MAVPoseDataset(Dataset):
    """Dataset that aligns images (one or more cameras) with ground-truth by timestamp.

    - For each camera, image filenames must contain timestamps (integer nanoseconds) as the basename.
    - The loader finds timestamps common to all selected cameras and matches each timestamp to the nearest
      ground-truth row.

    Returns: image tensor (C,H,W) and pose vector [tx,ty,tz, qw,qx,qy,qz] as float32 torch tensors.
    """

    def __init__(self,
                 data_root: str,
                 cams: List[str] = ["cam0"],
                 img_size: Tuple[int, int] = (240, 320),
                 max_dt_ns: int = 50000000,
                 transform=None,
                 use_imagenet_norm: bool = False,
                 augment: bool = False,
                 val_indices: Optional[List[int]] = None):
        self.data_root = data_root
        self.cams = cams
        self.img_size = img_size
        self.max_dt_ns = max_dt_ns
        self.transform = transform
        self.use_imagenet_norm = use_imagenet_norm
        self.augment = augment

        gt_csv = os.path.join(data_root, "state_groundtruth_estimate0", "data.csv")
        # read header even if it starts with '#'
        self.gt = pd.read_csv(gt_csv, header=0)
        # Normalize column names: strip whitespace and leading '#'
        cols = [c.strip() for c in self.gt.columns]
        cols = [c.lstrip('#').strip() for c in cols]
        self.gt.columns = cols
        # gt timestamps (int64). Prefer 'timestamp' or fallback to first column
        if 'timestamp' in self.gt.columns:
            self.gt_ts = self.gt['timestamp'].astype(np.int64).to_numpy()
        else:
            # use first column as timestamp
            self.gt_ts = self.gt.iloc[:, 0].astype(np.int64).to_numpy()

        # read per-camera images and timestamps
        cam_ts = []
        cam_files = []
        for cam in cams:
            img_dir = os.path.join(data_root, cam, "data")
            files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            ts = [_parse_timestamp_from_filename(f) for f in files]
            cam_ts.append(np.array(ts, dtype=np.int64))
            cam_files.append(files)

        # find common timestamps across cameras (intersection)
        common_ts = set(cam_ts[0])
        for t in cam_ts[1:]:
            common_ts &= set(t.tolist())
        common_ts = sorted(list(common_ts))

        # build list of samples: for each timestamp present in all cams, find nearest gt index
        self.samples = []  # list of (cam_file_paths..., gt_index)
        if len(common_ts) == 0:
            # fallback: use shortest camera list and match by nearest
            base_files = cam_files[0]
            for i in range(min(len(l) for l in cam_files)):
                ts_i = _parse_timestamp_from_filename(base_files[i])
                gt_idx = int(np.argmin(np.abs(self.gt_ts - ts_i)))
                if abs(int(self.gt_ts[gt_idx]) - ts_i) <= self.max_dt_ns:
                    files = [cam_files[c][i] for c in range(len(cams))]
                    self.samples.append((files, gt_idx))
        else:
            # create a mapping from timestamp->file for each camera for fast lookup
            cam_map = []
            for files in cam_files:
                d = { _parse_timestamp_from_filename(f): f for f in files }
                cam_map.append(d)
            for ts in common_ts:
                # find nearest gt index
                gt_idx = int(np.argmin(np.abs(self.gt_ts - ts)))
                if abs(int(self.gt_ts[gt_idx]) - ts) <= self.max_dt_ns:
                    files = [cam_map[c][ts] for c in range(len(cams))]
                    self.samples.append((files, gt_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No matched samples found; check timestamps and max_dt_ns")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        files, gt_idx = self.samples[idx]
        imgs = [read_image_gray(p, size=self.img_size) for p in files]
        img = np.stack(imgs, axis=0)  # C,H,W

        gt_row = self.gt.iloc[gt_idx]
        # ensure quaternion order is [qw,qx,qy,qz]
        quat = np.array([
            gt_row['q_RS_w []'],
            gt_row['q_RS_x []'],
            gt_row['q_RS_y []'],
            gt_row['q_RS_z []']
        ], dtype=np.float32)
        # normalize quaternion
        qn = quat / (np.linalg.norm(quat) + 1e-12)

        pose = np.array([
            gt_row['p_RS_R_x [m]'],
            gt_row['p_RS_R_y [m]'],
            gt_row['p_RS_R_z [m]'],
            qn[0], qn[1], qn[2], qn[3]
        ], dtype=np.float32)

        # img: C,H,W with float32 in [0,1]
        # Apply user-provided transform first (keeps backward compatibility)
        if self.transform is not None:
            img = self.transform(img)

        # Basic augmentations (in numpy) if requested
        if self.augment:
            # horizontal flip with 50% chance
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=2).copy()
            # brightness jitter: scale factor between 0.8 and 1.2
            factor = 0.8 + 0.4 * np.random.rand()
            img = np.clip(img * factor, 0.0, 1.0)

        # If ImageNet normalization is requested, ensure 3 channels and normalize per-channel
        if self.use_imagenet_norm:
            # if single-channel, repeat to 3 channels
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            # convert to torch tensor C,H,W
            img_t = torch.from_numpy(img).float()
            # normalize using ImageNet mean/std
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(-1, 1, 1)
            img_t = (img_t - mean) / std
        else:
            img_t = torch.from_numpy(img).float()
        pose_t = torch.from_numpy(pose).float()
        return img_t, pose_t
