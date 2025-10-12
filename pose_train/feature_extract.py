"""Compute ORB keypoints and descriptors for all images in a folder and save to .npz per-image.
"""
import os
import argparse
import numpy as np
import cv2


def extract_orb(img_dir, out_dir, max_features=1000):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    orb = cv2.ORB_create(max_features)
    for f in files:
        p = os.path.join(img_dir, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        # convert keypoints to simple arrays
        kparr = np.array([[int(k.pt[0]), int(k.pt[1]), int(k.size), int(k.angle)] for k in kp], dtype=np.int32) if kp else np.zeros((0,4), dtype=np.int32)
        out_path = os.path.join(out_dir, f.replace('.png', '.npz'))
        np.savez_compressed(out_path, kps=kparr, des=des)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-features', type=int, default=1000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    extract_orb(args.img_dir, args.out_dir, max_features=args.max_features)
