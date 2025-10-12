"""Extract keyframes from an image folder.

Two modes:
- interval: take every Nth frame
- orb_change: take frames where ORB matches with previous are below a threshold (scene change)

Writes selected frames to output folder.
"""
import os
import argparse
import cv2


def extract_interval(img_dir, out_dir, interval):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    for i, f in enumerate(files):
        if i % interval == 0:
            src = os.path.join(img_dir, f)
            dst = os.path.join(out_dir, f)
            cv2.imwrite(dst, cv2.imread(src))


def extract_orb_change(img_dir, out_dir, threshold=30, max_features=500):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    orb = cv2.ORB_create(max_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp = None
    prev_des = None
    for idx, f in enumerate(files):
        path = os.path.join(img_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        take = False
        if idx == 0:
            take = True
        else:
            if prev_des is None or des is None:
                take = True
            else:
                matches = bf.match(prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)
                avg_dist = sum([m.distance for m in matches[:50]]) / max(1, min(50, len(matches)))
                if avg_dist > threshold:
                    take = True
        if take:
            dst = os.path.join(out_dir, f)
            cv2.imwrite(dst, cv2.imread(path))
        prev_kp, prev_des = kp, des


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--mode', choices=['interval', 'orb_change'], default='interval')
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--threshold', type=float, default=30.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'interval':
        extract_interval(args.img_dir, args.out_dir, args.interval)
    else:
        extract_orb_change(args.img_dir, args.out_dir, args.threshold)
