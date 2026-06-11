"""
Entry-point script for pose estimation. Loads scene data (poses + intrinsics),
runs a configured estimator model (DUSt3R, MASt3R, HLoc, Reloc3r, VPR), and
outputs estimated camera pose and edge confidence scores.
"""

import torch
import argparse
from pathlib import Path
import time
import numpy as np
import pycolmap
import matplotlib.pyplot as plt

from estimator import get_estimator, available_models


def visualize_images(images):
    """
    Visualize list of images and a query image using matplotlib.
    
    Args:
        images: List of images (can be torch.Tensor or numpy arrays)
        img1: Query image (can be torch.Tensor or numpy array)
    """
    num_img0 = len(images) - 1
    fig, axs = plt.subplots(1, num_img0 + 1, figsize=(5 * (num_img0 + 1), 5))
    # Ensure axs is always iterable
    if (num_img0 + 1) == 1:
        axs = [axs]

    for i, img in enumerate(images[:-1]):
        if isinstance(img, torch.Tensor):
            img_vis = img.detach().cpu().numpy()
            if img_vis.shape[0] in (1, 3):  # CHW
                img_vis = np.transpose(img_vis, (1,2,0))
            if img_vis.shape[2] == 1:
                img_vis = img_vis[..., 0]
        else:
            img_vis = img
        axs[i].imshow(img_vis.astype(np.uint8) if img_vis.max() > 1 else img_vis)
        axs[i].axis('off')
        axs[i].set_title(f'image[{i}]')

    # Handle img1
    if isinstance(images[-1], torch.Tensor):
        img1_vis = images[-1].detach().cpu().numpy()
        if img1_vis.shape[0] in (1, 3):
            img1_vis = np.transpose(img1_vis, (1,2,0))
        if img1_vis.shape[2] == 1:
            img1_vis = img1_vis[..., 0]
    else:
        img1_vis = images[-1]
    axs[-1].imshow(img1_vis.astype(np.uint8) if img1_vis.max() > 1 else img1_vis)
    axs[-1].axis('off')
    axs[-1].set_title('image[-1]')

    plt.tight_layout()
    plt.show()


def load_scene_data(scene_root):
    """
    Load poses and intrinsics from a scene directory.

    Expects scene_root/poses.txt (qw qx qy qz tx ty tz per line)
    and scene_root/intrinsics.txt (fx fy cx cy W H per line).

    Returns (poses, intrinsics): dicts keyed by image name.
    """
    poses = {}
    with (scene_root / 'poses.txt').open('r') as f:
        for line in f.readlines():
            if '#' in line:
                continue
            line = line.strip().split(' ')
            img_name = line[0]
            qt = np.array(list(map(float, line[1:])))
            pose = pycolmap.Rigid3d()
            pose.translation = qt[4:]
            pose.rotation = pycolmap.Rotation3d(np.roll(qt[:4], -1))
            poses[img_name] = pose

    intrinsics = {}
    with (scene_root / 'intrinsics.txt').open('r') as f:
        for line in f.readlines():
            if '#' in line:
                continue
            line = line.strip().split(' ')
            img_name = line[0]
            fx, fy, cx, cy, W, H = map(float, line[1:])
            intrinsics[img_name] = {
                'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                'im_size': np.array([W, H]),
            }
    return poses, intrinsics


def main(args):
    est_opts = {
        'known_extrinsics': True,
        'known_intrinsics': False,
        'niter': 300,
        'two_stage_opt_niter': 50,
        'crop_image_to_database': True,
        'resize': (512, 288),
    }

    args.out_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1. Load scene data ----
    scene_root = Path(args.scene_root)
    poses_db, intr_db = load_scene_data(scene_root)
    print(f"Loaded {len(poses_db)} poses and {len(intr_db)} intrinsics from {scene_root}")

    # ---- 2. Configure estimator ----
    estimator = get_estimator(
        args.model,
        device=args.device,
        max_num_keypoint=args.max_num_keypoint,
        out_dir=args.out_dir,
    )
    estimator.verbose = True

    # ---- 3. Run estimation ----
    list_img0_name = [
        'seq1/frame_00000.jpg',
        'seq1/frame_00001.jpg',
    ]
    list_img0_name = list_img0_name[:]
    img1_name = 'seq0/frame_00000.jpg'

    list_img0_poses = []
    for name in list_img0_name:
        pose = np.eye(4)
        pose[:3, :] = poses_db[name].matrix()
        list_img0_poses.append(torch.from_numpy(np.linalg.inv(pose)))

    list_img0_intr = [
        {
            'K': torch.from_numpy(intr_db[name]['K']),
            'im_size': torch.from_numpy(intr_db[name]['im_size']),
        }
        for name in list_img0_name
    ]
    img1_intr = {
        'K': torch.from_numpy(intr_db[img1_name]['K']),
        'im_size': torch.from_numpy(intr_db[img1_name]['im_size']),
    }

    try:
        start_time = time.time()
        result = estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
        print(f"Processing time: {time.time() - start_time:.2f}s")
        print(f"Estimated pose_w2c: {result['im_pose'][:3, 3:4].T}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # ---- 4. Show edge confidence ----
    msp_edges = estimator.get_minimum_spanning_tree()
    conf_i, conf_j = estimator.scene.conf_i, estimator.scene.conf_j
    for edge in msp_edges:
        if edge[0] == 2 or edge[1] == 2:
            edge_str = f"{edge[0]}_{edge[1]}"
            conf = (conf_i[edge_str].mean() * conf_j[edge_str].mean()).detach().cpu().item()
            print(f"Conf of {edge_str}: {conf:.3f}")

    try:
        estimator.show_reconstruction()
    except Exception as e:
        print(f"Unable to show reconstruction: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose Estimator Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Choose estimator
    parser.add_argument(
        "--model",
        type=str,
        default="master",
        help=f"choose your model: {available_models}",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")
    parser.add_argument("--max_num_keypoint", type=int, default=2048, help="maximum number of keypoints")
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")
    parser.add_argument("--scene_root", type=Path, required=True, help="path to scene directory (contains poses.txt, intrinsics.txt, seq/)")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.model}")

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
