"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import sys
import torch
import argparse
import matplotlib
from pathlib import Path
import time
import numpy as np
import pycolmap
import random
import os

from estimator.utils import get_image_pairs_paths
from estimator import get_estimator, available_models
from estimator import BaseEstimator

# This is to be able to use matplotlib also without a GUI
# if not hasattr(sys, "ps1"):
#     matplotlib.use("Agg")

##### Load images
# Matterport3d
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/matterport3d/map_free_eval/test/s00000/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([640, 360]) # WxH

# Replica
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/replica/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([360, 640])

# ucl_campus
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/ucl_campus_aria/map_free_eval/test/s00006/')
# K = np.array([[504.79, 0.0, 481.30], [0.0, 542.79, 271.85], [0.0, 0.0, 1.0]])
# im_size = np.array([576, 1024]) # HxW

# map_free
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/mapfree/map_free_eval/val/s00460')
# K = np.array([[547.9946, 0.0, 269.9052], [0.0, 547.9946, 352.2056], [0.0, 0.0, 1.0]])
# im_size = np.array([720, 540]) # HxW

# hkustgz_campus
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/test_gray/s00000')
# K = np.array([[913.896, 0.0, 638.954], [0.0, 912.277, 364.884], [0.0, 0.0, 1.0]])
# im_size = np.array([1280, 720])

# 360loc_aria
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_aria/map_free_eval/test/s00002/')

# 360loc_device1
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device1/map_free_eval/test/s00002/')

# 360loc_device2
scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device2/map_free_eval/test/s00002/')

# 360loc_device3
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device3/map_free_eval/test/s00002/')

# 360loc_device4
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device4/map_free_eval/test/s00000/')

est_opts = {
    'known_extrinsics': True,
    'known_intrinsics': False,
    'niter': 300,
    'two_stage_opt_niter': 50,
    'handle_cross_device': True,
    'resize': (512, 288),
}

def visualize_images(images):
    """
    Visualize list of images and a query image using matplotlib.
    
    Args:
        images: List of images (can be torch.Tensor or numpy arrays)
        img1: Query image (can be torch.Tensor or numpy array)
    """
    import matplotlib.pyplot as plt
    
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

def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    estimator = get_estimator(
        args.model, 
        device=args.device, 
        max_num_keypoint=args.max_num_keypoint, 
        out_dir=args.out_dir
    )
    estimator.verbose = True

    for i in range(1):
        ##### Set image names
        list_img0_name = [
            'seq1/frame_00000.jpg',
            'seq1/frame_00001.jpg',
        ]
        list_img0_name = list_img0_name[:]
        img1_name = 'seq0/frame_00000.jpg'

        ##### Load poses and intrinsics
        poses_load = {}
        with (scene_root / 'poses.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line: continue
                line = line.strip().split(' ')
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                pose = pycolmap.Rigid3d()
                pose.translation = qt[4:]
                pose.rotation = pycolmap.Rotation3d(np.roll(qt[:4], -1))
                poses_load[img_name] = pose

        intr_load = {}
        with (scene_root / 'intrinsics.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line: continue
                line = line.strip().split(' ')
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])
                intr_load[img_name] = {'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), 'im_size': np.array([W, H])}

        list_img0_poses = [] # Pose from world to camera
        for name in list_img0_name:
            pose = np.eye(4)
            pose[:3, :] = poses_load[name].matrix()
            list_img0_poses.append(torch.from_numpy(np.linalg.inv(pose)))

        list_img0_intr = [{'K': torch.from_numpy(intr_load[name]['K']), 'im_size': torch.from_numpy(intr_load[name]['im_size'])} for name in list_img0_name]
        img1_intr = {'K': torch.from_numpy(intr_load[img1_name]['K']), 'im_size': torch.from_numpy(intr_load[img1_name]['im_size'])}

        ##### Check if the intrinsics are the same
        # if est_opts['handle_cross_device']:
        #     dest_size = intr_load[list_img0_name[0]]['im_size']
        # else:
        #     dest_size = intr_load[img1_name]['im_size']
        # list_img0 = [BaseEstimator.load_image(scene_root/name, resize=(512, 288)) for name in list_img0_name]
        # img1 = BaseEstimator.load_image(scene_root/img1_name, resize=(512, 288), dest_size=dest_size)
        # visualize_images(list_img0 + [img1])

        ##### Perform pose estimation
        try:
            start_time = time.time()
            result = estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
            print(f"Processing time: {time.time() - start_time:.2f}s")
            print(f"Estimated pose_w2c: {result['im_pose'][:3, 3:4].T}")
            # print(f"Edge score: {edge_scores}")
            # print(f"Focal length: {result['focal'][0]:.03f}")
            # print(f"Loss: {result['loss']:.03f}")
        except Exception as e:
            print(f"Error: {e}")
            pass

        msp_edges = estimator.get_minimum_spanning_tree()
        weight_i, weight_j = estimator.scene.weight_i, estimator.scene.weight_j
        for edge in msp_edges:
            if edge[0] == 2 or edge[1] == 2: # confidence of the query image
                edge_str = f"{edge[0]}_{edge[1]}"
                conf = (weight_i[edge_str].mean() * weight_j[edge_str].mean()).detach().cpu().item()
                print(f"Conf of {edge_str}: {conf:.3f}")

        estimator.show_reconstruction()

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
    # parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")
    parser.add_argument("--max_num_keypoint", type=int, default=2048, help="maximum number of keypoints")
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.model}")

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
