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

from estimator.utils import get_image_pairs_paths
from estimator import get_estimator, available_models

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")

def main(args):
    # image_size = [288, 512]
    # args.out_dir.mkdir(exist_ok=True, parents=True)
    estimator = get_estimator(args.model, device=args.device)

    # Load images
    scene_root = Path('/Rocket_ssd/dataset/data_litevloc/hkustgz_campus/map_free_eval/test/s00000/')
    for i in range(1):
        list_img0_name = [f'seq1/frame_{index:05}.jpg' for index in range(30)]
        img1_name = 'seq0/frame_00000.jpg'

        import numpy as np
        import pycolmap
        poses_load = {}
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
                poses_load[img_name] = pose
        
        list_img0_poses = []
        for name in list_img0_name:
            pose = np.eye(4)
            pose[:3, :] = poses_load[name].matrix()
            list_img0_poses.append(torch.from_numpy(pose))
        pose = np.eye(4)
        pose[:3, :] = poses_load[img1_name].matrix()
        init_img1_pose = torch.from_numpy(pose)
        # list_img0_poses = [torch.eye(4) for _ in list_img0_name]
        # init_img1_pose = torch.eye(4)

        list_img0_K = [torch.eye(3) for _ in list_img0_name]
        img1_K = torch.eye(3)

        start_time = time.time()
        option = {
            'known_camera_params': False,
            'resize': 512,
            'opt_cam': 'all',
        }
        result = estimator(scene_root, list_img0_name, img1_name, list_img0_poses, init_img1_pose, list_img0_K, img1_K, option)
        print(f"Processing time: {time.time() - start_time:.2f}s")
        print('Focal length: ', result['focal'])
        print('Estimated pose: ', result['im_pose'][:3, 3:4].T)
        print('Loss:', result['loss'])

        estimator.show_reconstruction(cam_size=0.2)

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
        choices=available_models,
        help="choose your model",
    )

    # Hyperparameters shared by all methods:
    # parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default="assets/example_pairs",
    #     help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line",
    # )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.model}")

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
