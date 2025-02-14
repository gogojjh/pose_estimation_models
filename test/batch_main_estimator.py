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

from estimator.utils import get_image_pairs_paths
from estimator import get_estimator, available_models

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")

scene_root_list = [
    Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/'),
    Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/'),
    Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/'),
    Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/'),
    Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/')
]
K_list = [
    np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]]),
    np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]]),
    np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]]),
    np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]]),
    np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]])
]
im_size_list = [
    np.array([1024, 576]), 
    np.array([1024, 576]),
    np.array([1024, 576]),
    np.array([1024, 576]),
    np.array([1024, 576])
]
list_img0_name_list = [
    ['seq/000000.color.jpg', 'seq/000004.color.jpg'],
    ['seq/000000.color.jpg', 'seq/000004.color.jpg'],
    ['seq/000007.color.jpg', 'seq/000008.color.jpg'],
    ['seq/000007.color.jpg', 'seq/000008.color.jpg'],
    ['seq/000017.color.jpg', 'seq/000018.color.jpg']
]
img1_name_list = [
    '../out_map2/seq/000000.color.jpg',
    '../out_map6/seq/000016.color.jpg',
    '../out_map2/seq/000007.color.jpg',
    '../out_map4/seq/000007.color.jpg',
    '../out_map4/seq/000016.color.jpg'
]
est_opts = {
    'known_extrinsics': True,
    'known_intrinsics': True,
    'resize': 512,
}

def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    estimator = get_estimator(args.model, device=args.device, max_num_keypoint=args.max_num_keypoint, out_dir=args.out_dir)
    for i in range(len(scene_root_list)):
        scene_root = scene_root_list[i]
        K = K_list[i]
        im_size = im_size_list[i]
        list_img0_name = list_img0_name_list[i]
        img1_name = img1_name_list[i]

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
        
        # Pose from world to camera
        list_img0_poses = []
        for name in list_img0_name:
            pose = np.eye(4)
            pose[:3, :] = poses_load[name].matrix()
            list_img0_poses.append(torch.from_numpy(np.linalg.inv(pose)))

        list_img0_intr = [{'K': torch.from_numpy(K), 'im_size': torch.from_numpy(im_size)} for _ in list_img0_name]
        img1_intr = {'K': torch.from_numpy(K), 'im_size': torch.from_numpy(im_size)}

        start_time = time.time()
        result = estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
        print(f"Processing time: {time.time() - start_time:.2f}s")
        print('Focal length: ', result['focal'][0])
        print('Estimated pose: ', result['im_pose'][:3, 3:4].T) # Pose from world to camera
        print('Loss:', result['loss'])
        print()

        # estimator.show_reconstruction()

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
    parser.add_argument("--max_num_keypoint", type=int, default=2048, help="maximum number of keypoints")
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.model}")

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)