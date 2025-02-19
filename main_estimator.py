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

##### Load images
# Matterport3d
N_ref_image = 3
scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/matterport3d/map_free_eval/test/s00000/')
K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
im_size = np.array([640, 360]) # WxH
est_opts = {
    'known_extrinsics': True,
    'known_intrinsics': True,
    'resize': 512,
}

# Replica
# N_ref_image = 3
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/replica/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([360, 640])
# est_opts = {
#     'known_extrinsics': False,
#     'known_intrinsics': False,
#     'resize': 512,
# }

# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s00000/out_map0/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([360, 640])
# est_opts = {
#     'known_extrinsics': True,
#     'known_intrinsics': True,
#     'resize': 512,
# }

# ucl_campus
# N_ref_image = 3
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus/map_free_eval/test/s00005/')
# K = np.array([[504.79, 0.0, 481.30], [0.0, 542.79, 271.85], [0.0, 0.0, 1.0]])
# im_size = np.array([540, 960]) # HxW
# est_opts = {
#     'known_extrinsics': True,
#     'known_intrinsics': True,
#     'resize': 512,
# }

# ucl_campus_meta_glass
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/')
# K = np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]])
# im_size = np.array([576, 1024])
# est_opts = {
#     'known_extrinsics': True,
#     'known_intrinsics': True,
#     'resize': 512,
# }

# hkustgz_campus
# N_ref_image = 3
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/test/s00005/')
# K = np.array([[913.896, 0.0, 638.954], [0.0, 912.277, 364.884], [0.0, 0.0, 1.0]])
# im_size = np.array([720, 1280])
# est_opts = {
#     'known_extrinsics': True,
#     'known_intrinsics': True,
#     'resize': 512,
# }

# 360Loc
# N_ref_image = 12
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/360loc/map_free_eval/test/s00000_test/')
# K = np.array([[1055.911, 0.0, 939.453], [0.0, 1052.383, 603.812], [0.0, 0.0, 1.0]])
# im_size = np.array([1200, 1920])
# est_opts = {
#     'known_extrinsics': False,
#     'known_intrinsics': False,
#     'resize': 512,
# }

# Wildscene
# N_ref_image = 3
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/wildscene/map_free_eval/test/s00000_test/')
# K = np.array([[1322.75469666, 0.0, 1014.8117275], [0.0, 1321.88964261, 752.801443314], [0.0, 0.0, 1.0]])
# im_size = np.array([1512, 2016])
# est_opts = {
#     'known_extrinsics': False,
#     'known_intrinsics': False,
#     'resize': 512,
# }

def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    estimator = get_estimator(args.model, device=args.device, max_num_keypoint=args.max_num_keypoint, out_dir=args.out_dir)
    estimator.verbose = True
    for i in range(1):
        list_img0_name = ['seq0/frame_00000.jpg', 'seq1/frame_00005.jpg']
        img1_name = 'seq1/frame_00009.jpg'

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
        edge_scores = estimator.get_similarity()
        print(f"Processing time: {time.time() - start_time:.2f}s")
        print(f"Edge score: {edge_scores}")
        print(f"Focal length: {result['focal'][0]:.03f}")
        print(f"Estimated pose: {result['im_pose'][:3, 3:4].T}") # Pose from world to camera
        print(f"Loss: {result['loss']:.03f}")
        
        list_depth_img_name = ['seq0/frame_00000.zed.png', 'seq1/frame_00005.zed.png', 'seq1/frame_00009.zed.png']
        save_img_dir = "/Titan/code/robohike_ws/src/pose_estimation_models/outputs_duster"
        estimator.save_results(scene_root, list_depth_img_name, save_img_dir, 0)

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
    parser.add_argument("--max_num_keypoint", type=int, default=2048, help="maximum number of keypoints")

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
