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
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")

##### Load images
# Matterport3d
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/matterport3d/map_free_eval/test/s00000/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([640, 360]) # WxH

# Replica
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/replica/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([360, 640])

# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s00000/out_map0/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([360, 640])

# ucl_campus
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus/map_free_eval/test/s00005/')
# K = np.array([[504.79, 0.0, 481.30], [0.0, 542.79, 271.85], [0.0, 0.0, 1.0]])
# im_size = np.array([540, 960]) # HxW

# map_free
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/mapfree/map_free_eval/val/s00460')
# K = np.array([[547.9946, 0.0, 269.9052], [0.0, 547.9946, 352.2056], [0.0, 0.0, 1.0]])
# im_size = np.array([720, 540]) # HxW

# ucl_campus_meta_glass
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/ucl_campus_aria/map_free_eval/test/s00001')
# K = np.array([[444.4927, 0.0, 511.500], [0.0, 444.4927, 287.500], [0.0, 0.0, 1.0]])
# im_size = np.array([576, 1024]) # HxW

# hkustgz_campus
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/test_gray/s00000')
# K = np.array([[913.896, 0.0, 638.954], [0.0, 912.277, 364.884], [0.0, 0.0, 1.0]])
# im_size = np.array([1280, 720])

# 360loc
scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_aria/map_free_eval/test/s00004/')
K = np.array([[444.4927, 0.0, 511.5], [0.0, 444.4927, 287.5], [0.0, 0.0, 1.0]])
im_size = np.array([576, 1024]) # HxW

est_opts = {
    'known_extrinsics': True,
    'known_intrinsics': False,
    'resize': 512,
    'niter': 300,
    'two_stage_opt_niter': 50
}

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
        list_img0_name = [
            'seq1/frame_00000.jpg',
            'seq1/frame_00001.jpg'
        ]
        list_img0_name = list_img0_name[:]
        img1_name = 'seq0/frame_00000.jpg'

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
        list_img0 = [BaseEstimator.load_image(scene_root/name, (512, 288)) for name in list_img0_name]
        img1 = BaseEstimator.load_image(scene_root/img1_name, (512, 288))
        print(f"Loading images took {time.time() - start_time}s")

        start_time = time.time()
        result = estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
        print(f"Processing time: {time.time() - start_time:.2f}s")
        print(f"Estimated pose: {result['im_pose'][:3, 3:4].T}") # Pose from world to camera

        # print(f"Edge score: {edge_scores}")
        # print(f"Focal length: {result['focal'][0]:.03f}")
        # print(f"Loss: {result['loss']:.03f}")

        # msp_edges = estimator.get_minimum_spanning_tree()
        # weight_i, weight_j = estimator.scene.weight_i, estimator.scene.weight_j
        # for edge in msp_edges:
        #     if edge[0] == 2 or edge[1] == 2: # confidence of the query image
        #         edge_str = f"{edge[0]}_{edge[1]}"
        #         conf = (weight_i[edge_str].mean() * weight_j[edge_str].mean()).detach().cpu().item()
        #         print(f"Conf of {edge_str}: {conf:.3f}")

        estimator.show_reconstruction()

        # Visualize results
        # result = estimator.get_matched_kpts(scene_root, list_img0[0], img1)
        # print(f"Number of inliers: {result['num_inliers']}")
        # exit()

        # import open3d as o3d
        # all_pts3d = estimator.scene.get_pts3d() # all pts3d in the world frame
        # msk_conf = estimator.scene.get_masks()
        # # pts3d_flat = all_pts3d[0][msk_conf[0]].reshape(-1, 3)
        # pts3d_flat = all_pts3d[0].reshape(-1, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts3d_flat.detach().cpu().numpy())
        # o3d.io.write_point_cloud('/Rocket_ssd/dataset/tmp/estimator_0.pcd', pcd)
        # # pts3d_flat = all_pts3d[1][msk_conf[1]].reshape(-1, 3)
        # pts3d_flat = all_pts3d[1].reshape(-1, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts3d_flat.detach().cpu().numpy())
        # o3d.io.write_point_cloud('/Rocket_ssd/dataset/tmp/estimator_1.pcd', pcd)

        # DEBUG(gogojjh):
        # import cv2
        # new_size = tuple((1024, 576)) # WxH
        # depth_maps = estimator.scene.get_depthmaps()
        # depth_map = (depth_maps[0].detach().cpu().numpy() * 1000.0).astype(np.uint16)
        # re_depth = cv2.resize(depth_map, new_size, interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkust_aria/hkust_P000_N001/map_free_eval/train/s00015/seq1/frame_00010.pdepth.png', re_depth)
        # depth_map = (depth_maps[1].detach().cpu().numpy() * 1000.0).astype(np.uint16)
        # re_depth = cv2.resize(depth_map, new_size, interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkust_aria/hkust_P000_N001/map_free_eval/train/s00015/seq1/frame_00014.pdepth.png', re_depth)

        # DEBUG(gogojjh):
        # list_depth_img_name = ['seq1/frame_00019.pdepth.png', 'seq1/frame_00019.pdepth.png', 'seq1/frame_00021.pdepth.png']
        # save_img_dir = "/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkust_aria/hkust_P000_N001/map_free_eval/train/s00021/preds"
        # estimator.save_results(save_img_dir, scene_root, list_depth_img_name, 0)

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
