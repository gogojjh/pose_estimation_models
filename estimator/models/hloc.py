import os
import numpy as np
from pathlib import Path

from estimator import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseEstimator
from estimator.utils import to_numpy, to_tensor, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath('Hierarchical-Localization'), insert=0)
add_to_path(THIRD_PARTY_DIR.joinpath('CF-3DGS'), insert=0)

import torch
import pycolmap

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

# packages for pose alignment
from utils.utils_poses.comp_ate import compute_ATE
from utils.utils_poses.ATE.align_utils import alignTrajectory
from utils.utils_poses.lie_group_helper import SO3_to_quat, convert3x4_4x4

from evo.core.trajectory import PosePath3D
import matplotlib.pyplot as plt
from evo.tools import plot
import copy

def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None, method='sim3'):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4)
    :param traj_b:  (N0, 3/4, 4)
    :param traj_c:  None or (N1, 3/4, 4)
    :return:        (N1, 4,   4)
    """
    if traj_c is None: traj_c = traj_a.copy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method=method)

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    ret_align = {'s': s, 'R': R, 't': t}
    return traj_c_aligned, ret_align  # (N1, 4, 4)

def plot_pose(ref_poses, est_poses, output_path : Path):
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)

    fig = plt.figure()
    traj_by_label = {
        # "estimate (not aligned)": traj_est,
        "Ours (aligned)": traj_est_aligned,
        "Ground-truth": traj_ref
    }
    plot_mode = plot.PlotMode.xyz
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.zaxis.set_tick_params(labelleft=False)
    colors = ['r', 'b']
    styles = ['-', '--']

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj,
                  styles[idx], colors[idx], label)
    ax.view_init(elev=10., azim=45)
    plt.tight_layout()
    fig.savefig(output_path / "vis_aligned_poses.png")

class HlocEstimator(BaseEstimator):
    def __init__(self, device="cpu", feature_name='disk', matcher_name='disk+lightglue', max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)

        self.feature_conf = extract_features.confs[feature_name]
        self.feature_conf['model']['max_keypoints'] = max_num_keypoints
        print('[Hloc] feature_conf: ', self.feature_conf)

        self.matcher_conf = match_features.confs[matcher_name]
        print('[Hloc] matcher_conf: ', self.matcher_conf)

        self.loc_conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        print('[Hloc] loc_conf: ', self.loc_conf)

        pycolmap.logging.minloglevel = 1
       
    def show_reconstruction(self, cam_size=0.2):
        # Visualize 2D points with successful triangulation
        visualization.visualize_sfm_2d(self.model, self.scene_root, color_by="visibility", n=1)
        plt.savefig(self.scene_root / "vis_sfm_2d.png")

        # Visualize 2D matching
        visualization.visualize_loc_from_log(self.scene_root, self.query_name, self.log, self.model)
        plt.savefig(self.scene_root / "vis_loc_2d_matching.png")
                
        scene = viz_3d.init_figure()
        # Visualize mapping
        viz_3d.plot_reconstruction(scene, self.model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True)
        pose = pycolmap.Image(cam_from_world=self.ret["cam_from_world"])
        # Visualize query camera and world coordinate system
        viz_3d.plot_camera_colmap(scene, pose, self.query_camera, color="rgba(0,255,0,0.5)", name=self.query_name, fill=True, size=cam_size)
        viz_3d.plot_camera_colmap(scene, pycolmap.Image(cam_from_world=pycolmap.Rigid3d()), self.query_camera, color="rgba(0,0,255,0.5)", name='world', fill=True, size=cam_size)
        # Visualize 2D-3D correspodences
        inl_3d = np.array([self.model.points3D[pid].xyz for pid in np.array(self.log["points3D_ids"])[self.ret["inliers"]]])
        viz_3d.plot_points(scene, inl_3d, color="lime", ps=1, name=self.query_name)
        scene.write_image(self.scene_root / "reconstruction.png")

        for image in self.model.images.values(): print(image)

        # scene.show()

    def recover_metric_pose(self, pose_img_dict):
        ##### Aligning colmap poses with groundtruth poses: scale, rotation, translation
        num_med_points3D = np.median(np.array([image.num_points3D for image in self.model.images.values()]))
        print('num_med_points3D: ', num_med_points3D)
        poses_pred, poses_gt = np.zeros((0, 4, 4)), np.zeros((0, 4, 4))
        for image in self.model.images.values():
            if image.num_points3D > num_med_points3D / 2: # filter out bad images with few 3D points
                pose = np.eye(4)
                
                pose[:3, :] = image.cam_from_world.matrix()
                pose = np.expand_dims(pose, axis=0) # (1, 4, 4)
                poses_pred = np.vstack((poses_pred, pose)) # (N, 4, 4)

                pose = np.expand_dims(pose_img_dict[image.name], axis=0)
                poses_gt = np.vstack((poses_gt, pose))
        c2ws_est_aligned, ret_align = align_ate_c2b_use_a2b(poses_pred, poses_gt)

        ##### Compute ATE        
        ate = compute_ATE(poses_gt, poses_pred)
        print('ATE before alignment: ', ate)
        # for i in range(1, len(poses_pred), 1): print(poses_pred[i, :3, 3].T, poses_gt[i, :3, 3].T)
        ate = compute_ATE(poses_gt, c2ws_est_aligned)
        print('ATE after alignment: ', ate)
        for i in range(1, len(c2ws_est_aligned), 1): print(c2ws_est_aligned[i, :3, 3].T, poses_gt[i, :3, 3].T)
        
        ##### Recover the metric pose of the query image
        s, R, t = ret_align['s'], ret_align['R'], ret_align['t']
        pred_pose = pose_img_dict[self.query_name]
        R_c = pred_pose[:3, :3]  # (N1, 3, 3)
        t_c = pred_pose[:3, 3:4]  # (N1, 3, 1)
        R_c_aligned = R @ R_c  # (N1, 3, 3)
        t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)        
        metric_pose = np.eye(4)
        metric_pose[:3,  :3] = R_c_aligned
        metric_pose[:3, 3:4] = t_c_aligned

        ##### Visualize the alignment
        plot_pose(poses_gt, c2ws_est_aligned, self.scene_root)

        return metric_pose

    def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, init_img1_pose, list_img0_K, img1_K, option):
        list_img0_poses = [to_numpy(pose) for pose in list_img0_poses]
        init_img1_pose = to_numpy(init_img1_pose)
        list_img0_K = [to_numpy(K) for K in list_img0_K]
        img1_K = to_numpy(img1_K)

        outputs = Path(os.path.join(THIRD_PARTY_DIR, "../../outputs_colmap/"))
        sfm_pairs = outputs / "pairs-sfm.txt"
        loc_pairs = outputs / "pairs-loc.txt"
        sfm_dir = outputs / "sfm"
        features = outputs / "features.h5"
        matches = outputs / "matches.h5"

        ##### Load Iamges
        references = list_img0_name

        # Reconstruct the scene
        extract_features.main(self.feature_conf, scene_root, image_list=references, feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_features.main(self.matcher_conf, sfm_pairs, features=features, matches=matches)

        ##### Provide known camera parameters and perform reconstruction
        intrinsics = list_img0_K[0]
        if option['known_camera_params']:
            opts = dict(camera_model='SIMPLE_RADIAL', camera_params=','.join(map(str, [intrinsics[0][0], intrinsics[0][2], intrinsics[1][2], 0]))) # f, cx, cy, k
            model = reconstruction.main(sfm_dir, scene_root, sfm_pairs, features, matches, image_list=references, 
                                        image_options=opts,
                                        mapper_options=dict(ba_refine_focal_length=False, ba_refine_extra_params=False),
                                        verbose=False)
        else:
            model = reconstruction.main(sfm_dir, scene_root, sfm_pairs, features, matches, image_list=references)
        if model is None: return None, None, None

        ##### Perform localization
        query = img1_name
        extract_features.main(self.feature_conf, scene_root, image_list=[query], feature_path=features, overwrite=True)
        pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
        match_features.main(self.matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

        ##### Analyze the localization results
        query_camera = pycolmap.infer_camera_from_image(scene_root / query)
        ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r) is not None]
        localizer = QueryLocalizer(model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, query, query_camera, ref_ids, features, matches)
        print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
       
        ##### Store results for visualization
        self.scene_root = scene_root
        self.query_camera = query_camera
        self.query_name = query
        self.ret = ret
        self.log = log
        self.model = model

        ##### Perform pose alignment
        pose_img_dict = {img_name: pose for img_name, pose in zip(list_img0_name, list_img0_poses)}
        pose_img_dict[query] = init_img1_pose
        est_im_pose = self.recover_metric_pose(pose_img_dict)

        est_focal = np.array(query_camera.params[0])
        loss = model.compute_mean_reprojection_error()

        return est_focal, est_im_pose, loss