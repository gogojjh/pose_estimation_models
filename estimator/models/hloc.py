import os
import numpy as np
from pathlib import Path

from estimator import THIRD_PARTY_DIR, BaseEstimator
from estimator.utils import to_numpy, add_to_path, convert_matrix_to_vec, convert_vec_to_matrix

add_to_path(THIRD_PARTY_DIR.joinpath('Hierarchical-Localization'), insert=0)

from hloc import (
    extract_features,
    match_features,
    triangulation,
    reconstruction,
    visualization,
    pairs_from_poses,
    pairs_from_exhaustive,
)
from hloc.utils import viz_3d
from hloc.utils.read_write_model import Camera, Image
from hloc.utils.read_write_model import write_model
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

import pycolmap
import matplotlib.pyplot as plt

# TODO(gogojjh): recover metric pose
# Package4 for trajectory alignment
# add_to_path(THIRD_PARTY_DIR.joinpath('CF-3DGS'), insert=0)
# from utils.utils.poses.comp_ate import compute_ATE
# from utils.utils_poses.ATE.align_utils import alignTrajectory
# from utils.utils_poses.lie_group_helper import SO3_to_quat, convert3x4_4x4

# from evo.core.trajectory import PosePath3D
# from evo.tools import plot
# import copy

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
    def __init__(self, device="cpu", feature_name='disk', matcher_name='disk+lightglue', 
                 max_num_keypoints=2048, out_dir='/tmp', *args, **kwargs):
        super().__init__(device, **kwargs)

        self.out_dir = Path(out_dir)

        self.feature_conf = extract_features.confs[feature_name]
        self.feature_conf['model']['max_keypoints'] = max_num_keypoints

        self.matcher_conf = match_features.confs[matcher_name]

        self.loc_conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        # print('[Hloc] loc_conf: ', self.loc_conf)

        pycolmap.logging.minloglevel = 100
       
    def save_results(self):    
        # Visualize 2D points with successful triangulation
        visualization.visualize_sfm_2d(self.model, self.scene_root, color_by="visibility", n=1)
        plt.savefig(self.out_dir / "vis_sfm_2d.png")

        # Visualize 2D matching
        visualization.visualize_loc_from_log(self.scene_root, self.query_name, self.log, self.model)
        plt.savefig(self.out_dir / "vis_loc_2d_matching.png")
                
        scene = viz_3d.init_figure()
        # Visualize mapping
        viz_3d.plot_reconstruction(scene, self.model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True)
        pose = pycolmap.Image(cam_from_world=self.ret["cam_from_world"])
        # Visualize query camera and world coordinate system
        viz_3d.plot_camera_colmap(scene, pose, self.query_camera, 
                                  color="rgba(0,255,0,0.5)", name=self.query_name, fill=True, 
                                  size=0.2)
        viz_3d.plot_camera_colmap(scene, pycolmap.Image(cam_from_world=pycolmap.Rigid3d()), self.query_camera, 
                                  color="rgba(0,0,255,0.5)", name='world', fill=True, 
                                  size=0.2)
        # Visualize 2D-3D correspodences
        inl_3d = np.array([self.model.points3D[pid].xyz for pid in np.array(self.log["points3D_ids"])[self.ret["inliers"]]])
        viz_3d.plot_points(scene, inl_3d, color="lime", ps=1, name=self.query_name)
        scene.write_image(self.out_dir / "reconstruction.png")
        for image in self.model.images.values(): print(image)
        # print('Query camera: ', self.query_camera)
        self.scene = scene

    def show_reconstruction(self, cam_size=0.2):
        self.scene.show()

    def recover_metric_pose(self, poses_pred, poses_gt, query_pose):
        ##### Aligning colmap poses with groundtruth poses: scale, rotation, translation
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
        R_c = query_pose[:3, :3]  # (N1, 3, 3)
        t_c = query_pose[:3, 3:4]  # (N1, 3, 1)
        R_c_aligned = R @ R_c  # (N1, 3, 3)
        t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)        
        metric_query_pose = np.eye(4)
        metric_query_pose[:3,  :3] = R_c_aligned
        metric_query_pose[:3, 3:4] = t_c_aligned

        ##### Visualize the alignment
        plot_pose(poses_gt, c2ws_est_aligned, self.scene_root)

        return metric_query_pose

    def feature_process(self, scene_root, references, query):
        ##### Extract features, match features, and triangulation of reference images
        extract_features.main(self.feature_conf, scene_root, image_list=references, feature_path=self.path_features)
        pairs_from_exhaustive.main(self.path_sfm_pairs, image_list=references)
        match_features.main(self.matcher_conf, self.path_sfm_pairs, features=self.path_features, matches=self.path_matches)
        ##### Extract features, match features, and triangulation of the query image
        extract_features.main(self.feature_conf, scene_root, image_list=[query], feature_path=self.path_features, overwrite=True)
        pairs_from_exhaustive.main(self.path_loc_pairs, image_list=[query], ref_list=references)
        match_features.main(self.matcher_conf, self.path_loc_pairs, features=self.path_features, matches=self.path_matches, overwrite=True)

    def estimate_pose(self, scene_root, references, query, cam_opts=None, mapper_opts=None):
        self.feature_process(scene_root, references, query)
        # Reconstruction
        model = reconstruction.main(self.path_sfm_dir, scene_root, self.path_sfm_pairs, 
                                    self.path_features, self.path_matches, 
                                    image_options=cam_opts,
                                    mapper_options=mapper_opts,
                                    image_list=references)
        num_med_points3D = np.median(np.array([image.num_points3D for image in model.images.values()]))
        cnt = sum(1 for image in model.images.values() if image.num_points3D > num_med_points3D / 2)
        print('Successful trianglation:', cnt, '/', len(model.images))
        ##### Perform localization
        if cam_opts is not None:
            query_camera = pycolmap.infer_camera_from_image(scene_root / query, cam_opts)
        else:
            query_camera = pycolmap.infer_camera_from_image(scene_root / query)
        ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r) is not None]
        localizer = QueryLocalizer(model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, query, query_camera, ref_ids, self.path_features, self.path_matches)
        print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
        ret_query = {
            'num_med_points3D': num_med_points3D,
            'query_name': query,
            'query_camera': query_camera,
            'ret': ret,
            'log': log
        }
        est_focal = query_camera.params[0]
        est_im_pose = ret['cam_from_world'].matrix()
        loss = model.compute_mean_reprojection_error()

        return model, ret_query, est_focal, est_im_pose, loss
    
    def estimate_pose_with_int_and_ext(self, scene_root, references, query, reference_poses, reference_ints, query_int):
        # Reference images
        images, cameras = {}, {}
        for idx, ref_img in enumerate(references):
            pose_w2c = reference_poses[idx]
            # Convert colmap Pose from world to camera
            pose_c2w = np.linalg.inv(pose_w2c)
            txyz, qwxyz = convert_matrix_to_vec(pose_c2w, 'wxyz')
            image = Image(
                id=idx, qvec=qwxyz, tvec=txyz,
                camera_id=idx, name=ref_img,
                xys=[], point3D_ids=[])
            images[idx] = image

            fx, fy, cx, cy, width, height = \
                reference_ints[idx]['K'][0][0], reference_ints[idx]['K'][1][1], \
                reference_ints[idx]['K'][0][2], reference_ints[idx]['K'][1][2], \
                reference_ints[idx]['im_size'][0], reference_ints[idx]['im_size'][1]
            camera = Camera(
                id=idx, model='PINHOLE', 
                width=int(width), height=int(height), 
                params=[fx, fy, cx, cy])
            cameras[idx] = camera
        # print('Writing the COLMAP model ...')
        colmap_arkit = self.out_dir / 'colmap_arkit'
        colmap_arkit.mkdir(exist_ok=True, parents=True)
        write_model(images=images, cameras=cameras, points3D={}, path=str(colmap_arkit), ext='.bin')

        # Extract and matrix features of reference images
        colmap_sparse = self.out_dir / 'colmap_sparse'
        colmap_sparse.mkdir(exist_ok=True, parents=True)
        extract_features.main(self.feature_conf, scene_root, image_list=references, feature_path=self.path_features)
        pairs_from_poses.main(colmap_arkit, self.path_sfm_pairs, len(references))
        match_features.main(self.matcher_conf, self.path_sfm_pairs, features=self.path_features, matches=self.path_matches)
        # NOTE(gogjjh): this is the bug, that triangulation.main() sometimes cannot work properly
        # model = triangulation.main(colmap_sparse, colmap_arkit, scene_root, self.path_sfm_pairs, self.path_features, self.path_matches)
        model = reconstruction.main(self.path_sfm_dir, scene_root, self.path_sfm_pairs, self.path_features, self.path_matches, image_list=references)
        num_med_points3D = np.median(np.array([image.num_points3D for image in model.images.values()]))
        cnt = sum(1 for image in model.images.values() if image.num_points3D > num_med_points3D / 2)
        print('Successful trianglation:', cnt, '/', len(model.images))        
        
        # Extract and matrix features of the query image
        extract_features.main(self.feature_conf, scene_root, image_list=[query], feature_path=self.path_features, overwrite=True)
        pairs_from_exhaustive.main(self.path_loc_pairs, image_list=[query], ref_list=references)
        match_features.main(self.matcher_conf, self.path_loc_pairs, features=self.path_features, matches=self.path_matches, overwrite=True)        
        fx, fy, cx, cy = query_int['K'][0][0], query_int['K'][1][1], query_int['K'][0][2], query_int['K'][1][2]
        cam_opts = dict(camera_model='PINHOLE', camera_params=','.join(map(str, [fx, fy, cx, cy])))
        query_camera = pycolmap.infer_camera_from_image(scene_root / query, cam_opts)
        # Perform localization
        ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r) is not None]
        localizer = QueryLocalizer(model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, query, query_camera, ref_ids, self.path_features, self.path_matches)
        print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences of the query image.')
        ret_query = {
            'num_med_points3D': num_med_points3D,
            'query_name': query,
            'query_camera': query_camera,
            'ret': ret,
            'log': log
        }
        est_focal = query_camera.params[0]
        est_im_pose = ret['cam_from_world'].matrix()
        loss = model.compute_mean_reprojection_error()

        return model, ret_query, est_focal, est_im_pose, loss

    def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts):
        list_img0_poses = [to_numpy(pose) for pose in list_img0_poses]
        list_img0_intr = [{'K': to_numpy(intrinsics['K']), 'im_size': to_numpy(intrinsics['im_size'])} for intrinsics in list_img0_intr]
        img1_intr = {'K': to_numpy(img1_intr['K']), 'im_size': to_numpy(img1_intr['im_size'])}

        self.path_sfm_pairs = self.out_dir / "pairs-sfm.txt"
        self.path_loc_pairs = self.out_dir / "pairs-loc.txt"
        self.path_sfm_dir = self.out_dir / "sfm"
        self.path_features = self.out_dir / "features.h5"
        self.path_matches = self.out_dir / "matches.h5"

        ##### Perform reconstruction
        # know intrinsics
        if est_opts['known_extrinsics']:
            # estimate query poses with known extrinsics
            model, ret_query, est_focal, est_im_pose, loss = \
                self.estimate_pose_with_int_and_ext(scene_root, list_img0_name, img1_name, 
                                                    list_img0_poses, list_img0_intr, img1_intr)
        elif est_opts['known_intrinsics']:
            # estimate reference and query poses with known intrinsics
            # assume the same intrinsics for all images
            fx, fy, cx, cy = list_img0_intr[0]['K'][0][0], list_img0_intr[0]['K'][1][1], list_img0_intr[0]['K'][0][2], list_img0_intr[0]['K'][1][2]
            cam_opts = dict(camera_model='PINHOLE', camera_params=','.join(map(str, [fx, fy, cx, cy])))
            mapper_opts = dict(ba_refine_focal_length=False, ba_refine_extra_params=False)
            model, ret_query, est_focal, est_im_pose, loss = \
                self.estimate_pose(scene_root, list_img0_name, img1_name, cam_opts, mapper_opts)
        else:
            # estimate reference and query poses and focal length
            model, ret_query, est_focal, est_im_pose, loss = \
                self.estimate_pose(scene_root, list_img0_name, img1_name)

        if model.num_points3D() < 100:
            return None, None, None
        
        ##### Store results for visualization
        self.model = model
        self.scene_root = scene_root
        self.query_name = ret_query['query_name']
        self.query_camera = ret_query['query_camera']
        self.ret = ret_query['ret']
        self.log = ret_query['log']
        return est_focal, est_im_pose, loss