import numpy as np
from pathlib import Path
from estimator import THIRD_PARTY_DIR, BaseEstimator
from estimator.utils import to_numpy, add_to_path, convert_matrix_to_vec, align_poses

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
from hloc.utils.read_write_model import Camera, Image, write_model
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import pycolmap
import matplotlib.pyplot as plt

class HlocEstimator(BaseEstimator):
	def __init__(self, device="cpu", feature_name='disk', matcher_name='disk+lightglue', 
				 out_dir='/tmp', **kwargs):
		super().__init__(device, **kwargs)
		self.out_dir = Path(out_dir)
		self.paths = {
			'sfm_pairs': self.out_dir/"pairs-sfm.txt",
			'loc_pairs': self.out_dir/"pairs-loc.txt",
			'sfm_dir': self.out_dir/"sfm",
			'features': self.out_dir/"features.h5",
			'matches': self.out_dir/"matches.h5"
		}
		self.feature_conf = extract_features.confs[feature_name]
		self.matcher_conf = match_features.confs[matcher_name]
		self.loc_conf = {
			"estimation": {"ransac": {"max_error": 12}},
			"refinement": {"refine_focal_length": True, "refine_extra_params": True},
		}
		pycolmap.logging.minloglevel = 100

	def save_results(self, log_dir):
		log_dir = log_dir/"preds"
		log_dir.mkdir(exist_ok=True)
		
		# 2D Visualizations
		visualization.visualize_sfm_2d(self.model, self.scene_root, color_by="visibility", n=1)
		plt.savefig(log_dir/"vis_sfm_2d.png")
		visualization.visualize_loc_from_log(self.scene_root, self.query_name, self.log, self.model)
		plt.savefig(log_dir/"vis_loc_2d_matching.png")

		# 3D Visualization
		scene = viz_3d.init_figure()
		viz_3d.plot_reconstruction(scene, self.model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True)
		for pose, color, name in [(self.ret["cam_from_world"], "rgba(0,255,0,0.5)", self.query_name),
								(pycolmap.Rigid3d(), "rgba(0,0,255,0.5)", 'world')]:
			viz_3d.plot_camera_colmap(scene, pycolmap.Image(cam_from_world=pose), self.query_camera,
									color=color, name=name, fill=True, size=0.2)
		inl_3d = np.array([self.model.points3D[pid].xyz 
						 for pid in np.array(self.log["points3D_ids"])[self.ret["inliers"]]])
		viz_3d.plot_points(scene, inl_3d, color="lime", ps=1, name=self.query_name)
		scene.write_image(log_dir/"reconstruction.png")
		self.scene = scene

	def show_reconstruction(self, cam_size=0.2):
		self.scene.show()

	def estimate_pose(self, scene_root, references, query, cam_opts=None, mapper_opts=None):
		print(scene_root, references, query)
		##### Feature processing
		# Reference
		extract_features.main(
			self.feature_conf, scene_root, 
			image_list=references, feature_path=self.paths['features'], overwrite=True
		)
		pairs_from_exhaustive.main(self.paths['sfm_pairs'], image_list=references)
		match_features.main(
			self.matcher_conf, self.paths['sfm_pairs'], 
			features=self.paths['features'], matches=self.paths['matches'],
			overwrite=True
		)
		##### Reconstruction
		model = reconstruction.main(
			self.paths['sfm_dir'], scene_root, self.paths['sfm_pairs'],
			self.paths['features'], self.paths['matches'],
			image_options=cam_opts,
			mapper_options=mapper_opts,
			image_list=references
		)
		if not model or model.num_points3D() < 100: return (None,)*5
		median_points = np.median([img.num_points3D for img in model.images.values()])
		print(f'Triangulation success: {sum(1 for img in model.images.values() if img.num_points3D > median_points/2)}/{len(model.images)}')

		##### Localization
		# Query
		extract_features.main(
			self.feature_conf, scene_root, 
			image_list=[query], feature_path=self.paths['features'], overwrite=True
		)
		pairs_from_exhaustive.main(self.paths['loc_pairs'], image_list=[query], ref_list=references)
		match_features.main(
			self.matcher_conf, self.paths['loc_pairs'], 
			features=self.paths['features'], matches=self.paths['matches'], 
			overwrite=True
		)
		query_cam = pycolmap.infer_camera_from_image(scene_root/query, cam_opts or {})
		ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r)]
		ret, log = pose_from_cluster(
			QueryLocalizer(model, self.loc_conf), query, query_cam, ref_ids,
			self.paths['features'], self.paths['matches']
		)
		if not ret: return (None,)*5
		
		print(f'Inliers: {ret["num_inliers"]}/{len(ret["inliers"])}')
		pose = np.vstack((ret['cam_from_world'].matrix(), [0,0,0,1]))
		focal = query_cam.params[0]
		return model, {
			'num_med_points3D': median_points,
			'query_name': query,
			'query_camera': query_cam,
			'ret': ret,
			'log': log
		}, focal, pose, model.compute_mean_reprojection_error()

	def _forward(self, scene_root, refs, query, ref_poses, ref_ints, query_int, est_opts):
		ref_poses = [to_numpy(p) for p in ref_poses]
		ref_ints = [{'K': to_numpy(i['K']), 'im_size': to_numpy(i['im_size'])} for i in ref_ints]
		query_int = {k: to_numpy(v) for k,v in query_int.items()}

		if est_opts['known_extrinsics']:
			model, result, focal, pose, loss = self.estimate_pose(scene_root, refs, query)
			if model is None: return None, None, None
			self.model = model
			self.scene_root = scene_root
			self.query_name = result['query_name']
			self.query_camera = result['query_camera']
			self.ret = result['ret']
			self.log = result['log']
			
			# Pose alignment
			est_poses = []
			for img in model.images.values():
				pose_w = np.linalg.inv(np.vstack((img.cam_from_world.matrix(), [0, 0, 0, 1])))
				est_poses.append(pose_w)
			gt_poses = [ref_poses[refs.index(img.name)] for img in model.images.values()]
			_, (s, R, t) = align_poses(est_poses, gt_poses)
			
			aligned_pose = np.eye(4)
			est_c2w = np.linalg.inv(pose)
			aligned_pose[:3, :3] = R @ est_c2w[:3, :3]
			aligned_pose[:3, 3] = s*R @ est_c2w[:3, 3] + t

			return focal, aligned_pose, loss

		cam_opts = mapper_opts = None
		if est_opts['known_intrinsics']:
			cam_opts = dict(camera_model='PINHOLE', camera_params=','.join(
				[str(ref_ints[0]['K'][i][j]) for i,j in [(0,0),(1,1),(0,2),(1,2)]]))
			mapper_opts = dict(ba_refine_focal_length=False, ba_refine_extra_params=False)
		
		model, result, focal, pose, loss = self.estimate_pose(scene_root, refs, query, cam_opts, mapper_opts)
		self.model = model
		self.scene_root = scene_root
		self.query_name = result['query_name']
		self.query_camera = result['query_camera']
		self.ret = result['ret']
		self.log = result['log']		

		return (focal, np.linalg.inv(pose), loss) if model else (None,)*3
