import sys
from pathlib import Path
import os
import torch
import torchvision.transforms as tfm
import py3_wget
import numpy as np
import matplotlib.pyplot as plt

from estimator.utils import add_to_path, resize_to_divisible
from estimator import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseEstimator

add_to_path(THIRD_PARTY_DIR.joinpath('duster'))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.cloud_opt.commons import edge_str
import dust3r.cloud_opt.init_im_poses as init_fun

class Dust3rEstimator(BaseEstimator):
	model_path = WEIGHTS_DIR.joinpath("duster_vit_large.pth")
	vit_patch_size = 16

	def __init__(self, device="cpu", use_calib=False, use_lora=False, *args, **kwargs):
		super().__init__(device, **kwargs)
		self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		self.verbose = False
		
		# Initialize training schedule parameters
		self.schedule = 'cosine'
		self.lr = 0.01
		self.niter = 300

		# Load base model weights
		self.download_weights()
		self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path)
		print(f'Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}')

		# Handle LoRA integration before moving to target device
		if use_lora:
			if 'lora_path' not in kwargs:
				raise RuntimeError("Missing required 'lora_path' argument for LoRA integration")
			self._safely_integrate_lora(kwargs['lora_path'], target_device=device)

		# Final device placement and model configuration
		self.model = self.model.to(device)
		self.model.eval()

	# NOTE(gogojjh): support loading another lora params on-the-fly
	def _safely_integrate_lora(self, lora_path: str, target_device: str):
		"""Safely integrates LoRA weights with CPU-based processing to prevent CUDA errors.
		
		Args:
			lora_path: Path to LoRA weights file
			target_device: Original device for the model (preserves device context)
		"""
		from dust3r.lora import LoraLayer, inject_lora
		
		# Store original device and force CPU context
		original_device = next(self.model.parameters()).device
		self.model = self.model.to('cpu')

		# Phase 1: Inject LoRA adapters
		for name, module in self.model.named_modules():
			if any(n in name.split('.') for n in ['qkv']) and isinstance(module, torch.nn.Linear):
				inject_lora(self.model, name, module)

		# Phase 2: Load LoRA weights
		try:
			lora_weights = torch.load(lora_path, map_location='cpu')
			self.model.load_state_dict(lora_weights, strict=False)
			print(f'LoRA Parameters: {sum(v.numel() for v in lora_weights.values()):,}')
		except Exception as e:
			raise RuntimeError(f"LoRA integration failed: {str(e)}")

		# Phase 3: Merge LoRA weights into base model
		for name, module in self.model.named_modules():
			if isinstance(module, LoraLayer):
				parent = self.model
				# Traverse module hierarchy: a.b.c → getattr(a, 'b')
				for component in name.split('.')[:-1]:
					parent = getattr(parent, component)
				
				# Mathematical merge: W' = W + (A*B)*(α/r)
				merged_weight = module.raw_linear.weight + (module.lora_a @ module.lora_b) * module.alpha / module.r
				module.raw_linear.weight.data.copy_(merged_weight)
				
				# Replace composite layer with merged linear layer
				setattr(parent, name.split('.')[-1], module.raw_linear)

		# Restore original device context
		self.model = self.model.to(target_device)

	@staticmethod
	def download_weights():
		url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

		if not os.path.isfile(Dust3rEstimator.model_path):
			print("Downloading Dust3r(ViT large)... (takes a while)")
			py3_wget.download_file(url, Dust3rEstimator.model_path)

	@staticmethod
	def get_edge_str(i, j):
		return edge_str(i, j)

	def preprocess(self, img):
		_, h, w = img.shape
		orig_shape = h, w

		img = resize_to_divisible(img, self.vit_patch_size)
		img = self.normalize(img).unsqueeze(0)
		return img, orig_shape

	def get_similarity(self, option='mean'):
		if option == 'mean':
			edge_scores = {e: float(self.scene.conf_i[e].mean() * self.scene.conf_j[e].mean()) 
						   for e in self.scene.str_edges}
		elif option == 'median':
			edge_scores = {e: float(self.scene.conf_i[e].median() * self.scene.conf_j[e].median()) 
						   for e in self.scene.str_edges}
		return edge_scores

	def get_minimum_spanning_tree(self):
		_, msp_edges, _, _ = init_fun.minimum_spanning_tree(
			self.scene.imshapes, self.scene.edges,
			self.scene.pred_i, self.scene.pred_j, 
			self.scene.conf_i, self.scene.conf_j, 
			self.scene.im_conf, 
			self.scene.min_conf_thr,
			self.scene.device, 
			has_im_poses=self.scene.has_im_poses, 
			verbose=self.scene.verbose
		)
		return msp_edges

	def save_results(self, save_log, scene_root, list_depth_img_name, indice):
		fig0 = self.visualize_weights_errors()
		fig1, avg_depth_error, corr_score = self.visualize_depth_result(scene_root, list_depth_img_name)
		if indice % 5 == 0:
			fig0.savefig(os.path.join(save_log, f"img_weight_error_{indice}.jpg"))
			fig1.savefig(os.path.join(save_log, f"depth_alignment_{indice}.jpg"))
		plt.close(fig0)
		plt.close(fig1)
		return avg_depth_error, corr_score

	def show_reconstruction(self, cam_size=0.2):
		self.scene.show(cam_size=cam_size)

	def set_calib_params(self, new_calib_params):
		self.calib_params = new_calib_params

	def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts):
		"""
		Performs the forward pass of the pose estimation model.

		Args:
			scene_root (str): The root directory of the scene.
			list_img0_name (list): A list of image names for the reference images.
			img1_name (str): The name of the target image.
			list_img0_poses (list): A list of poses for the reference images.
			list_img0_intr (list): A list of intrinsic camera matrices for the reference images.
			img1_intr (dict): The intrinsic camera matrix for the target image.
			est_opts (dict): Additional options for the pose estimation.

		Returns:
			tuple: A tuple containing the estimated focal length, estimated image pose, and the loss value.
		"""

		resize = est_opts.get('resize', 512)
		imgs_path = [str(scene_root / img_name) for img_name in list_img0_name] + [str(scene_root / img1_name)]
		images = load_images(imgs_path, size=resize, verbose=self.verbose)
		pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
		assert len(imgs_path) == len(images)

		############## 
		# At this stage, you have the raw dust3r predictions
		# here, view1, pred1, view2, pred2 are dicts of lists of len(2)
		#  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
		# in each view you have:
		# an integer image identifier: view1['idx'] and view2['idx']
		# the img: view1['img'] and view2['img']
		# the image shape: view1['true_shape'] and view2['true_shape']
		# an instance string output by the dataloader: view1['instance'] and view2['instance']
		# pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
		# pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
		# pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']
		# next we'll use the global_aligner to align the predictions
		# depending on your task, you may be fine with the raw output and not need it
		# with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
		# if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
		############## 
		# Summary: Keys of output, view1, pred1, view2, pred2
		#   output['view1', 'view2', 'pred1', 'pred2', 'loss'])
		#   view1['img', 'true_shape', 'idx', 'instance'])
		#   view2['img', 'true_shape', 'idx', 'instance'])
		#   pred1['pts3d', 'conf', 'desc', 'desc_conf'])
		#   pred2['conf', 'desc', 'desc_conf', 'pts3d_in_other_view'])
		output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

		##### GlobalAlignerMode.PointCloudOptimizer
		scene = global_aligner(
			output, 
			device=self.device, 
			mode=GlobalAlignerMode.ModularPointCloudOptimizer, 
			verbose=self.verbose,
			conf='log',
			calib_params=self.calib_params
		)

		if est_opts['known_extrinsics']:
			known_poses = [pose for pose in list_img0_poses] + [np.eye(4)]
			scene.preset_pose(known_poses=known_poses, pose_msk=[True] * len(list_img0_poses) + [False])
			
		if est_opts['known_intrinsics']:
			list_img_intr = list_img0_intr + [img1_intr]
			resize_list_img_K = []
			for idx, image in enumerate(images):
				ori_K = list_img_intr[idx]['K']
				ori_im_size = list_img_intr[idx]['im_size']                    # WxH
				new_im_size = torch.from_numpy(image['true_shape']).squeeze(0) # HxW
				scale_w = new_im_size[1] / ori_im_size[0]
				scale_h = new_im_size[0] / ori_im_size[1]
				new_K = torch.zeros_like(ori_K)
				new_K[0, 0] = ori_K[0, 0] * scale_w  # Focal length X
				new_K[1, 1] = ori_K[1, 1] * scale_h  # Focal length Y
				new_K[0, 2] = ori_K[0, 2] * scale_w  # Principal point X
				new_K[1, 2] = ori_K[1, 2] * scale_h  # Principal point Y
				resize_list_img_K.append(new_K)
			scene.preset_intrinsics(resize_list_img_K)

		##### Perform optimization
		loss = scene.compute_global_alignment(
			init="mst", 
			niter=self.niter, 
			schedule=self.schedule, 
			lr=self.lr
		)

		##### Get results
		self.scene = scene
		focals, im_poses = scene.get_focals(), scene.get_im_poses()
		est_focal, est_im_pose = focals[-1], im_poses[-1]
		return est_focal.detach(), est_im_pose.detach(), loss

	def evaluate_correlation(self, values1, values2):
		from scipy import stats
		valid_mask = np.isfinite(values1) & np.isfinite(values2)
		valid_values1 = values1[valid_mask]
		valid_values2 = values2[valid_mask]
		# Spearman Rank Correlation (measures monotonic relationship)
		# -1 = perfect inverse relationship
		spearman_corr, spearman_p = stats.spearmanr(valid_values2, valid_values1)
		# print(f"Spearman Correlation: {spearman_corr:.3f}")
		return spearman_corr

	@torch.no_grad()
	def visualize_depth_result(self, scene_root, list_depth_img_name):
		import numpy as np
		from PIL import Image
		from matplotlib.gridspec import GridSpec
		import torch

		depth_scale = 1000.0
		max_depth = 15.0

		gt_depth_imgs = []
		imgs_path = [str(scene_root / img_name) for img_name in list_depth_img_name]		
		for path in imgs_path:
			with Image.open(path) as img:
				gt_depth = np.array(img).astype(np.float32) / depth_scale  # mm -> m
				gt_depth[gt_depth > max_depth] = 0.0
			gt_depth_imgs.append(gt_depth)

		pred_depth_imgs = self.scene.get_depthmaps()
		assert len(gt_depth_imgs) == len(pred_depth_imgs)

		# Align depth map
		aligned_depths, scale_factors, error_maps, weight_maps, conf_maps = [], [], [], [], []
		for idx, (gt_depth, pred_depth) in enumerate(zip(gt_depth_imgs, pred_depth_imgs)):
			pred_depth_np = pred_depth.cpu().numpy() if torch.is_tensor(pred_depth) else pred_depth
			
			ratios = []
			scale_y = gt_depth.shape[0] / pred_depth_np.shape[0] 
			scale_x = gt_depth.shape[1] / pred_depth_np.shape[1] 
			for y in range(pred_depth_np.shape[0]):
				for x in range(pred_depth_np.shape[1]):
					if pred_depth_np[y][x] > 0.0:
						re_y = int(np.floor(y * scale_y))
						re_x = int(np.floor(x * scale_x))
						if gt_depth[re_y][re_x] > 0.0:
							ratios.append(gt_depth[re_y][re_x] / pred_depth_np[y][x])
			ratios = np.array(ratios)
			scale_factor = np.median(ratios[np.isfinite(ratios)])
			scale_factors.append(scale_factor)
			
			error_map = np.zeros_like(pred_depth_np)
			aligned_depth = pred_depth_np * scale_factor
			aligned_depths.append(aligned_depth)
			for y in range(aligned_depth.shape[0]):
				for x in range(aligned_depth.shape[1]):
					if aligned_depth[y][x] > 0:
						re_y = int(np.floor(y * scale_y))
						re_x = int(np.floor(x * scale_x))
						if gt_depth[re_y][re_x] > 0:
							error_map[y][x] = np.abs(gt_depth[re_y][re_x] - aligned_depth[y][x])
			error_maps.append(error_map)
			
			# Handle weight/confidence maps using dictionary key pattern
			if idx == 0:
				if self.scene.weight_i['0_1'].mean() > self.scene.weight_j['1_0'].mean():
					weight_maps.append(self.scene.weight_i['0_1'].cpu().numpy())
					conf_maps.append(self.scene.conf_trf(self.scene.conf_i['0_1']).cpu().numpy())
				else:
					weight_maps.append(self.scene.weight_j['1_0'].cpu().numpy())
					conf_maps.append(self.scene.conf_trf(self.scene.conf_j['1_0']).cpu().numpy())
			else:
				key1, key2 = f"{0}_{idx}", f"{idx}_{0}"
				if self.scene.weight_i[key2].mean() > self.scene.weight_j[key1].mean():
					weight_maps.append(self.scene.weight_i[key2].cpu().numpy())
					conf_maps.append(self.scene.conf_trf(self.scene.conf_i[key2]).cpu().numpy())
				else:
					weight_maps.append(self.scene.weight_j[key1].cpu().numpy())
					conf_maps.append(self.scene.conf_trf(self.scene.conf_j[key1]).cpu().numpy())					

		# Use GridSpec for flexible subplot layout
		num_imgs = len(gt_depth_imgs)
		fig = plt.figure(figsize=(24, 5 * (num_imgs + 2)))  # Adjust height for additional plots
		gs = GridSpec(num_imgs + 1, 5)  # +2 rows for the two additional plots
		# Plot the original images
		for i in range(num_imgs):
			# GT Depth
			ax = fig.add_subplot(gs[i, 0])
			im = ax.imshow(gt_depth_imgs[i], cmap='jet', vmin=0, vmax=10)
			plt.colorbar(im, ax=ax)
			ax.set_title(f'GT Depth {i}')
			
			# Aligned Depth
			ax = fig.add_subplot(gs[i, 1])
			im = ax.imshow(aligned_depths[i], cmap='jet', vmin=0, vmax=10)
			plt.colorbar(im, ax=ax)
			ax.set_title(f'Aligned Depth {i} (Scale: {scale_factors[i]:.2f})')

			# Error Map
			ax = fig.add_subplot(gs[i, 2])
			im = ax.imshow(error_maps[i], cmap='jet', vmin=0, vmax=1)
			plt.colorbar(im, ax=ax)
			ax.set_title(f'Error Map {i}')

			# Raw Conf Map
			ax = fig.add_subplot(gs[i, 3])
			im = ax.imshow(conf_maps[i], cmap='jet')
			plt.colorbar(im, ax=ax)
			ax.set_title(f'Raw Conf Map {i}')

			# Calibrate Conf Map
			ax = fig.add_subplot(gs[i, 4])
			im = ax.imshow(weight_maps[i], cmap='jet')
			plt.colorbar(im, ax=ax)
			ax.set_title(f'Calibrated conf Map {i}')

		# Compute flattened arrays
		all_errors = np.concatenate([m.flatten() for m in error_maps])
		all_weights = np.concatenate([m.flatten() for m in weight_maps])
		all_confs = np.concatenate([m.flatten() for m in conf_maps])
		# print(f'Size of error items: {len(all_errors)}')
		
		cor0 = self.evaluate_correlation(all_confs, all_errors)
		print(f'Evaluate Relation: errors - raw_conf - Spearman Correlation: {cor0:.5f}')
		cor1 = self.evaluate_correlation(all_weights, all_errors)
		print(f'Evaluate Relation: errors - calibrated_conf - Spearman Correlation: {cor1:.5f}')

		ax = fig.add_subplot(gs[num_imgs, :2])  # Span all columns in row `num_imgs + 1`
		ax.plot(all_confs, all_errors, '.', markersize=3)
		ax.set_xlabel('Raw Confidence')
		ax.set_ylabel('Depth Errors')
		ax.set_title(f'Error vs. Raw Confidence {cor0:.5f}')

		ax = fig.add_subplot(gs[num_imgs, 2:4])  # Span all columns in row `num_imgs`
		ax.plot(all_weights, all_errors, '.', markersize=3)
		ax.set_xlabel('Calibrated Confidence')
		ax.set_ylabel('Depth Errors')
		ax.set_title(f'Error vs. Calibrated Confidence {cor1:.5f}')

		plt.tight_layout()
		return fig, np.mean(all_errors), cor1
			
	@torch.no_grad()
	def visualize_weights_errors(self):
		from dust3r.utils.geometry import geotrf
		from dust3r.utils.device import to_numpy

		if self.calib_params is not None:
			mu = self.calib_params['mu']
			pseudo_gt_thre = self.calib_params['pseudo_gt_thre']
		else:
			mu, pseudo_gt_thre = 1.0, 1.5			
		max_error, max_conf = 0.125, 2.8

		edge_str_key = f"{0}_{1}"
		i, j = map(int, edge_str_key.split('_')) # default: 0, 1
		pw_poses = self.scene.get_pw_poses()  # cam-to-world
		pw_adapt = self.scene.get_adaptors()
		proj_pts3d = self.scene.get_pts3d()  # optimized point in the global coordinate

		# Compute error maps
		aligned_pred_j = geotrf(pw_poses[self.scene.str_edges.index(edge_str_key)], pw_adapt[self.scene.str_edges.index(edge_str_key)] * self.scene.pred_j[edge_str_key])
		res_j = proj_pts3d[j] - aligned_pred_j
		error_map_j = self.scene.dist(res_j, torch.zeros_like(res_j), torch.ones_like(res_j)[:,:,1].squeeze()).cpu().numpy()

		aligned_pred_i = geotrf(pw_poses[self.scene.str_edges.index(edge_str_key)], pw_adapt[self.scene.str_edges.index(edge_str_key)] * self.scene.pred_i[edge_str_key])
		res_i = proj_pts3d[i] - aligned_pred_i
		error_map_i = self.scene.dist(res_i, torch.zeros_like(res_i), torch.ones_like(res_i)[:,:,1].squeeze()).cpu().numpy()

		# Fetch raw images, confidence maps, and weights
		raw_image_i = to_numpy(self.scene.imgs[i])
		if np.issubdtype(raw_image_i.dtype, np.floating):
			raw_image_i = np.uint8(255*raw_image_i.clip(min=0, max=1))
		raw_image_j = to_numpy(self.scene.imgs[j])
		if np.issubdtype(raw_image_j.dtype, np.floating):
			raw_image_j = np.uint8(255*raw_image_j.clip(min=0, max=1))
		
		if self.scene.conf_i["0_1"].mean() >  self.scene.conf_j["1_0"].mean():
			C_i = self.scene.conf_trf(self.scene.conf_i["0_1"]).cpu().numpy()
			C_j = self.scene.conf_trf(self.scene.conf_j["0_1"]).cpu().numpy()
		else:
			C_i = self.scene.conf_trf(self.scene.conf_j["1_0"]).cpu().numpy()
			C_j = self.scene.conf_trf(self.scene.conf_i["1_0"]).cpu().numpy()			
		# C_i = self.scene.conf_trf(self.scene.conf_i["0_1"]).cpu().numpy()
		# C_j = self.scene.conf_trf(self.scene.conf_j["0_1"]).cpu().numpy()
		weight_i = C_i / (1 + error_map_i / mu)**2
		weight_j = C_j / (1 + error_map_j / mu)**2

		mask_i = weight_i < pseudo_gt_thre
		mask_j = weight_j < pseudo_gt_thre

		# Extract values for the ith and jth maps
		random_coords_i = np.array([[30, 470], [80, 400]])
		C_i_values = np.array([C_i[y, x] for y, x in random_coords_i])
		weight_i_values = np.array([weight_i[y, x] for y, x in random_coords_i])
		error_i_values = np.array([error_map_i[y, x] for y, x in random_coords_i])

		random_coords_j = np.array([[150, 300], [100, 400]])
		C_j_values = np.array([C_j[y, x] for y, x in random_coords_j])
		weight_j_values = np.array([weight_j[y, x] for y, x in random_coords_j])
		error_j_values = np.array([error_map_j[y, x] for y, x in random_coords_j])

		opt_depthmaps = self.scene.get_depthmaps()

		# Create figure
		fig, axes = plt.subplots(2, 7, figsize=(24, 8))

		# Plot ith row
		axes[0, 0].imshow(raw_image_i, cmap='gray')
		axes[0, 0].set_title(f'Raw Image {i}')
		for idx in range(len(random_coords_i)):
			axes[0, 0].plot([random_coords_i[idx][1]], [random_coords_i[idx][0]], color='r', marker='x', markersize=10)

		raw_image_i[mask_i] = [0, 0, 0]
		axes[0, 1].imshow(raw_image_i, cmap='gray')
		axes[0, 1].set_title(f'Filtered Image {i}')

		im = axes[0, 2].imshow(C_i, cmap='jet')
		axes[0, 2].set_title(f'Confidence Map {i}')
		fig.colorbar(im, ax=axes[0, 2])
		
		im = axes[0, 3].imshow(weight_i, cmap='jet')
		axes[0, 3].set_title(f'Weight {i}')
		fig.colorbar(im, ax=axes[0, 3])
		
		im = axes[0, 4].imshow(error_map_i, cmap='jet')
		axes[0, 4].set_title(f'Error Map {i}')
		fig.colorbar(im, ax=axes[0, 4])
		
		im = axes[0, 5].imshow(opt_depthmaps[i].detach().cpu().numpy(), cmap='jet')
		axes[0, 5].set_title(f'Depth Map {i}')
		fig.colorbar(im, ax=axes[0, 5], shrink=0.8)
		
		for idx in range(len(random_coords_i)):
			axes[0, 6].plot([C_i_values[idx]], [error_i_values[idx]], color='r', marker='x')
			axes[0, 6].plot([weight_i_values[idx]], [error_i_values[idx]], color='k', marker='x')
		axes[0, 6].set_xlim(-0.3, max_conf)
		axes[0, 6].set_ylim(-0.01, max_error)

		# Plot jth row
		axes[1, 0].imshow(raw_image_j, cmap='gray')
		axes[1, 0].set_title(f'Raw Image {j}')
		for idx in range(len(random_coords_j)):
			axes[1, 0].plot([random_coords_j[idx][1]], [random_coords_j[idx][0]], color='r', marker='x', markersize=10)

		raw_image_j[mask_j] = [0, 0, 0]
		axes[1, 1].imshow(raw_image_j, cmap='gray')
		axes[1, 1].set_title(f'Filtered Image {j}')

		im = axes[1, 2].imshow(C_j, cmap='jet')
		axes[1, 2].set_title(f'Confidence Map {j}')
		fig.colorbar(im, ax=axes[1, 2])
		
		im = axes[1, 3].imshow(weight_j, cmap='jet')
		axes[1, 3].set_title(f'Weight {j}')
		fig.colorbar(im, ax=axes[1, 3])
		
		im = axes[1, 4].imshow(error_map_j, cmap='jet')
		axes[1, 4].set_title(f'Error Map {j}')
		fig.colorbar(im, ax=axes[1, 4])
		
		im = axes[1, 5].imshow(opt_depthmaps[j].detach().cpu().numpy(), cmap='jet')
		axes[1, 5].set_title(f'Depth Map {j}')
		fig.colorbar(im, ax=axes[1, 5], shrink=0.8)
		
		for idx in range(len(random_coords_j)):
			axes[1, 6].plot([C_j_values[idx]], [error_j_values[idx]], color='r', marker='x')
			axes[1, 6].plot([weight_j_values[idx]], [error_j_values[idx]], color='k', marker='x')
		axes[1, 6].set_xlim(-0.3, max_conf)
		axes[1, 6].set_ylim(-0.01, max_error)			

		plt.tight_layout()
		return fig