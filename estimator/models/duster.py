import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tfm
import py3_wget

from pathlib import Path

from estimator.utils import add_to_path, resize_to_divisible, align_poses, to_numpy, to_tensor
from estimator import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseEstimator

add_to_path(THIRD_PARTY_DIR.joinpath('duster'))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.cloud_opt.commons import edge_str
import dust3r.cloud_opt.init_im_poses as init_fun
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.cloud_opt.base_opt import global_alignment_loop

from typing import Union, List, Tuple, Dict
from pathlib import Path

class Dust3rEstimator(BaseEstimator):
    model_path = WEIGHTS_DIR.joinpath("duster_vit_large.pth")
    vit_patch_size = 16

    def __init__(self, device="cpu", use_calib=False, *args, **kwargs):
        """Initializes the Dust3rEstimator.

        Args:
            device (str): Device to run the model on.
            model_args (str): Model arguments.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

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

        # Set default calib_params
        if use_calib:
            print('Use calibrated confidence map and weight optimization')
            calib_params = dict(mu=5.0, conf_thre=0.5, pseudo_gt_thre=0.0, use_weight_opt=True, use_alt_opt=False)
            self.set_calib_params(calib_params)
        else:
            self.set_calib_params(None)

        self.transform = tfm.Compose([])

        # Final device placement and model configuration
        self.model = self.model.to(device)
        self.model.eval()

    def set_calib_params(self, new_calib_params):
        self.calib_params = new_calib_params

    ##### NOTE(gogojjh): Not used for now
    # # Loading another lora params on-the-fly
    # def _safely_integrate_lora(self, lora_path: str, target_device: str):
    #     """Safely integrates LoRA weights with CPU-based processing to prevent CUDA errors.

    #     Args:
    #         lora_path: Path to LoRA weights file
    #         target_device: Original device for the model (preserves device context)
    #     """
    #     # Store original device and force CPU context
    #     # original_device = next(self.model.parameters()).device
    #     self.model = self.model.to('cpu')

    #     # Phase 1: Inject LoRA adapters
    #     for name, module in self.model.named_modules():
    #         if any(n in name.split('.') for n in ['qkv']) and isinstance(module, torch.nn.Linear):
    #             inject_lora(self.model, name, module)

    #     # Phase 2: Load LoRA weights
    #     try:
    #         lora_weights = torch.load(lora_path, map_location='cpu')
    #         self.model.load_state_dict(lora_weights, strict=False)
    #         print(f'LoRA Parameters: {sum(v.numel() for v in lora_weights.values()):,}')
    #     except Exception as e:
    #         raise RuntimeError(f"LoRA integration failed: {str(e)}")

    #     # Phase 3: Merge LoRA weights into base model
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, LoraLayer):
    #             parent = self.model
    #             # Traverse module hierarchy: a.b.c → getattr(a, 'b')
    #             for component in name.split('.')[:-1]:
    #                 parent = getattr(parent, component)

    #             # Mathematical merge: W' = W + (A*B)*(α/r)
    #             lora_weight = ((module.lora_a @ module.lora_b) * module.alpha / module.r).T
    #             merged_weight = module.raw_linear.weight + lora_weight
    #             module.raw_linear.weight.data.copy_(merged_weight)

    #             # Replace composite layer with merged linear layer
    #             setattr(parent, name.split('.')[-1], module.raw_linear)

    @staticmethod
    def download_weights():
        url = ("https://download.europe.naverlabs.com/ComputerVision/DUSt3R/"
               "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

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

    def show_reconstruction(self, cam_size=None):
        self.scene.show() if cam_size is None else self.scene.show(cam_size=cam_size)

    def _get_matched_kpts(self, img0, img1):
        """Gets the matched keypoints.

        Args:
            img0 (torch.Tensor): First image tensor.
            img1 (torch.Tensor): Second image tensor.

        Returns:
            tuple: Matched keypoints and other related data.
        """
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        img_pair = [
            {"img": img0, "idx": 0, "instance": 0, "true_shape": np.int32([img0.shape[-2:]])},
            {"img": img1, "idx": 1, "instance": 1, "true_shape": np.int32([img1.shape[-2:]])},
        ]
        output = inference([tuple(img_pair)], self.model, self.device, batch_size=1, verbose=False)

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PairViewer,
            conf='log',
            verbose=self.verbose
        )
        confidence_masks = scene.get_masks()
        pts3d = scene.get_pts3d()
        imgs = scene.imgs

        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, _ = find_reciprocal_matches(*pts3d_list)

        mkpts1 = pts2d_list[1][reciprocal_in_P2]
        mkpts0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None

    def _forward(
        self, 
        scene_root: Path,
        list_img0: Union[List[torch.Tensor], List[str], List[Path]], 
        img1: Union[List[torch.Tensor], List[str], List[Path]], 
        list_img0_poses: Union[List[torch.Tensor]], 
        list_img0_intr: Union[List[torch.Tensor]], 
        img1_intr: Union[torch.Tensor],
        est_opts: Dict
    ):
        """Performs the forward pass of the pose estimation model.

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
        self.niter = est_opts.get('niter', self.niter)
        num_img_input = len(list_img0) + 1

        # Load database images
        if isinstance(list_img0[0], torch.Tensor) and isinstance(img1, torch.Tensor):
            images = []
            for img in list_img0:
                image = {
                    "img": self.preprocess(img)[0], 
                    "idx": len(images), 
                    "instance": str(len(images)), 
                    "true_shape": np.int32([img.shape[-2:]])
                }
                images.append(image)
    
            images.append({
                "img": self.preprocess(img1)[0], 
                "idx": len(images), 
                "instance": str(len(images)), 
                "true_shape": np.int32([img.shape[-2:]])
            })
        else:
            resize = est_opts.get('resize', 512)
            imgs_path = [str(scene_root / img_name) for img_name in list_img0 + [img1]]
            images = load_images(imgs_path, size=resize, verbose=self.verbose)

        # Preprocess images (convert into gray image)
        for image in images: 
            image['img'] = self.transform(image['img'])

        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        assert num_img_input == len(images)

        # NOTE(gogojjh): At this stage, you have the raw dust3r predictions
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
        # Summary: Keys of output, view1, pred1, view2, pred2
        #   output['view1', 'view2', 'pred1', 'pred2', 'loss'])
        #   view1['img', 'true_shape', 'idx', 'instance'])
        #   view2['img', 'true_shape', 'idx', 'instance'])
        #   pred1['pts3d', 'conf', 'desc', 'desc_conf'])
        #   pred2['conf', 'desc', 'desc_conf', 'pts3d_in_other_view'])
        
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)
        self.output_inference = output

        # GlobalAlignerMode.PairViewer
        if num_img_input == 2:
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.PairViewer,
                conf='log',
                verbose=self.verbose
            )
            loss = 0.0
            return None, None, loss
        # GlobalAlignerMode.PointCloudOptimizer
        else:
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.ModularPointCloudOptimizer,
                verbose=self.verbose,
                dist='l1',
                conf='log',
                calib_params=self.calib_params
            )

            if est_opts['known_intrinsics']:
                list_img_intr = list_img0_intr.copy()
                if img1_intr is not None:
                    list_img_intr.append(img1_intr)

                resize_list_img_K = []
                for idx, image in enumerate(images):
                    ori_K = list_img_intr[idx]['K']
                    ori_im_size = list_img_intr[idx]['im_size']
                    new_im_size = torch.from_numpy(image['true_shape']).squeeze(0)
                    scale_w = new_im_size[1] / ori_im_size[0]
                    scale_h = new_im_size[0] / ori_im_size[1]
                    new_K = torch.zeros_like(ori_K)
                    new_K[0, 0] = ori_K[0, 0] * scale_w  # Focal length X
                    new_K[1, 1] = ori_K[1, 1] * scale_h  # Focal length Y
                    new_K[0, 2] = ori_K[0, 2] * scale_w  # Principal point X
                    new_K[1, 2] = ori_K[1, 2] * scale_h  # Principal point Y
                    resize_list_img_K.append(new_K)

                scene.preset_intrinsics(resize_list_img_K)

            if est_opts['known_extrinsics']:
                known_poses = list_img0_poses.copy() + [torch.eye(4)]
                pose_msk = [True] * len(list_img0_poses) + [False]
                scene.preset_pose(known_poses=known_poses, pose_msk=pose_msk)

            # Perform optimization
            loss = scene.compute_global_alignment(
                init="mst",
                niter=self.niter,
                schedule=self.schedule,
                lr=self.lr
            )

            ###########################
            # Perform two-stage optimization to adjust the noisy poses
            if est_opts['known_extrinsics']:
                if 'two_stage_opt_niter' in est_opts:
                    im_poses_first = [im_pose.clone() for im_pose in scene.get_im_poses()]
                    for pose_param in scene.im_poses:
                        pose_param.requires_grad_(True)
                    loss = global_alignment_loop(scene, niter=est_opts['two_stage_opt_niter'], schedule=self.schedule, lr=self.lr)
                    im_poses_second = scene.get_im_poses()
                    for idx in range(len(im_poses_second)):
                        diff = np.linalg.norm(im_poses_second[idx].detach().cpu().numpy()[:3, 3].T - im_poses_first[idx].detach().cpu().numpy()[:3, 3].T)
                        print(f"{idx}: {im_poses_first[idx].detach().cpu().numpy()[:3, 3].T} -> {im_poses_second[idx].detach().cpu().numpy()[:3, 3].T} (diff: {diff:.3f} m)")
            ###########################

            self.scene = scene           
            focals, im_poses = scene.get_focals(), scene.get_im_poses()
            est_focal = focals[-1].detach()
            est_im_pose = im_poses[-1].detach()
            
            return est_focal, est_im_pose, loss
                       
    def save_results(self, save_log, scene_root, list_depth_img_name, indice):
        fig0 = self.visualize_weights_errors()
        fig1, avg_depth_error, corr_score = self.visualize_depth_result(scene_root, list_depth_img_name)
        if indice % 5 == 0:
            fig0.savefig(os.path.join(save_log, f"img_weight_error_{indice}.jpg"))
            fig1.savefig(os.path.join(save_log, f"depth_alignment_{indice}.jpg"))
        plt.close(fig0)
        plt.close(fig1)

        return avg_depth_error, corr_score

