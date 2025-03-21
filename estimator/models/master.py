import os
import torch
import torchvision.transforms as tfm
import py3_wget
import numpy as np

from estimator import BaseEstimator, WEIGHTS_DIR, THIRD_PARTY_DIR
from estimator.utils import resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("mast3r"))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.cloud_opt.commons import edge_str
from dust3r.lora import LoraLayer, inject_lora
import dust3r.cloud_opt.init_im_poses as init_fun


class Mast3rEstimator(BaseEstimator):
    """Estimator class for MASt3R model."""
    model_path = WEIGHTS_DIR.joinpath("MASt3r_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    vit_patch_size = 16

    def __init__(self, device="cpu", use_calib=False, use_lora=False, *args, **kwargs):
        """Initializes the Mast3rEstimator.

        Args:
            device (str): Device to run the model on.
            use_calib (bool): Whether to use calibration.
            use_lora (bool): Whether to use LoRA.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.verbose = False

        self.schedule = 'cosine'
        self.lr = 0.01
        self.niter = 300

        self.download_weights()
        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(device)
        print(f'Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}')

        # Set default calib_params
        if use_calib:
            self.set_calib_params(dict(mu=1.0, conf_thre=0.5, pseudo_gt_thre=1.5, use_weight_opt=False))
        else:
            self.set_calib_params(None)

        # Handle LoRA integration before moving to target device
        if use_lora:
            if 'lora_path' not in kwargs:
                raise RuntimeError("Missing required 'lora_path' argument for LoRA integration")
            self._safely_integrate_lora(kwargs['lora_path'], target_device=device)

        # Final device placement and model configuration
        self.model = self.model.to(device)
        self.model.eval()

    def set_calib_params(self, new_calib_params):
        """Sets the calibration parameters.

        Args:
            new_calib_params (dict): New calibration parameters.
        """
        self.calib_params = new_calib_params

    # Loading another lora params on-the-fly
    def _safely_integrate_lora(self, lora_path: str, target_device: str):
        """Safely integrates LoRA weights with CPU-based processing to prevent CUDA errors.

        Args:
            lora_path (str): Path to LoRA weights file.
            target_device (str): Original device for the model (preserves device context).
        """
        # Store original device and force CPU context
        # original_device = next(self.model.parameters()).device
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
                lora_weight = ((module.lora_a @ module.lora_b) * module.alpha / module.r).T
                merged_weight = module.raw_linear.weight + lora_weight
                module.raw_linear.weight.data.copy_(merged_weight)

                # Replace composite layer with merged linear layer
                setattr(parent, name.split('.')[-1], module.raw_linear)

    @staticmethod
    def download_weights():
        """Downloads the weights for the MASt3R model."""
        url = ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
               "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")

        if not os.path.isfile(Mast3rEstimator.model_path):
            print("Downloading Master(ViT large)... (takes a while)")
            py3_wget.download_file(url, Mast3rEstimator.model_path)

    @staticmethod
    def get_edge_str(i, j):
        """Gets the edge string.

        Args:
            i (int): First node.
            j (int): Second node.

        Returns:
            str: Edge string.
        """
        return edge_str(i, j)

    def preprocess(self, img):
        """Preprocesses the image.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            tuple: Preprocessed image and original shape.
        """
        _, h, w = img.shape
        orig_shape = h, w

        img = resize_to_divisible(img, self.vit_patch_size)
        img = self.normalize(img).unsqueeze(0)
        return img, orig_shape

    def get_similarity(self, option='mean'):
        """Gets the similarity scores.

        Args:
            option (str): Option for similarity calculation.

        Returns:
            dict: Similarity scores.
        """
        if option == 'mean':
            edge_scores = {e: float(self.scene.conf_i[e].mean() * self.scene.conf_j[e].mean())
                           for e in self.scene.str_edges}
        elif option == 'median':
            edge_scores = {e: float(self.scene.conf_i[e].median() * self.scene.conf_j[e].median())
                           for e in self.scene.str_edges}
        return edge_scores

    def get_minimum_spanning_tree(self):
        """Gets the minimum spanning tree.

        Returns:
            list: Minimum spanning tree edges.
        """
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
        """Shows the reconstruction.

        Args:
            cam_size (tuple): Camera size.
        """
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
        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=self.device, dist="dot", block_size=2**13
        )

        # ignore small border around the edge
        H0, W0 = view1["true_shape"][0]
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3)
            & (matches_im0[:, 0] < int(W0) - 3)
            & (matches_im0[:, 1] >= 3)
            & (matches_im0[:, 1] < int(H0) - 3)
        )

        H1, W1 = view2["true_shape"][0]
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3)
            & (matches_im1[:, 0] < int(W1) - 3)
            & (matches_im1[:, 1] >= 3)
            & (matches_im1[:, 1] < int(H1) - 3)
        )

        valid_matches = valid_matches_im0 & valid_matches_im1
        mkpts0, mkpts1 = matches_im0[valid_matches], matches_im1[valid_matches]
        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None

    def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts):
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
        resize = est_opts.get('resize', 512)
        imgs_path = [str(scene_root / img_name) for img_name in list_img0_name]
        if img1_name is not None:
            imgs_path.append(str(scene_root / img1_name))
        images = load_images(imgs_path, size=resize, verbose=self.verbose)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        assert len(imgs_path) == len(images)

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
        # Summary: Keys of output, view1, pred1, view2, pred2
        #   output['view1', 'view2', 'pred1', 'pred2', 'loss'])
        #   view1['img', 'true_shape', 'idx', 'instance'])
        #   view2['img', 'true_shape', 'idx', 'instance'])
        #   pred1['pts3d', 'conf', 'desc', 'desc_conf'])
        #   pred2['conf', 'desc', 'desc_conf', 'pts3d_in_other_view'])
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        # GlobalAlignerMode.PairViewer
        if len(imgs_path) == 2:
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.PairViewer,
                conf='log',
                verbose=self.verbose
            )
            loss = 0.0
        # GlobalAlignerMode.PointCloudOptimizer
        else:
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.ModularPointCloudOptimizer,
                verbose=self.verbose,
                conf='log',
                calib_params=self.calib_params
            )

            if est_opts['known_extrinsics']:
                known_poses = list_img0_poses.copy()
                pose_msk = [True] * len(list_img0_poses)

                if img1_name is not None:
                    known_poses.append(np.eye(4))
                    pose_msk.append(False)

                scene.preset_pose(known_poses=known_poses, pose_msk=pose_msk)

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

            # Perform optimization
            loss = scene.compute_global_alignment(
                init="mst",
                niter=self.niter,
                schedule=self.schedule,
                lr=self.lr
            )

        # Get results
        self.scene = scene
        focals, im_poses = scene.get_focals(), scene.get_im_poses()
        est_focal = focals[-1] if img1_name is not None else None
        est_im_pose = im_poses[-1] if img1_name is not None else None
        return est_focal.detach() if est_focal is not None else None, \
            est_im_pose.detach() if est_im_pose is not None else None, \
            loss

    def save_results(self):
        """Saves the results (not implemented)."""
        pass