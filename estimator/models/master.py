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

class Mast3rEstimator(BaseEstimator):
    model_path = WEIGHTS_DIR.joinpath("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    vit_patch_size = 16

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.min_conf_thr = 0.0
        self.verbose = False

        self.schedule = 'cosine'
        self.lr = 0.01
        self.niter = 600

        self.download_weights()
        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

        if not os.path.isfile(Mast3rEstimator.model_path):
            print("Downloading Master(ViT large)... (takes a while)")
            py3_wget.download_file(url, Mast3rEstimator.model_path)

    def save_results(self):
        pass

    def show_reconstruction(self, cam_size=None):
        self.scene.show() if cam_size is None else self.scene.show(cam_size=cam_size)

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

    def _get_matched_kpts(self, img0, img1):
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

    def _forward(self, 
                 scene_root, 
                 list_img0_name: list, 
                 img1_name, 
                 list_img0_poses, 
                 list_img0_intr, 
                 img1_intr, 
                 est_opts):
        """
        Performs the forward pass of the pose estimation model.

        Args:
            scene_root (str | Path): The root directory of the scene.
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
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        ##### GlobalAlignerMode.PairViewer
        if len(imgs_path) == 2:
            scene = global_aligner(
                output,
                device=self.device, 
                mode=GlobalAlignerMode.PairViewer, 
                verbose=self.verbose
            )
            loss = 0.0
        ##### GlobalAlignerMode.PointCloudOptimizer
        else:
            scene = global_aligner(
                output,
                device=self.device, 
                mode=GlobalAlignerMode.ModularPointCloudOptimizer, 
                verbose=self.verbose
            )
            if est_opts['known_extrinsics']:
                known_poses = list_img0_poses + [torch.eye(4)]
                pose_msk = [True] * len(list_img0_poses) + [False]
                scene.preset_pose(known_poses=known_poses, pose_msk=pose_msk)

            ###################################
            ##### TODO(gogojjh):
            # if list_img0_K is not None:
            #     scene.preset_intrinsics(list_img0_K + [img1_K])        
            ####################################
                
            loss = scene.compute_global_alignment(init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr)

        ##### Get results
        focals, im_poses = scene.get_focals(), scene.get_im_poses()
        est_focal, est_im_pose = focals[-1], im_poses[-1]

        self.scene = scene
        return est_focal.detach(), est_im_pose.detach(), loss
