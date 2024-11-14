import sys
from pathlib import Path
import os
import torchvision.transforms as tfm
import py3_wget
import numpy as np

from estimator.utils import add_to_path, resize_to_divisible
from estimator import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseEstimator

add_to_path(THIRD_PARTY_DIR.joinpath('duster'))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid


class Dust3rEstimator(BaseEstimator):
    model_path = WEIGHTS_DIR.joinpath("duster_vit_large.pth")
    vit_patch_size = 16

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.verbose = False

        self.schedule = 'cosine'
        self.lr = 0.01
        self.niter = 300

        self.download_weights()
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

        if not os.path.isfile(Dust3rEstimator.model_path):
            print("Downloading Dust3r(ViT large)... (takes a while)")
            py3_wget.download_file(url, Dust3rEstimator.model_path)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        img = resize_to_divisible(img, self.vit_patch_size)
        img = self.normalize(img).unsqueeze(0)
        return img, orig_shape

    def _forward(self, list_img0, img1, list_img0_poses, init_img1_pose, list_img0_K, img1_K, option):
        images = []
        for i, img0 in enumerate(list_img0):
            img0_pre, _ = self.preprocess(img0)
            images.append({"img": img0_pre, 
                           "idx": i, 
                           "instance": i, 
                           "true_shape": np.int32([img0.shape[-2:]])})

        img1_pre, _ = self.preprocess(img1)
        images.append({"img": img1_pre, 
                       "idx": len(list_img0), 
                       "instance": len(list_img0), 
                       "true_shape": np.int32([img1.shape[-2:]])})

        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        ##### GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, verbose=self.verbose)
        if option == 'single':
            known_poses = [pose for pose in list_img0_poses] + [init_img1_pose] # Pose: from world to camera
            scene.preset_pose(known_poses=known_poses,
                              pose_msk=[True] * len(list_img0_poses) + [False])
        # if list_img0_K is not None:
        #     scene.preset_intrinsics(list_img0_K + [img1_K])
        
        # Perform optimization
        loss = scene.compute_global_alignment(init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr)

        ##### Get results
        focals, im_poses = scene.get_focals(), scene.get_im_poses()
        est_focal, est_im_pose = focals[-1], im_poses[-1]
        # print('focals:\n', focals)
        # print('poses:\n', im_poses, im_poses.shape)

        self.scene = scene
        return est_focal, est_im_pose, loss
