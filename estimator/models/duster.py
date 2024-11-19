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
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

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

    def show_reconstruction(self, cam_size=0.2):
        self.scene.show(cam_size=cam_size)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        img = resize_to_divisible(img, self.vit_patch_size)
        img = self.normalize(img).unsqueeze(0)
        return img, orig_shape

    def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_K, img1_K, img_size, est_opts):
        """
        Performs the forward pass of the pose estimation model.

        Args:
            scene_root (str): The root directory of the scene.
            list_img0_name (list): A list of image names for the reference images.
            img1_name (str): The name of the target image.
            list_img0_poses (list): A list of poses for the reference images.
            list_img0_K (list): A list of intrinsic camera matrices for the reference images.
            img1_K (matrix): The intrinsic camera matrix for the target image.
            est_opts (dict): Additional options for the pose estimation.

        Returns:
            tuple: A tuple containing the estimated focal length, estimated image pose, and the loss value.
        """

        imgs_path = [str(scene_root / img_name) for img_name in list_img0_name] + [str(scene_root / img1_name)]
        resize = est_opts.get('resize', 512)
        images = load_images(imgs_path, size=resize)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        ##### GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, verbose=self.verbose)
        if est_opts['known_extrinsics']:
            known_poses = [pose for pose in list_img0_poses] + [np.eye(4)]
            scene.preset_pose(known_poses=known_poses, pose_msk=[True] * len(list_img0_poses) + [False])
        # TODO(gogojjh):
        # if list_img0_K is not None:
        #     scene.preset_intrinsics(list_img0_K + [img1_K])
        
        ##### Perform optimization
        loss = scene.compute_global_alignment(init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr)

        ##### Get results
        focals, im_poses = scene.get_focals(), scene.get_im_poses()
        est_focal, est_im_pose = focals[-1], im_poses[-1]

        self.scene = scene
        return est_focal.detach(), est_im_pose.detach(), loss
