import sys
from pathlib import Path
import os
import torch
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
        self.niter = 301

        self.download_weights()
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

        if not os.path.isfile(Dust3rEstimator.model_path):
            print("Downloading Dust3r(ViT large)... (takes a while)")
            py3_wget.download_file(url, Dust3rEstimator.model_path)

    def save_results(self):
        pass

    def show_reconstruction(self, cam_size=0.2):
        self.scene.show(cam_size=cam_size)

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
            conf='log'
        )

        if est_opts['known_extrinsics']:
            known_poses = [pose for pose in list_img0_poses] + [np.eye(4)]
            scene.preset_pose(known_poses=known_poses, pose_msk=[True] * len(list_img0_poses) + [False])
            print(known_poses)
            
        if est_opts['known_intrinsics']:
            list_img_intr = list_img0_intr + [img1_intr]
            resize_list_img_K = []
            for idx, image in enumerate(images):
                ori_K = list_img_intr[idx]['K']
                ori_im_size = list_img_intr[idx]['im_size']                    # HxW
                new_im_size = torch.from_numpy(image['true_shape']).squeeze(0) # HxW
                scale_h = new_im_size[0] / ori_im_size[0]
                scale_w = new_im_size[1] / ori_im_size[1]
                new_K = ori_K.clone()
                new_K[0, 0] *= scale_w  # Focal length X
                new_K[1, 1] *= scale_h  # Focal length Y
                new_K[0, 2]  = new_K[0, 2] * scale_w  # Principal point X
                new_K[1, 2]  = new_K[1, 2] * scale_h  # Principal point Y
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
        focals, im_poses = scene.get_focals(), scene.get_im_poses()
        est_focal, est_im_pose = focals[-1], im_poses[-1]

        self.scene = scene
        return est_focal.detach(), est_im_pose.detach(), loss
