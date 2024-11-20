# This code is refered to gmberton's repo: https://github.com/gmberton/VPR-methods-evaluation/blob/master/vpr_models/__init__.py

import os
import torchvision.transforms as tfm
import numpy as np
import torch
import faiss
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from estimator import BaseEstimator, THIRD_PARTY_DIR
from estimator.utils import affine_combination, pose_interpolation, pca_analysis, add_to_path, to_numpy

add_to_path(THIRD_PARTY_DIR.joinpath("vpr_models"))
from resizing_wrapper import ResizingWrapper
import sfrs, convap, mixvpr, netvlad

class VPREstimator(BaseEstimator):
    def __init__(self, device="cpu", method='cosplace', backbone='ResNet18', des_dimension='512', out_dir='/tmp', *args, **kwargs):
        super().__init__(device, **kwargs)
        self.verbose = False
        self.out_dir = Path(out_dir)

        if method == "sfrs":
            model = sfrs.SFRSModel()
        elif method == "netvlad":
            model = netvlad.NetVLAD(descriptors_dimension=des_dimension)
        elif method == "cosplace":
            model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                                backbone=backbone, fc_output_dim=des_dimension)
        elif method == "mixvpr":
            model = mixvpr.get_mixvpr(descriptors_dimension=des_dimension)
        elif method == "convap":
            model = convap.get_convap(descriptors_dimension=des_dimension)
        elif method == "eigenplaces":
            model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                                backbone=backbone, fc_output_dim=des_dimension)
        elif method == "eigenplaces-indoor":
            model = torch.hub.load("Enrico-Chiavassa/Indoor-VPR", "get_trained_model",
                                backbone=backbone, fc_output_dim=des_dimension)
        elif method == "anyloc":
            anyloc = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device="cuda")
            model = ResizingWrapper(anyloc, resize_type="dino_v2_resize")
        elif method == "salad":
            salad = torch.hub.load("serizba/salad", "dinov2_salad")
            model = ResizingWrapper(salad, resize_type="dino_v2_resize")
        elif method == "cricavpr":
            cricavpr = torch.hub.load("Lu-Feng/CricaVPR", "trained_model")
            model = ResizingWrapper(cricavpr, resize_type=224)

        self.method = method
        self.backbone = backbone
        self.des_dimension = des_dimension
        print(f"VPREstimator: {method} with backbone {backbone} and descriptor dimension {des_dimension}")

        self.model = model.eval().to(device)
        self.recall_value = 5

    def show_reconstruction(self):
        scene = pca_analysis(self.db_names, self.query_names, self.db_descriptors, self.query_descriptors)        
        scene.savefig(self.out_dir / 'vpr_pca_analysis.png')

        db_images = [Image.open(self.scene_root / name) for name in self.db_names]
        query_images = [Image.open(self.scene_root / name) for name in self.query_names]
        fig, axes = plt.subplots(self.predictions.shape[0], self.recall_value + 1, figsize=(2 * (self.recall_value + 1), 2))
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        for query_id in range(self.predictions.shape[0]):
            axes[query_id, 0].imshow(query_images[query_id])
            axes[query_id, 0].set_title(f'Q-{query_id}', fontsize=10)
            for i in range(self.recall_value):
                axes[query_id, i + 1].imshow(db_images[self.predictions[query_id, i]])
                axes[query_id, i + 1].set_title(f'DB-{self.predictions[query_id, i]}', fontsize=10)
        fig.tight_layout()
        plt.savefig(self.out_dir / 'vpr_result.png')

    def load_image(self, path_image, im_size=None):
        transformations = [
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if im_size:
            transformations.append(tfm.Resize(size=im_size, antialias=True))    
        transform = tfm.Compose(transformations)
        pil_img = Image.open(path_image).convert("RGB")
        normalized_img = transform(pil_img)
        return normalized_img.unsqueeze(0)

    def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts):
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
        with torch.inference_mode():
            db_descriptors = np.empty((len(list_img0_name), self.des_dimension), dtype='float32')
            for idx, path_image in enumerate(list_img0_name):
                image = self.load_image(scene_root / path_image, resize)
                descriptor = self.model(image.to(self.device)).detach().cpu().numpy()
                db_descriptors[idx, :] = descriptor
            
            query_descriptors = np.empty((1, self.des_dimension), dtype='float32')
            image = self.load_image(scene_root / img1_name, resize)
            descriptor = self.model(image.to(self.device)).detach().cpu().numpy()
            query_descriptors[0, :] = descriptor

        print('DB Descriptor shpae: ', db_descriptors.shape)
        print('Query Descriptor shpae: ', query_descriptors.shape)

        # Calculate recalls
        faiss_index = faiss.IndexFlatL2(self.des_dimension)
        faiss_index.add(db_descriptors)
        _, predictions = faiss_index.search(query_descriptors, len(list_img0_name))
        print(predictions) # 1 x recall_values

        # Calculate the affline combination and do pose interpolation
        a_opt, loss = affine_combination(query_descriptors[0, :], db_descriptors[predictions[0], :])
        est_im_pose = pose_interpolation(to_numpy(list_img0_poses), a_opt)
        print(a_opt)
        
        ##### Store results for visualization
        self.scene_root = scene_root
        self.db_names = list_img0_name
        self.query_names = [img1_name]
        self.db_descriptors = db_descriptors
        self.query_descriptors = query_descriptors
        self.predictions = predictions

        ##### Get results
        est_focal = list_img0_intr[0]['K'][0][0].item()

        return est_focal, est_im_pose, loss
