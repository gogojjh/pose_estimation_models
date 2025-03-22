import os
import torch
import torchvision.transforms as tfm
import py3_wget
import numpy as np

from estimator.utils import resize_to_divisible, add_to_path
from estimator import BaseEstimator, WEIGHTS_DIR, THIRD_PARTY_DIR

add_to_path(THIRD_PARTY_DIR.joinpath("reloc3r"))
add_to_path(THIRD_PARTY_DIR.joinpath("duster"))

from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from dust3r.utils.image import load_images

class Reloc3rEstimator(BaseEstimator):
    """Estimator class for Reloc3r model."""
    model_path = WEIGHTS_DIR.joinpath("reloc3r-512.pth")
    vit_patch_size = 16

    def __init__(self, device="cpu", model_args="512", use_lora=False, *args, **kwargs):
        """Initializes the Reloc3rEstimator.

        Args:
            device (str): Device to run the model on.
            model_args (str): Model arguments.
            use_lora (bool): Whether to use LoRA.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.verbose = False

        self.visloc = Reloc3rVisloc()

        self.download_weights(model_args)
        # Load the model weights using torch.load and pass them to Reloc3rRelpose
        model_weights = torch.load(self.model_path)
        self.model = Reloc3rRelpose()
        self.model.load_state_dict(model_weights)
        print(f'Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}')

        # Handle LoRA integration before moving to target device
        if use_lora:
            if 'lora_path' not in kwargs:
                raise RuntimeError("Missing required 'lora_path' argument for LoRA integration")
            self._safely_integrate_lora(kwargs['lora_path'], target_device=device)

        self.model.to(device)
        self.model.eval()

    def download_weights(self, model_args):
        """Downloads the weights for the Reloc3r model.

        Args:
            model_args (str): Model arguments.
        """
        if '224' in model_args:
            url = "https://huggingface.co/siyan824/reloc3r-224/resolve/main/Reloc3r-224.pth"
        elif '512' in model_args:
            url = "https://huggingface.co/siyan824/reloc3r-512/resolve/main/Reloc3r-512.pth"

        if not os.path.isfile(self.model_path):
            print(f"Downloading Reloc3r ({model_args})... (takes a while)")
            py3_wget.download_file(url, self.model_path)

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

    def show_reconstruction(self, cam_size=None):
        """Shows the reconstruction (not implemented).

        Args:
            cam_size (tuple): Camera size.
        """
        pass  # Placeholder, not implemented in this version

    def _forward(self, scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts):
        """Performs the forward pass of the pose estimation model using Reloc3r.

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
        img0s_dict = load_images(imgs_path, size=resize, verbose=self.verbose) # list of dict()
        img1_dict = load_images([str(scene_root / img1_name)], size=resize, verbose=self.verbose)[0] # dict()

        relative_poses = []
        for img0_dict in img0s_dict:
            view1 = {'img': img0_dict['img'], 'true_shape': torch.tensor(img0_dict['true_shape'])}
            view2 = {'img': img1_dict['img'], 'true_shape': torch.tensor(img1_dict['true_shape'])}
            pose2to1 = inference_relpose([view1, view2], self.model, self.device)
            relative_poses.append(pose2to1.squeeze(0).cpu().numpy())

        # Assuming list_img0_poses are the absolute poses of the reference images
        absolute_poses = [pose.cpu().numpy() for pose in list_img0_poses]
        est_im_pose = self.visloc.motion_averaging(absolute_poses, relative_poses)

        est_focal = None  # Focal length not estimated in this setup
        loss = 0.0  # Placeholder for loss value

        return est_focal, est_im_pose, loss

    def save_results(self):
        """Saves the results (not implemented)."""
        pass  # Placeholder, not implemented in this version