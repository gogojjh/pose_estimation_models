import torch
import numpy as np

from pathlib import Path
from typing import Union, List, Dict

from estimator import BaseEstimator, WEIGHTS_DIR, THIRD_PARTY_DIR
from estimator.utils import add_to_path, align_poses, to_numpy

add_to_path(THIRD_PARTY_DIR.joinpath("vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VggtEstimator(BaseEstimator):
    """Estimator class for the VGGT feed-forward multi-view model.

    Unlike DUSt3R/MASt3R, VGGT predicts every camera pose in a single forward pass
    (no iterative global alignment), in its own arbitrary-scale reference frame anchored
    at the first input image. To recover an absolute pose for the query image, the
    reference images' predicted poses are aligned (via a similarity transform) to their
    known absolute poses, and that same transform is applied to the query prediction.
    """

    model_name = "facebook/VGGT-1B"

    def __init__(self, device="cpu", *args, **kwargs):
        """Initializes the VggtEstimator.

        Args:
            device (str): Device to run the model on.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(device, **kwargs)
        self.verbose = False

        self.model = self.download_weights()
        print(f'Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        self.model = self.model.to(device)
        self.model.eval()

        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    def download_weights(self):
        """Loads the VGGT-1B weights, cached under WEIGHTS_DIR via the HuggingFace Hub."""
        return VGGT.from_pretrained(self.model_name, cache_dir=WEIGHTS_DIR)

    def show_reconstruction(self, cam_size=None):
        """Shows the reconstruction (not implemented)."""
        pass

    @staticmethod
    def _w2c_to_c2w(Rt: np.ndarray) -> np.ndarray:
        """Inverts a 3x4 world-to-camera [R|t] matrix into a 4x4 camera-to-world matrix."""
        R, t = Rt[:3, :3], Rt[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    def _forward(
        self,
        scene_root: Path,
        list_img0: Union[List[str], List[Path]],
        img1: Union[str, Path],
        list_img0_poses: List[torch.Tensor],
        list_img0_intr: List[Dict],
        img1_intr: Dict,
        est_opts: Dict,
    ):
        """Runs a single VGGT forward pass over all reference + query images, then aligns the
        predicted reference poses to their known absolute poses to recover the query pose.

        Args:
            scene_root (Path): The root directory of the scene.
            list_img0 (list): Image names for the reference images.
            img1 (str): The name of the query image.
            list_img0_poses (list): Known absolute (camera-to-world) poses of the reference images.
            list_img0_intr (list): Unused (VGGT predicts its own intrinsics).
            img1_intr (dict): Unused (VGGT predicts its own intrinsics).
            est_opts (dict): Unused (VGGT is feed-forward, no optimization options).

        Returns:
            tuple: (estimated focal length, estimated camera-to-world pose, loss)
        """
        assert isinstance(list_img0[0], (str, Path)) and isinstance(
            img1, (str, Path)
        ), "VggtEstimator expects image names/paths, not preloaded tensors"

        num_ref = len(list_img0)
        image_paths = [str(scene_root / name) for name in list_img0] + [str(scene_root / img1)]
        images = load_and_preprocess_images(image_paths).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype, enabled=self.dtype != torch.float32):
            predictions = self.model(images)
        self.output_inference = predictions

        H, W = images.shape[-2:]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"].float(), (H, W))
        extrinsic, intrinsic = extrinsic[0], intrinsic[0]  # drop batch dim -> (S, 3, 4), (S, 3, 3)

        vggt_c2w = [self._w2c_to_c2w(to_numpy(extrinsic[i])) for i in range(extrinsic.shape[0])]
        vggt_c2w_ref, vggt_c2w_query = vggt_c2w[:num_ref], vggt_c2w[num_ref]
        known_c2w_ref = [to_numpy(pose) for pose in list_img0_poses]

        if num_ref >= 2:
            _, (scale, R, translation) = align_poses(vggt_c2w_ref, known_c2w_ref)
            est_im_pose = np.eye(4)
            est_im_pose[:3, :3] = R @ vggt_c2w_query[:3, :3]
            est_im_pose[:3, 3] = scale * (R @ vggt_c2w_query[:3, 3]) + translation
        else:
            # A single reference pose can't constrain scale, so compose directly and assume
            # VGGT's own scale already matches the world scale between these two frames.
            est_im_pose = known_c2w_ref[0] @ np.linalg.inv(vggt_c2w_ref[0]) @ vggt_c2w_query

        est_focal = intrinsic[num_ref, 0, 0].detach()
        loss = 0.0
        self.scene = None

        return est_focal, est_im_pose, loss

    def save_results(self, log_dir):
        """Saves the results (not implemented)."""
        pass
