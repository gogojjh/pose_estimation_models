import torch
import numpy as np

from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

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
        self._recon_state: Dict = {}

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

    def show_reconstruction(self, cam_size: Optional[float] = None, conf_thres: float = 50.0):
        """Shows the point cloud and camera poses in an interactive trimesh window.

        Reuses dust3r's SceneViz so the viewer matches Dust3rEstimator/Mast3rEstimator.
        Points and cameras share the alignment computed in `_forward`, so what is drawn
        lives in the same world frame as the returned pose estimate.

        Args:
            cam_size: camera marker size; derived from the pose spread when None.
            conf_thres: percentage of lowest-confidence points to drop.

        Returns:
            The SceneViz, or None if the estimator has not been run yet.
        """
        if not self._recon_state:
            print("No reconstruction to show: run the estimator first.")
            return None

        add_to_path(THIRD_PARTY_DIR.joinpath("duster"))
        from dust3r.viz import SceneViz, auto_cam_size

        state = self._recon_state
        align, num_ref = state["align"], state["num_ref"]

        imgs = [np.transpose(img, (1, 2, 0)) for img in state["images"]]
        pts3d = [self._align_points(align, pts) for pts in state["world_points"]]
        masks = self._confidence_masks(state["world_points_conf"], conf_thres)
        poses = [self._align_pose(align, c2w) for c2w in state["vggt_c2w"]]
        # Query frame in red so the localization target stands out from the references.
        colors = [(0, 128, 255)] * num_ref + [(255, 0, 0)]

        viz = SceneViz()
        viz.add_pointcloud(pts3d, imgs, masks)
        viz.add_cameras(
            poses,
            state["focals"],
            images=imgs,
            colors=colors,
            cam_size=auto_cam_size(np.stack(poses)) if cam_size is None else cam_size,
        )
        viz.show()

        return viz

    @staticmethod
    def _w2c_to_c2w(Rt: np.ndarray) -> np.ndarray:
        """Inverts a 3x4 world-to-camera [R|t] matrix into a 4x4 camera-to-world matrix."""
        R, t = Rt[:3, :3], Rt[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    @staticmethod
    def _compute_alignment(
        vggt_c2w_ref: List[np.ndarray], known_c2w_ref: List[np.ndarray]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Similarity transform (scale, R, t) mapping VGGT's world frame to the known one.

        align_poses solves rotation from the reference orientations, so N=2 is fully
        constrained; with a single reference it degrades to a rigid transform (scale=1).
        """
        _, (scale, rotation, translation) = align_poses(vggt_c2w_ref, known_c2w_ref)
        return float(scale), rotation, translation

    @staticmethod
    def _align_pose(align: Tuple[float, np.ndarray, np.ndarray], c2w: np.ndarray) -> np.ndarray:
        """Applies a (scale, R, t) similarity transform to a 4x4 camera-to-world pose."""
        scale, rotation, translation = align
        aligned = np.eye(4)
        aligned[:3, :3] = rotation @ c2w[:3, :3]
        aligned[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
        return aligned

    @staticmethod
    def _align_points(align: Tuple[float, np.ndarray, np.ndarray], pts: np.ndarray) -> np.ndarray:
        """Applies a (scale, R, t) similarity transform to points of shape (..., 3)."""
        scale, rotation, translation = align
        return scale * (pts @ rotation.T) + translation

    @staticmethod
    def _confidence_masks(conf: np.ndarray, conf_thres: float) -> List[np.ndarray]:
        """Per-frame boolean masks keeping only the most confident points.

        Args:
            conf: per-pixel confidence of shape (S, H, W).
            conf_thres: percentage of lowest-confidence points to drop, in [0, 100).

        Returns:
            One (H, W) boolean mask per frame. The threshold is computed over all frames
            at once so every frame keeps points on the same confidence scale.
        """
        threshold = np.percentile(conf, conf_thres) if conf_thres > 0 else 0.0
        keep = (conf >= threshold) & (conf > 1e-5)
        return [keep[i] for i in range(keep.shape[0])]

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

        align = self._compute_alignment(vggt_c2w_ref, known_c2w_ref)
        est_im_pose = self._align_pose(align, vggt_c2w_query)

        est_focal = intrinsic[num_ref, 0, 0].detach()
        loss = 0.0
        self.scene = None
        self._recon_state = {
            "images": to_numpy(images),                                     # (S, 3, H, W) in [0, 1]
            # VGGT currently runs its heads under autocast(enabled=False), so these are
            # already float32 and .float() is a no-op. Kept as a guard: if that ever
            # changes, bfloat16 would reach numpy(), which rejects it outright.
            "world_points": to_numpy(predictions["world_points"][0].float()),       # (S, H, W, 3)
            "world_points_conf": to_numpy(predictions["world_points_conf"][0].float()),  # (S, H, W)
            "focals": [float(intrinsic[i, 0, 0]) for i in range(intrinsic.shape[0])],
            "vggt_c2w": vggt_c2w,
            "align": align,
            "num_ref": num_ref,
        }

        return est_focal, est_im_pose, loss

    def save_results(self, log_dir):
        """Saves the results (not implemented)."""
        pass
