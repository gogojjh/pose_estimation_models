import sys
from pathlib import Path
import os
import torchvision.transforms as tfm
import py3_wget
import torch
import argparse
import zipfile

sys.path.append(str(Path(__file__).parent.parent.joinpath("third_party")))
sys.path.append(str(Path(__file__).parent.parent.joinpath("third_party/mickey")))
from mickey.lib.models.builder import build_model
from mickey.lib.datasets.utils import correct_intrinsic_scale
from mickey.lib.models.MicKey.modules.utils.training_utils import (
    colorize,
    generate_heat_map,
)
from mickey.config.default import cfg
import numpy as np
from pathlib import Path
import cv2

from matching.base_matcher import BaseMatcher
from matching.utils import to_numpy

from matching import WEIGHTS_DIR


class MickeyMatcher(BaseMatcher):
    zip_path = WEIGHTS_DIR.joinpath("mickey.zip")
    model_path = WEIGHTS_DIR.joinpath("mickey_weights/mickey.ckpt")
    cfg_path = WEIGHTS_DIR.joinpath("mickey_weights/config.yaml")

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        self.verbose = False

        self.download_weights()

        self.cfg = cfg.clone()
        self.cfg.merge_from_file(MickeyMatcher.cfg_path)

        self.model = build_model(self.cfg, checkpoint=MickeyMatcher.model_path)

    @staticmethod
    def download_weights():
        url = "https://storage.googleapis.com/niantic-lon-static/research/mickey/assets/mickey_weights.zip"

        if not os.path.isfile(MickeyMatcher.model_path):
            print("Downloading Mickey... (takes a while)")
            py3_wget.download_file(url, MickeyMatcher.zip_path)
            with zipfile.ZipFile(MickeyMatcher.zip_path, "r") as zip_ref:
                zip_ref.extractall(WEIGHTS_DIR)

    def preprocess(self, img):
        return img.unsqueeze(0)

    def read_intrinsics(self, path_intrinsics, resize):
        Ks = {}
        with Path(path_intrinsics).open("r") as f:
            for line in f.readlines():
                if "#" in line:
                    continue

                line = line.strip().split(" ")
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                if resize is not None:
                    K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H).numpy()
                Ks[img_name] = K
        return Ks

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        K = self.read_intrinsics(self.path_intrinsics, self.resize)

        data = {}
        data["image0"] = img0
        data["image1"] = img1
        data["K_color0"] = torch.from_numpy(K["im0.jpg"]).unsqueeze(0).to(self.device)
        data["K_color1"] = torch.from_numpy(K["im1.jpg"]).unsqueeze(0).to(self.device)

        print("Running MicKey relative pose estimation...")
        self.model(data)
        self.scene = data

        """Retrieve matching results"""
        R, t = data["R"], data["t"]
        print("Estimated Poses:\n", R, "\n", t)

        # Extract keypoints and descriptors
        kpts0, kpts1 = data["kps0"].squeeze(0).T, data["kps1"].squeeze(0).T
        desc0, desc1 = data["dsc0"].squeeze(0).T, data["dsc1"].squeeze(0).T

        # Matched keypoints
        inliers_list = data["inliers_list"][0]
        mkpts0, mkpts1 = inliers_list[:, :2], inliers_list[:, 2:4]

        # Compute confidence score
        num_inliers = data["inliers"]
        confidence = num_inliers / self.cfg.PROCRUSTES.NUM_SAMPLED_MATCHES
        print(f"{confidence=}, {num_inliers=}, {inliers_list.shape=}")

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
