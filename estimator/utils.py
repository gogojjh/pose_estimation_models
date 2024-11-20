import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tfm
import os, contextlib
from yacs.config import CfgNode as CN
from typing import Union

from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logger = logging.getLogger()
logger.setLevel(31)  # Avoid printing useless low-level logs

def convert_vec_to_matrix(vec_p, vec_q, mode='xyzw'):
	# Initialize a 4x4 identity matrix
	tf = np.eye(4)
	if mode == 'xyzw':
		# Set the rotation part of the transformation matrix using the quaternion
		tf[:3, :3] = Rotation.from_quat(vec_q).as_matrix()
		# Set the translation part of the transformation matrix
		tf[:3, 3] = vec_p
	elif mode == 'wxyz':
		# Set the rotation part of the transformation matrix using the quaternion
		tf[:3, :3] = Rotation.from_quat(np.roll(vec_q, -1)).as_matrix()
		# Set the translation part of the transformation matrix
		tf[:3, 3] = vec_p
	return tf
def convert_matrix_to_vec(tf_matrix, mode='xyzw'):
	if mode == 'xyzw':
		# Extract the translation vector from the matrix
		vec_p = tf_matrix[:3, 3]
		# Extract the rotation part of the matrix and convert it to a quaternion
		vec_q = Rotation.from_matrix(tf_matrix[:3, :3]).as_quat()
	if mode == 'wxyz':
		# Extract the translation vector from the matrix
		vec_p = tf_matrix[:3, 3]
		# Extract the rotation part of the matrix and convert it to a quaternion
		vec_q = np.roll(Rotation.from_matrix(tf_matrix[:3, :3]).as_quat(), 1)
	return vec_p, vec_q

def affine_combination(d_test, d_train):
    """
    Solves the affine combination optimization problem:
    Minimize ||d_test - Σ(a_i * d_train[i])||_2 subject to Σ(a_i) = 1.

    Parameters:
    - d_test: np.array, the test descriptor (1D array).
    - d_train: np.array, database descriptors (2D array, each row is a descriptor).

    Returns:
    - a_opt: np.array, optimized weights for the affine combination.
    - loss: float, the L2 loss of the optimized combination.
    """
    k = d_train.shape[0]  # Number of database descriptors

    # Define the objective function to minimize
    def objective(a):
        combination = np.sum(a[:, np.newaxis] * d_train, axis=0)
        return np.linalg.norm(d_test - combination)

    # Initial guess for weights
    a0 = np.ones(k) / k  # Uniform distribution as starting point

    # Constraints: sum of weights equals 1
    constraints = {'type': 'eq', 'fun': lambda a: np.sum(a) - 1}

    # Bounds: weights can be any real number (change to (0, None) if only positive weights are allowed)
    bounds = [(0, 1) for _ in range(k)]

    # Solve the optimization problem
    result = minimize(objective, a0, bounds=bounds, constraints=constraints)

    # Return optimized weights
    return result.x, objective(result.x)

def pose_interpolation(poses, weights):
    weights /= np.sum(weights)

    list_quat = [Rotation.from_matrix(pose[:3, :3]).as_quat() for pose in poses]
    list_tsl = [pose[:3, 3] for pose in poses]

    weighted_quaternion = np.average(list_quat, axis=0, weights=weights)
    weighted_quaternion /= np.linalg.norm(weighted_quaternion)  # Normalize quaternion
    final_rotation = Rotation.from_quat(weighted_quaternion).as_matrix()

    translations = np.array(list_tsl)
    final_translation = np.average(translations, axis=0, weights=weights)

    est_pose = np.eye(4)
    est_pose[:3, :3] = final_rotation
    est_pose[:3, 3] = final_translation

    return est_pose

def pca_analysis(db_names, query_names, db_descriptors, query_descriptors):
    pca = PCA(n_components=2)
    db_embeddings_2d = pca.fit_transform(db_descriptors)
    print('DB 2D embedding shape: ', db_embeddings_2d.shape)
    query_embeddings_2d = pca.transform(query_descriptors)
    print('Query 2D embedding shape: ', query_embeddings_2d.shape)

    # Plotting the embeddings
    plt.figure(figsize=(8, 6))

    # Plot database embeddings with text labels
    for i, (x, y) in enumerate(db_embeddings_2d):
        plt.scatter(x, y, marker='o', color='blue', label='Database' if i == 0 else "")
        plt.text(x, y, db_names[i], fontsize=15, color='blue', ha='right', va='bottom')

    # Plot query embeddings with text labels
    for i, (x, y) in enumerate(query_embeddings_2d):
        plt.scatter(x, y, marker='^', color='green', label='Query' if i == 0 else "")
        plt.text(x, y, query_names[i], fontsize=15, color='green', ha='left', va='top')

    # Add labels and legend
    plt.title("PCA Visualization of Global Descriptors")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.5)

    return plt

def get_image_pairs_paths(inputs):
    inputs = Path(inputs)
    if not inputs.exists():
        raise RuntimeError(f"{inputs} does not exist")

    if inputs.is_file():
        with open(inputs) as file:
            lines = file.read().splitlines()
        pairs_of_paths = [line.strip().split(" ") for line in lines]
        for pair in pairs_of_paths:
            if len(pair) != 2:
                raise RuntimeError(f"{pair} should be a pair of paths")
        pairs_of_paths = [(Path(path0.strip()), Path(path1.strip())) for path0, path1 in pairs_of_paths]
    else:
        pair_dirs = sorted(Path(inputs).glob("*"))
        pairs_of_paths = [list(pair_dir.glob("*")) for pair_dir in pair_dirs]
        for pair in pairs_of_paths:
            if len(pair) != 2:
                raise RuntimeError(f"{pair} should be a pair of paths")
    return pairs_of_paths

def to_numpy(x: Union[torch.Tensor, np.ndarray, dict, list]) -> np.ndarray:
    """convert item or container of items to numpy

    Args:
        x (Union[torch.Tensor, np.ndarray, dict, list]): input

    Returns:
        np.ndarray: numpy array of input
    """
    if isinstance(x, list):
        return np.array([to_numpy(i) for i in x])
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
        return x
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x

def to_tensor(x: Union[np.ndarray, torch.Tensor], device: str = None) -> torch.Tensor:
    """Convert to tensor and place on device

    Args:
        x (np.ndarray | torch.Tensor): item to convert to tensor
        device (str, optional): device to place tensor on. Defaults to None.

    Returns:
        torch.Tensor: tensor with data from `x` on device `device`
    """
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if device is not None:
        return x.to(device)


def to_normalized_coords(pts: Union[np.ndarray, torch.Tensor], height: int, width: int) -> np.ndarray:
    """normalize kpt coords from px space to [0,1]
    Assumes pts are in x, y order in array/tensor shape (N, 2)

    Args:
        pts (Union[np.ndarray, torch.Tensor]): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.ndarray: kpts in normalized [0,1] coords
    """
    # normalize kpt coords from px space to [0,1]
    # assume pts are in x,y order
    assert pts.shape[-1] == 2, f"input to `to_normalized_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts).astype(float)
    pts[:, 0] /= width
    pts[:, 1] /= height

    return pts

def to_px_coords(pts: Union[np.ndarray, torch.Tensor], height: int, width: int) -> np.ndarray:
    """unnormalized kpt coords from [0,1] to px space
    Assumes pts are in x, y order

    Args:
        pts (Union[np.ndarray, torch.Tensor]): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.ndarray: kpts in normalized [0,1] coords
    """
    assert pts.shape[-1] == 2, f"input to `to_px_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts)
    pts[:, 0] *= width
    pts[:, 1] *= height

    return pts

def resize_to_divisible(img: torch.Tensor, divisible_by: int = 14) -> torch.Tensor:
    """Resize to be divisible by a factor. Useful for ViT based models.

    Args:
        img (torch.Tensor): img as tensor, in (*, H, W) order
        divisible_by (int, optional): factor to make sure img is divisible by. Defaults to 14.

    Returns:
        torch.Tensor: img tensor with divisible shape
    """
    h, w = img.shape[-2:]

    divisible_h = round(h / divisible_by) * divisible_by
    divisible_w = round(w / divisible_by) * divisible_by
    img = tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)

    return img


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

def load_module(module_name: str, module_path: Union[Path, str]) -> None:
    """Load module from `module_path` into the interpreter with the namespace given by module_name.

    Note that `module_path` is usually the path to an `__init__.py` file.

    Args:
        module_name (str): module name (will be used to import from later, as in `from module_name import my_function`)
        module_path (Path | str): path to module (usually an __init__.py file)
    """
    import importlib

    # load gluefactory into namespace
    # module_name = 'gluefactory'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def add_to_path(path: Union[str, Path], insert=None) -> None:
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    if insert is None:
        sys.path.append(path)
    else:
        sys.path.insert(insert, path)

