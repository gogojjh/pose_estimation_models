"""
File to import matchers. The module's import are within the functions, so that
a module is imported only iff needed, reducing the number of raised errors and
warnings due to unused modules.
"""
from pathlib import Path
from .utils import supress_stdout, add_to_path
from .models.base_estimator import BaseEstimator

# add viz2d from lightglue to namespace - thanks lightglue!
THIRD_PARTY_DIR = Path(__file__).parent.joinpath('third_party')

WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"

available_models = [
    "hloc_disk_dilg",
    "vpr_cosplace_resnet18_512",
    "vpr_netvlad_resnet18_4096",
    "duster_nocalib_pretrain",
    "duster_calib_pretrain",
    "duster_nocalib_ftlora_*pdepth",
    "duster_nocalib_ftlora_*gtdepth",
    "master",
]

def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch

# @supress_stdout
def get_estimator(estimator_name="master", device="cpu", max_num_keypoints=2048, out_dir='/tmp', *args, **kwargs):
    if 'hloc' in estimator_name:
        from estimator.models import hloc
        
        # ['superpoint_max', 'superpoint_inloc', 'r2d2', 'd2net-ss', 'sift', 'sosnet', 'disk', 'aliked-n16']
        name1 = estimator_name.split('_')[1]
        if name1 == 'suerpoint':
            feature_name = 'superpoint_max'
        elif name1 == 'r2d2':
            feature_name = 'r2d2'
        elif name1 == 'd2net':
            feature_name = 'd2net-ss'
        elif name1 == 'sift':
            feature_name = 'sift'
        elif name1 == 'sosnet':
            feature_name = 'sosnet'
        elif name1 == 'disk':
            feature_name = 'disk'
        elif name1 == 'aliked':
            feature_name = 'aliked-n16'
        else:
            raise RuntimeError(
                f"Feature {name1} for hloc not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )

        # ['superpoint+lightglue', 'disk+lightglue', 'aliked+lightglue', 'superglue', 'superglue-fast', 'NN-superpoint', 'NN-ratio', 'NN-mutual', 'adalam']
        name2 = estimator_name.split('_')[2]
        if name2 == 'splg':
            matcher_name = 'superpoint+lightglue'
        elif name2 == 'dilg':
            matcher_name = 'disk+lightglue'
        elif name2 == 'alilg':
            matcher_name = 'aliked+lightglue'
        elif name2 == 'sg':
            matcher_name = 'superglue'
        elif name2 == 'sgfast':
            matcher_name = 'superglue-fast'
        elif name2 == 'nnsp':
            matcher_name = 'NN-superpoint'
        elif name2 == 'nnratio':
            matcher_name = 'NN-ratio'
        elif name2 == 'nnmutual':
            matcher_name = 'NN-mutual'
        elif name2 == 'adalam':
            matcher_name = 'adalam'
        else:
            raise RuntimeError(
                f"Matcher {name2} for hloc not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )

        return hloc.HlocEstimator(device, feature_name, matcher_name, max_num_keypoints, out_dir, *args, **kwargs)

    elif 'vpr' in estimator_name:
        method_name = estimator_name.split('_')[1]
        if method_name != 'netvlad' and \
           method_name != 'convap' and method_name != 'maxvpr' and \
           method_name != 'eigenplaces' and method_name != 'anyloc' and \
           method_name != 'cosplace' and method_name != 'sfrs' and \
           method_name != 'salad' and method_name != 'cricavpr':
            raise RuntimeError(
                f"Method {method_name} for vpr not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )

        backbone_name = estimator_name.split('_')[2]
        if backbone_name == 'vgg16':
            backbone_name = 'VGG16'
        elif backbone_name == 'resnet18':
            backbone_name = 'ResNet18'
        elif backbone_name == 'resnet50':
            backbone_name = 'ResNet50'
        elif backbone_name == 'resnet101':
            backbone_name = 'ResNet101'
        elif backbone_name == 'resnet152':
            backbone_name = 'ResNet152'
        else:
            raise RuntimeError(
                f"Backbone {backbone_name} for vpr not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )
        
        des_dimension = int(estimator_name.split('_')[3])
        if des_dimension != 128 and des_dimension != 256 and des_dimension != 512 and des_dimension != 1024 and des_dimension != 2048 and des_dimension != 4096:
            raise RuntimeError(
                f"Descriptor dimension {des_dimension} for vpr not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
            )

        from estimator.models import vpr
        return vpr.VPREstimator(device, method_name, backbone_name, des_dimension, out_dir, *args, **kwargs)

    elif 'duster' in estimator_name or 'dust3r' in estimator_name:
        # Check for calibration and LoRA flags
        use_calib = '_calib' in estimator_name
        use_lora = '_ftlora' in estimator_name
        
        # Validate estimator name structure
        is_base_model = estimator_name in ('duster', 'dust3r')
        has_valid_components = any(s in estimator_name for s in ('_calib', '_nocalib', '_pretrain', '_ftlora'))
        if not is_base_model and not has_valid_components:
            raise RuntimeError(
                f"Estimator {estimator_name} for duster not yet supported. Consider submitting a PR to add it. Available models: {available_models}"
            )

        from estimator.models import duster
        return duster.Dust3rEstimator(device, use_calib=use_calib, use_lora=use_lora, *args, **kwargs)
    
    if 'master' in estimator_name or 'mast3r' in estimator_name:
        # master, master_calib, master_calib_lora
        use_calib = '_calib' in estimator_name
        use_lora = '_lora' in estimator_name
        if not (estimator_name == 'master' or estimator_name == 'mast3r') and \
           not ('_calib' in estimator_name or '_lora' in estimator_name):
            raise RuntimeError(
                f"Estimator {estimator_name} for master not yet supported. Consider submitting a PR to add it. Available models: {available_models}"
            )

        from estimator.models import master
        # return duster.Mast3rEstimator(device, use_calib=use_calib, use_lora=use_lora, *args, **kwargs)
        return master.Mast3rEstimator(device, *args, **kwargs)

    else:
        raise RuntimeError(
            f"Estimator {estimator_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )
