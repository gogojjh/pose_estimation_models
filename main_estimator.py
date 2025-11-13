"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import sys
import torch
import argparse
import matplotlib
from pathlib import Path
import time
import numpy as np
import pycolmap
import random
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from estimator.utils import get_image_pairs_paths
from estimator import get_estimator, available_models
from estimator import BaseEstimator

# This is to be able to use matplotlib also without a GUI
# if not hasattr(sys, "ps1"):
#     matplotlib.use("Agg")

##### Load images
# Matterport3d
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/matterport3d/map_free_eval/test/s00000/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([640, 360]) # WxH

# Replica
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/replica/')
# K = np.array([[205.46963, 0.0, 320], [0.0, 205.46963, 180], [0.0, 0.0, 1.0]])
# im_size = np.array([360, 640])

# ucl_campus
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/ucl_campus_aria/map_free_eval/test/s00006/')
# K = np.array([[504.79, 0.0, 481.30], [0.0, 542.79, 271.85], [0.0, 0.0, 1.0]])
# im_size = np.array([576, 1024]) # HxW

# map_free
# scene_root = Path('/Titan/dataset/data_litevloc/data_tro2025/map_free_eval/mapfree/map_free_eval/val/s00460')
# K = np.array([[547.9946, 0.0, 269.9052], [0.0, 547.9946, 352.2056], [0.0, 0.0, 1.0]])
# im_size = np.array([720, 540]) # HxW

# hkustgz_campus
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/test_gray/s00000')
# K = np.array([[913.896, 0.0, 638.954], [0.0, 912.277, 364.884], [0.0, 0.0, 1.0]])
# im_size = np.array([1280, 720])

# 360loc_aria
scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_aria/map_free_eval/test/s00019/')

# 360loc_device1
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device1/map_free_eval/test/s00000/')

# 360loc_device2
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device2/map_free_eval/test/s00001/')

# 360loc_device3
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device3/map_free_eval/test/s00002/')

# 360loc_device4
# scene_root = Path('/Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_device4/map_free_eval/test/s00000/')

est_opts = {
    'known_extrinsics': True,
    'known_intrinsics': False,
    'niter': 300,
    'two_stage_opt_niter': 50,
    'crop_image_to_database': True,
    'resize': (512, 288),
}

def visualize_images(images):
    """
    Visualize list of images and a query image using matplotlib.
    
    Args:
        images: List of images (can be torch.Tensor or numpy arrays)
        img1: Query image (can be torch.Tensor or numpy array)
    """
    import matplotlib.pyplot as plt
    
    num_img0 = len(images) - 1
    fig, axs = plt.subplots(1, num_img0 + 1, figsize=(5 * (num_img0 + 1), 5))
    # Ensure axs is always iterable
    if (num_img0 + 1) == 1:
        axs = [axs]

    for i, img in enumerate(images[:-1]):
        if isinstance(img, torch.Tensor):
            img_vis = img.detach().cpu().numpy()
            if img_vis.shape[0] in (1, 3):  # CHW
                img_vis = np.transpose(img_vis, (1,2,0))
            if img_vis.shape[2] == 1:
                img_vis = img_vis[..., 0]
        else:
            img_vis = img
        axs[i].imshow(img_vis.astype(np.uint8) if img_vis.max() > 1 else img_vis)
        axs[i].axis('off')
        axs[i].set_title(f'image[{i}]')

    # Handle img1
    if isinstance(images[-1], torch.Tensor):
        img1_vis = images[-1].detach().cpu().numpy()
        if img1_vis.shape[0] in (1, 3):
            img1_vis = np.transpose(img1_vis, (1,2,0))
        if img1_vis.shape[2] == 1:
            img1_vis = img1_vis[..., 0]
    else:
        img1_vis = images[-1]
    axs[-1].imshow(img1_vis.astype(np.uint8) if img1_vis.max() > 1 else img1_vis)
    axs[-1].axis('off')
    axs[-1].set_title('image[-1]')

    plt.tight_layout()
    plt.show()


def visualize_optimization_analysis(scene, out_dir, warmup_iters=None):
    """
    Comprehensive visualization to understand warmup importance and outlier rejection.
    
    Args:
        scene: The optimizer scene object with loss_log, weight_i, weight_j, conf_i, conf_j
        out_dir: Directory to save visualizations
        warmup_iters: Number of warmup iterations (if used)
    """
    # print("\n" + "="*70)
    # print("📊 OPTIMIZATION ANALYSIS & VISUALIZATION")
    # print("="*70)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # 1. LOSS CURVE ANALYSIS - Why warmup matters
    # ========================================================================
    if hasattr(scene, 'loss_log') and len(scene.loss_log) > 0:
        # print("\n1️⃣ Loss Curve Analysis (Warmup Importance)")
        # print("-"*70)
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        losses = np.array(scene.loss_log)
        iterations = np.arange(len(losses))
        
        # Plot 1: Full loss curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(iterations, losses, 'b-', linewidth=2, alpha=0.7, label='Loss')
        if warmup_iters and warmup_iters < len(losses):
            ax1.axvline(x=warmup_iters, color='r', linestyle='--', linewidth=2, 
                       label=f'Warmup End (iter {warmup_iters})')
            ax1.axvspan(0, warmup_iters, alpha=0.2, color='yellow', label='Warmup Phase')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Progression (Why Warmup Matters)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Log-scale loss (shows early convergence better)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.semilogy(iterations, losses, 'g-', linewidth=2, alpha=0.7)
        if warmup_iters and warmup_iters < len(losses):
            ax2.axvline(x=warmup_iters, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=10)
        ax2.set_ylabel('Loss (log scale)', fontsize=10)
        ax2.set_title('Log-Scale Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss gradient (rate of change)
        ax3 = fig.add_subplot(gs[1, 1])
        if len(losses) > 1:
            loss_grad = np.gradient(losses)
            ax3.plot(iterations, loss_grad, 'r-', linewidth=2, alpha=0.7)
            if warmup_iters and warmup_iters < len(losses):
                ax3.axvline(x=warmup_iters, color='r', linestyle='--', linewidth=2)
            ax3.set_xlabel('Iteration', fontsize=10)
            ax3.set_ylabel('Loss Gradient', fontsize=10)
            ax3.set_title('Rate of Convergence', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Plot 4: Warmup vs Post-warmup statistics
        ax4 = fig.add_subplot(gs[1, 2])
        if warmup_iters and warmup_iters < len(losses):
            warmup_losses = losses[:warmup_iters]
            postwarmup_losses = losses[warmup_iters:]
            
            stats_data = [
                [np.mean(warmup_losses), np.mean(postwarmup_losses)],
                [np.std(warmup_losses), np.std(postwarmup_losses)],
                [np.max(warmup_losses) - np.min(warmup_losses), 
                 np.max(postwarmup_losses) - np.min(postwarmup_losses)]
            ]
            
            x = np.arange(3)
            width = 0.35
            ax4.bar(x - width/2, [stats_data[i][0] for i in range(3)], width, 
                   label='Warmup', color='orange', alpha=0.7)
            ax4.bar(x + width/2, [stats_data[i][1] for i in range(3)], width, 
                   label='Post-Warmup', color='blue', alpha=0.7)
            ax4.set_ylabel('Value', fontsize=10)
            ax4.set_title('Phase Comparison', fontsize=12)
            ax4.set_xticks(x)
            ax4.set_xticklabels(['Mean Loss', 'Std Dev', 'Range'], fontsize=9)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Convergence speed analysis
        ax5 = fig.add_subplot(gs[2, :])
        if len(losses) > 10:
            window_size = max(5, len(losses) // 20)
            smoothed_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_iters = iterations[:len(smoothed_loss)]
            ax5.plot(iterations, losses, 'b-', alpha=0.3, label='Raw Loss')
            ax5.plot(smoothed_iters, smoothed_loss, 'b-', linewidth=3, label='Smoothed Loss')
            if warmup_iters and warmup_iters < len(losses):
                ax5.axvline(x=warmup_iters, color='r', linestyle='--', linewidth=2, 
                           label=f'Warmup End')
            ax5.set_xlabel('Iteration', fontsize=12)
            ax5.set_ylabel('Loss', fontsize=12)
            ax5.set_title('Smoothed Convergence (Shows Stability)', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=10)
        
        plt.savefig(out_dir / 'optimization_loss_analysis.png', dpi=150, bbox_inches='tight')
        # print(f"✅ Saved: {out_dir / 'optimization_loss_analysis.png'}")
        
        # Print statistics
        # print(f"\n📈 Loss Statistics:")
        # print(f"   Initial loss: {losses[0]:.6f}")
        # print(f"   Final loss: {losses[-1]:.6f}")
        # print(f"   Reduction: {(losses[0] - losses[-1])/losses[0]*100:.2f}%")
        if warmup_iters and warmup_iters < len(losses):
            # print(f"   Loss at warmup end: {losses[warmup_iters]:.6f}")
            # print(f"   Warmup reduction: {(losses[0] - losses[warmup_iters])/losses[0]*100:.2f}%")
            pass
        
        plt.close()
    
    # ========================================================================
    # 2. CONFIDENCE & WEIGHT MAP ANALYSIS - Outlier rejection
    # ========================================================================
    # print("\n2️⃣ Confidence & Weight Map Analysis (Outlier Rejection)")
    # print("-"*70)
    
    if hasattr(scene, 'weight_i') and hasattr(scene, 'conf_i'):
        # Analyze first edge (you can loop through all edges if needed)
        edge_keys = list(scene.weight_i.keys())
        
        for idx, edge_key in enumerate(edge_keys[:min(3, len(edge_keys))]):  # Show up to 3 edges
            # print(f"\n🔗 Analyzing Edge: {edge_key}")
            
            weight_i = scene.weight_i[edge_key].detach().cpu().numpy()
            conf_i = scene.conf_i[edge_key].detach().cpu().numpy()
            
            fig = plt.figure(figsize=(18, 10))
            gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
            
            # Plot 1: Raw confidence map
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(conf_i, cmap='jet', aspect='auto')
            ax1.set_title(f'Raw Confidence Map\n(Edge {edge_key})', fontsize=11)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            ax1.axis('off')
            
            # Plot 2: Final weight map
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(weight_i, cmap='jet', aspect='auto')
            ax2.set_title(f'Final Weight Map\n(After Optimization)', fontsize=11)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            ax2.axis('off')
            
            # Plot 3: Weight/Confidence ratio (shows adjustment)
            ax3 = fig.add_subplot(gs[0, 2])
            conf_trf = scene.conf_trf(torch.from_numpy(conf_i)).numpy()
            ratio = weight_i / (conf_trf + 1e-8)
            im3 = ax3.imshow(ratio, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')
            ax3.set_title(f'Weight/Conf Ratio\n(>1: upweighted, <1: downweighted)', fontsize=11)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            ax3.axis('off')
            
            # Plot 4: Outlier mask (if threshold is used)
            ax4 = fig.add_subplot(gs[0, 3])
            if hasattr(scene, 'CONF_THRE'):
                outlier_mask = weight_i < scene.CONF_THRE
                im4 = ax4.imshow(outlier_mask, cmap='RdYlGn_r', aspect='auto')
                ax4.set_title(f'Outlier Mask\n(Rejected: {outlier_mask.sum()}/{outlier_mask.size} = {outlier_mask.sum()/outlier_mask.size*100:.1f}%)', 
                             fontsize=11)
                plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            else:
                ax4.text(0.5, 0.5, 'No Threshold\nUsed', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=14)
            ax4.axis('off')
            
            # Plot 5: Confidence histogram
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.hist(conf_i.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax5.set_xlabel('Confidence Value', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.set_title('Confidence Distribution', fontsize=11)
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Weight histogram
            ax6 = fig.add_subplot(gs[1, 1])
            ax6.hist(weight_i.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
            if hasattr(scene, 'CONF_THRE'):
                ax6.axvline(x=scene.CONF_THRE, color='r', linestyle='--', linewidth=2, 
                           label=f'Threshold={scene.CONF_THRE:.3f}')
                ax6.legend(fontsize=9)
            ax6.set_xlabel('Weight Value', fontsize=10)
            ax6.set_ylabel('Frequency', fontsize=10)
            ax6.set_title('Weight Distribution', fontsize=11)
            ax6.grid(True, alpha=0.3)
            
            # Plot 7: 2D scatter (Confidence vs Weight)
            ax7 = fig.add_subplot(gs[1, 2])
            conf_flat = conf_trf.flatten()[::10]  # Subsample for speed
            weight_flat = weight_i.flatten()[::10]
            ax7.scatter(conf_flat, weight_flat, alpha=0.3, s=1)
            ax7.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (no change)')
            if hasattr(scene, 'CONF_THRE'):
                ax7.axhline(y=scene.CONF_THRE, color='orange', linestyle='--', 
                           linewidth=2, label=f'Threshold')
            ax7.set_xlabel('Transformed Confidence', fontsize=10)
            ax7.set_ylabel('Final Weight', fontsize=10)
            ax7.set_title('Confidence vs Weight\n(Shows Outlier Downweighting)', fontsize=11)
            ax7.legend(fontsize=9)
            ax7.grid(True, alpha=0.3)
            
            # Plot 8: Statistics comparison
            ax8 = fig.add_subplot(gs[1, 3])
            stats_labels = ['Mean', 'Median', 'Std', 'Max', 'Min']
            conf_stats = [np.mean(conf_i), np.median(conf_i), np.std(conf_i), 
                         np.max(conf_i), np.min(conf_i)]
            weight_stats = [np.mean(weight_i), np.median(weight_i), np.std(weight_i), 
                           np.max(weight_i), np.min(weight_i)]
            
            x = np.arange(len(stats_labels))
            width = 0.35
            ax8.bar(x - width/2, conf_stats, width, label='Confidence', alpha=0.7)
            ax8.bar(x + width/2, weight_stats, width, label='Weight', alpha=0.7)
            ax8.set_ylabel('Value', fontsize=10)
            ax8.set_title('Statistical Comparison', fontsize=11)
            ax8.set_xticks(x)
            ax8.set_xticklabels(stats_labels, rotation=45, fontsize=9)
            ax8.legend(fontsize=9)
            ax8.grid(True, alpha=0.3, axis='y')
            
            # Plot 9-12: Spatial patterns (showing where outliers are)
            ax9 = fig.add_subplot(gs[2, 0])
            # High confidence regions
            high_conf_mask = conf_i > np.percentile(conf_i, 75)
            ax9.imshow(high_conf_mask, cmap='Greens', aspect='auto')
            ax9.set_title(f'High Confidence Regions\n({high_conf_mask.sum()} pixels)', fontsize=11)
            ax9.axis('off')
            
            ax10 = fig.add_subplot(gs[2, 1])
            # Low confidence (potential outliers)
            low_conf_mask = conf_i < np.percentile(conf_i, 25)
            ax10.imshow(low_conf_mask, cmap='Reds', aspect='auto')
            ax10.set_title(f'Low Confidence Regions\n({low_conf_mask.sum()} pixels)', fontsize=11)
            ax10.axis('off')
            
            ax11 = fig.add_subplot(gs[2, 2])
            # Final inliers (after weighting)
            if hasattr(scene, 'CONF_THRE'):
                inlier_mask = weight_i >= scene.CONF_THRE
                ax11.imshow(inlier_mask, cmap='RdYlGn', aspect='auto')
                ax11.set_title(f'Final Inliers\n({inlier_mask.sum()} pixels = {inlier_mask.sum()/inlier_mask.size*100:.1f}%)', 
                              fontsize=11)
            else:
                inlier_mask = weight_i > np.median(weight_i)
                ax11.imshow(inlier_mask, cmap='RdYlGn', aspect='auto')
                ax11.set_title(f'High Weight Regions\n({inlier_mask.sum()} pixels)', fontsize=11)
            ax11.axis('off')
            
            ax12 = fig.add_subplot(gs[2, 3])
            # Weight adjustment magnitude
            adjustment = np.abs(weight_i - conf_trf)
            im12 = ax12.imshow(adjustment, cmap='plasma', aspect='auto')
            ax12.set_title(f'Weight Adjustment Magnitude\n(How much changed)', fontsize=11)
            plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)
            ax12.axis('off')
            
            fig.suptitle(f'Confidence & Weight Analysis - Edge {edge_key}\n' + 
                        f'Shows How Optimization Identifies and Downweights Outliers', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            plt.savefig(out_dir / f'confidence_weight_analysis_edge_{idx}.png', dpi=150, bbox_inches='tight')
            # print(f"✅ Saved: {out_dir / f'confidence_weight_analysis_edge_{idx}.png'}")
            
            # Print edge statistics
            # print(f"   Confidence: mean={np.mean(conf_i):.4f}, std={np.std(conf_i):.4f}")
            # print(f"   Weight: mean={np.mean(weight_i):.4f}, std={np.std(weight_i):.4f}")
            if hasattr(scene, 'CONF_THRE'):
                outlier_ratio = (weight_i < scene.CONF_THRE).sum() / weight_i.size
                # print(f"   Outliers rejected: {outlier_ratio*100:.2f}%")
                pass
            
            plt.close()
    
    # print("\n" + "="*70)
    # print("✅ Optimization analysis complete!")
    # print(f"📁 All visualizations saved to: {out_dir}")
    # print("="*70)

def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    estimator = get_estimator(
        args.model, 
        device=args.device, 
        max_num_keypoint=args.max_num_keypoint, 
        out_dir=args.out_dir
    )
    estimator.verbose = True

    for i in range(1):
        ##### Set image names
        list_img0_name = [
            'seq1/frame_00003.jpg',
            'seq1/frame_00024.jpg'
        ]
        list_img0_name = list_img0_name[:]
        img1_name = 'seq0/frame_00000.jpg'

        ##### Load poses and intrinsics
        poses_load = {}
        with (scene_root / 'poses.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line: continue
                line = line.strip().split(' ')
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                pose = pycolmap.Rigid3d()
                pose.translation = qt[4:]
                pose.rotation = pycolmap.Rotation3d(np.roll(qt[:4], -1))
                poses_load[img_name] = pose

        intr_load = {}
        with (scene_root / 'intrinsics.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line: continue
                line = line.strip().split(' ')
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])
                intr_load[img_name] = {'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), 'im_size': np.array([W, H])}

        list_img0_poses = [] # Pose from world to camera
        for name in list_img0_name:
            pose = np.eye(4)
            pose[:3, :] = poses_load[name].matrix()
            list_img0_poses.append(torch.from_numpy(np.linalg.inv(pose)))

        list_img0_intr = [{'K': torch.from_numpy(intr_load[name]['K']), 'im_size': torch.from_numpy(intr_load[name]['im_size'])} for name in list_img0_name]
        img1_intr = {'K': torch.from_numpy(intr_load[img1_name]['K']), 'im_size': torch.from_numpy(intr_load[img1_name]['im_size'])}

        ##### Perform pose estimation
        try:
            start_time = time.time()
            result = estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
            print(f"Processing time: {time.time() - start_time:.2f}s")
            print(f"Estimated pose_w2c: {result['im_pose'][:3, 3:4].T}")
            # print(f"Edge score: {edge_scores}")
            # print(f"Focal length: {result['focal'][0]:.03f}")
            # print(f"Loss: {result['loss']:.03f}")
        except Exception as e:
            print(f"Error: {e}")
            pass

        msp_edges = estimator.get_minimum_spanning_tree()
        weight_i, weight_j = estimator.scene.weight_i, estimator.scene.weight_j
        for edge in msp_edges:
            if edge[0] == 2 or edge[1] == 2: # confidence of the query image
                edge_str = f"{edge[0]}_{edge[1]}"
                conf = (weight_i[edge_str].mean() * weight_j[edge_str].mean()).detach().cpu().item()
                print(f"Conf of {edge_str}: {conf:.3f}")

        # ========================================================================
        # VISUALIZATION: Analyze optimization dynamics and outlier rejection
        # ========================================================================
        if hasattr(estimator, 'scene') and estimator.scene is not None:
            warmup_iters = None
            if hasattr(estimator.scene, 'calib_params') and estimator.scene.calib_params is not None:
                warmup_iters = estimator.scene.calib_params.get('warmup_iters', None)
            
            visualize_optimization_analysis(
                estimator.scene, 
                args.out_dir / 'optimization_analysis',
                warmup_iters=warmup_iters
            )
        
        estimator.show_reconstruction()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose Estimator Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Choose estimator
    parser.add_argument(
        "--model",
        type=str,
        default="master",
        help=f"choose your model: {available_models}",
    )

    # Hyperparameters shared by all methods:
    # parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")
    parser.add_argument("--max_num_keypoint", type=int, default=2048, help="maximum number of keypoints")
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.model}")

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
