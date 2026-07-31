import numpy as np

from estimator.models.vggt import VggtEstimator


def _rand_pose(rng: np.random.Generator) -> np.ndarray:
    """Builds a random 4x4 camera-to-world pose with a proper rotation."""
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = rng.standard_normal(3)
    return pose


def test_single_reference_alignment_is_rigid_and_anchors_the_reference():
    rng = np.random.default_rng(0)
    vggt_ref, known_ref = _rand_pose(rng), _rand_pose(rng)

    align = VggtEstimator._compute_alignment([vggt_ref], [known_ref])

    assert align[0] == 1.0
    np.testing.assert_allclose(VggtEstimator._align_pose(align, vggt_ref), known_ref, atol=1e-9)


def test_multi_reference_alignment_recovers_a_known_similarity_transform():
    rng = np.random.default_rng(1)
    vggt_refs = [_rand_pose(rng) for _ in range(4)]
    truth = (2.5, _rand_pose(rng)[:3, :3], np.array([1.0, -2.0, 0.5]))
    known_refs = [VggtEstimator._align_pose(truth, pose) for pose in vggt_refs]

    align = VggtEstimator._compute_alignment(vggt_refs, known_refs)

    np.testing.assert_allclose(align[0], 2.5, atol=1e-9)
    for vggt_pose, known_pose in zip(vggt_refs, known_refs):
        np.testing.assert_allclose(
            VggtEstimator._align_pose(align, vggt_pose), known_pose, atol=1e-9
        )


def test_align_points_agrees_with_align_pose_on_camera_centers():
    rng = np.random.default_rng(2)
    align = (1.7, _rand_pose(rng)[:3, :3], np.array([0.3, 0.4, -1.2]))
    pose = _rand_pose(rng)

    aligned_center = VggtEstimator._align_points(align, pose[:3, 3].reshape(1, 1, 3))

    np.testing.assert_allclose(
        aligned_center[0, 0], VggtEstimator._align_pose(align, pose)[:3, 3], atol=1e-9
    )


def test_align_points_preserves_leading_dimensions():
    rng = np.random.default_rng(3)
    align = (0.5, _rand_pose(rng)[:3, :3], np.zeros(3))

    assert VggtEstimator._align_points(align, rng.standard_normal((4, 5, 3))).shape == (4, 5, 3)


def test_confidence_masks_drop_the_requested_percentile():
    conf = np.arange(1, 101, dtype=np.float64).reshape(1, 10, 10)

    masks = VggtEstimator._confidence_masks(conf, 50.0)

    assert len(masks) == 1
    assert masks[0].sum() == 50


def test_confidence_masks_threshold_is_global_not_per_frame():
    conf = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 9.0)])

    masks = VggtEstimator._confidence_masks(conf, 50.0)

    assert not masks[0].any()
    assert masks[1].all()


def test_confidence_masks_keep_everything_at_zero_threshold():
    masks = VggtEstimator._confidence_masks(np.full((2, 3, 3), 0.7), 0.0)

    assert len(masks) == 2
    assert all(mask.all() for mask in masks)


def test_confidence_masks_always_drop_near_zero_confidence():
    masks = VggtEstimator._confidence_masks(np.zeros((1, 2, 2)), 0.0)

    assert not masks[0].any()
