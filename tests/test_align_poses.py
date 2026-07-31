from typing import Tuple

import numpy as np

from estimator.utils import align_poses


def _rand_pose(rng: np.random.Generator) -> np.ndarray:
    """Builds a random 4x4 camera-to-world pose with a proper rotation."""
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = rng.standard_normal(3)
    return pose


def _apply(align: Tuple[float, np.ndarray, np.ndarray], c2w: np.ndarray) -> np.ndarray:
    scale, rotation, translation = align
    out = np.eye(4)
    out[:3, :3] = rotation @ c2w[:3, :3]
    out[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
    return out


def test_two_reference_recovery_of_a_known_similarity_transform():
    rng = np.random.default_rng(10)
    vggt_refs = [_rand_pose(rng) for _ in range(2)]
    truth = (2.5, _rand_pose(rng)[:3, :3], np.array([1.0, -2.0, 0.5]))
    known_refs = [_apply(truth, pose) for pose in vggt_refs]

    aligned, (scale, rotation, translation) = align_poses(vggt_refs, known_refs)

    np.testing.assert_allclose(scale, 2.5, atol=1e-10)
    np.testing.assert_allclose(rotation, truth[1], atol=1e-10)
    np.testing.assert_allclose(translation, truth[2], atol=1e-10)
    for aligned_pose, known_pose in zip(aligned, known_refs):
        np.testing.assert_allclose(aligned_pose, known_pose, atol=1e-10)


def test_two_reference_alignment_is_stable_under_small_noise():
    # Regression for the rank-1 degeneracy: with 2 references the old
    # translation-only fit left one rotational DOF unconstrained, so 1e-3
    # jitter produced O(1) query swings (measured up to 6.3). The
    # rotation-constrained fit must keep the amplification bounded.
    rng = np.random.default_rng(11)
    vggt_refs = [_rand_pose(rng) for _ in range(2)]
    truth = (2.0, _rand_pose(rng)[:3, :3], np.array([0.3, 1.0, -0.7]))
    known_refs = [_apply(truth, pose) for pose in vggt_refs]
    query = _rand_pose(rng)

    probes = []
    for seed in range(8):
        noise_rng = np.random.default_rng(100 + seed)
        noisy_refs = []
        for pose in vggt_refs:
            noisy = pose.copy()
            noisy[:3, 3] += 1e-3 * noise_rng.standard_normal(3)
            noisy_refs.append(noisy)
        _, align = align_poses(noisy_refs, known_refs)
        probes.append(_apply(align, query)[:3, 3])

    assert np.ptp(np.array(probes), axis=0).max() < 0.05

    _, clean_align = align_poses(vggt_refs, known_refs)
    np.testing.assert_allclose(_apply(clean_align, query), _apply(truth, query), atol=1e-9)


def test_single_reference_equals_rigid_composition():
    rng = np.random.default_rng(12)
    vggt_ref, known_ref = _rand_pose(rng), _rand_pose(rng)

    _, (scale, rotation, translation) = align_poses([vggt_ref], [known_ref])

    compose = known_ref @ np.linalg.inv(vggt_ref)
    assert scale == 1.0
    np.testing.assert_allclose(rotation, compose[:3, :3], atol=1e-9)
    np.testing.assert_allclose(translation, compose[:3, 3], atol=1e-9)


def test_coincident_centers_fall_back_to_unit_scale():
    rng = np.random.default_rng(13)
    ref_a, ref_b = _rand_pose(rng), _rand_pose(rng)
    ref_b[:3, 3] = ref_a[:3, 3]
    known = [_rand_pose(rng), _rand_pose(rng)]

    _, (scale, rotation, translation) = align_poses([ref_a, ref_b], known)

    assert scale == 1.0
    assert np.isfinite(rotation).all() and np.isfinite(translation).all()


def test_rotation_stays_proper_for_reflected_inputs():
    reflection = np.eye(4)
    reflection[:3, :3] = np.diag([1.0, 1.0, -1.0])
    rng = np.random.default_rng(14)
    refs = [_rand_pose(rng), _rand_pose(rng)]
    mirrored = [reflection @ refs[0], reflection @ _rand_pose(rng)]

    _, (_, rotation, _) = align_poses(refs, mirrored)

    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-9)


def test_negative_scale_fit_falls_back_to_unit_scale():
    # Swapping the two known centers makes the centered targets anti-correlate
    # with the rotated sources, driving the least-squares scale negative; the
    # guard must fall back to the literal 1.0 instead of shrinking geometry
    # through a negative or near-zero scale.
    rng = np.random.default_rng(15)
    ref_a, ref_b = _rand_pose(rng), _rand_pose(rng)
    known_a, known_b = ref_a.copy(), ref_b.copy()
    known_a[:3, 3], known_b[:3, 3] = ref_b[:3, 3].copy(), ref_a[:3, 3].copy()

    _, (scale, rotation, translation) = align_poses([ref_a, ref_b], [known_a, known_b])

    assert scale == 1.0
    assert np.isfinite(rotation).all() and np.isfinite(translation).all()
