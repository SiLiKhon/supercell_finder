from typing import Tuple

import numpy as np


def det_i(x: np.ndarray):
    assert x.dtype in ["int32", "int64"]
    return np.round(np.linalg.det(x)).astype(int)

def calculate_interplanar_distances(lattices: np.ndarray) -> np.ndarray:
    """
    Given an array of 3d lattices, calculate distances between opposing faces
    of the corresponding cells.

    Parameters
    ----------
    lattices: np.ndarray
        Numpy array defining the lattices. Should have `ndim >= 2`. The last
        axis should contain xyz coordinates of the lattice vectors, while the
        second to last should enumerate those vectors for a single cell (so,
        `shape[-2:]` has to be `(3, 3)`).

    Returns
    -------
    distances: np.ndarray
        Array of distances for each of the input cells (the three distances
        are stacked in the last axis)
    """
    assert lattices.shape[-2:] == (3, 3)
    vol = np.linalg.det(lattices)
    cross_vecs = np.cross(lattices, np.concatenate([lattices[..., 1:, :], lattices[..., :1, :]], axis=-2))
    result = np.abs(vol[..., None] / np.linalg.norm(cross_vecs, axis=-1))
    return np.concatenate([result[..., 1:], result[..., :1]], axis=-1)


def optimize_trasforms(
    lattices: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    """
    Given source lattices and their transformations, modify transformations (by SL(3, Z) operations only),
    such that their smallest inteplanar distances are maximized.

    Parameters
    ----------
    lattices: np.ndarray
        Source lattices (should be aligned with last dimensions of `transforms`, see assertions below)
    transforms: np.ndarray
        Transformations to optimize

    Returns
    -------
    np.ndarray
        Optimized transformations
    """
    assert np.allclose(transforms, np.round(transforms))
    transforms_ddims = transforms.ndim - lattices.ndim
    assert transforms_ddims >= 0
    assert lattices.shape == transforms.shape[transforms_ddims:]
    assert lattices.shape[-2:] == (3, 3)

    t_curr = transforms.copy()

    for i in range(3):
        super_lattices = t_curr @ lattices
        orts = np.cross(
            super_lattices,
            np.concatenate([super_lattices[..., 1:, :], super_lattices[..., :1, :]], axis=-2)
        )
        orts = np.concatenate([orts[..., 1:, :], orts[..., :1, :]], axis=-2)

        j, k = (i + 1) % 3, (i + 2) % 3
        ha, hb, hc = (
            orts[..., i, :],
            orts[..., j, :],
            orts[..., k, :],
        )
        bb = (hb * hb).sum(axis=-1)
        cc = (hc * hc).sum(axis=-1)
        ab = (ha * hb).sum(axis=-1)
        ac = (ha * hc).sum(axis=-1)
        bc = (hb * hc).sum(axis=-1)

        _denom = bb * cc - bc**2
        beta = (ab * cc - ac * bc) / _denom
        gamma = (ac * bb - ab * bc) / _denom

        beta = np.stack([np.ceil(beta), np.floor(beta)], axis=0).astype(int)
        gamma = np.stack([np.ceil(gamma), np.floor(gamma)], axis=0).astype(int)

        obj = np.linalg.norm(ha - beta[:, None, ..., None] * hb - gamma[None, :, ..., None] * hc, axis=-1)
        obj = obj.reshape(4, *obj.shape[2:])
        idx_min = obj.argmin(axis=0, keepdims=True)
        idx_beta, idx_gamma = np.unravel_index(idx_min, (2, 2))
        beta = np.take_along_axis(beta, idx_beta, axis=0)
        gamma = np.take_along_axis(gamma, idx_gamma, axis=0)

        delta_transf = np.ones(shape=super_lattices.shape[:-2], dtype=int)[..., None, None] * np.eye(3, dtype=int)
        delta_transf[..., j, i] = beta
        delta_transf[..., k, i] = gamma

        t_curr = delta_transf @ t_curr

    return t_curr


def find_optimal_supercell(
    lattice: np.ndarray,
    d_inner: float,
    minimal_volume_increase: int = 1,
    max_candidates: int = 1000,
    rng_seed: int = 42,
    bandwidth: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a minimal supercell that fits a sphere of `d_inner` diameter.

    Parameters
    ----------
    lattice: np.ndarray
        Lattice defining the elementary cell
    d_inner: float
        Diameter of a sphere to fit in
    minimal_volume_increase: int
        Force lower bound on the determinant of the transformation (defaults to 1).
    max_candidates: int
        Limits the number of candidate transformations to optimize. Ignored if less than
        or equals to zero.
    rng_seed: int
        Random seed for selecting `max_candidates` transformations
    bandwidth: int
        Single search iteration will be in [D - bandwidth, D + bandwidth] determinant range.
    verbose: bool
        Controls whether to print out various info

    Returns
    -------
    supercell_lattice: np.ndarray
        Resulting supercell
    transformation: np.ndarray
        Matrix transforming the original lattice into the supercell lattice
    """
    assert lattice.shape == (3, 3)
    assert d_inner > 0
    assert isinstance(minimal_volume_increase, int) and minimal_volume_increase >= 1
    rng = np.random.default_rng(rng_seed)

    # The smallest possible cell fitting the sphere is a cube with side = d_inner.
    # Hence, this is the minimal volume change we need.
    det_lower_bound = np.ceil(
        d_inner**3 / np.abs(np.linalg.det(lattice))
    ).astype(int)

    # The simplest (possibly sub-optimal) solution is to extend the original cell
    # along its axes. This defines the maximal volume change we need.
    hh = calculate_interplanar_distances(lattice)
    diagonal_scalings = np.ceil(d_inner / hh).astype(int)
    det_upper_bound = diagonal_scalings.prod()

    # Apply additional constraints due to minimal increase requirement:
    det_lower_bound = max(det_lower_bound, minimal_volume_increase)
    while minimal_volume_increase > det_upper_bound:
        diagonal_scalings += 1
        det_upper_bound = diagonal_scalings.prod()

    if verbose:
        print("Setting best to diagonal")
    best = np.diag(diagonal_scalings)
    if det_lower_bound != det_upper_bound:  # if so, there may be a better solution than diagonal
        assert det_lower_bound < det_upper_bound

        # We start by generating transformations of various volume changes
        target_dets = np.arange(det_lower_bound, det_upper_bound)
        if verbose:
            print("Optimizing seed...")


        seed_transform = optimize_trasforms(lattice, best) / det_i(best)**(1.0 / 3)
        candidates = np.round(
            target_dets[:, None, None]**(1.0 / 3) * seed_transform[None]
        ).astype(int)

        candidates_structured = candidates.reshape(-1).view(
            np.dtype([(str(i), candidates.dtype) for i in range(9)])
        )
        assert candidates_structured.shape == (len(candidates),)
        _, unique_ids = np.unique(candidates_structured, return_index=True)
        candidates = candidates[unique_ids]

        # Augment the transformation candidates with the neighbor lattice vector options
        # (this scales up the number of candidates by a factor of ~20k)
        augmentations = (-1, 0, 1)
        augmentations = np.stack(
            np.meshgrid(*[augmentations] * 9),
            axis=-1,
        ).reshape(len(augmentations)**9, 3, 3)
        candidates = (candidates[None] + augmentations[:, None]).reshape(-1, 3, 3)

        # Check which candidates fit into the required volume change limitations
        dets = det_i(candidates)
        valid_dets = (det_lower_bound <= dets) & (dets < det_upper_bound)
        if valid_dets.any():
            dets = dets[valid_dets]
            candidates_pool = candidates[valid_dets]

            window_min = dets.min()
            window_max = dets.max()
            while True:
                remaining_dets = dets[(dets >= window_min) & (dets <= window_max)]
                if len(remaining_dets) == 0:
                    if verbose:
                        print("Can't improve further")
                    break
                attempt_det = np.quantile(
                    np.unique(remaining_dets), 0.5, method="closest_observation"
                )

                selection = (
                    dets >= max(attempt_det - bandwidth, window_min)
                ) & (
                    dets <= min(attempt_det + bandwidth, window_max)
                )
                current_candidates = candidates_pool[selection]
                current_dets = dets[selection]

                assert len(current_candidates) > 0
                if len(current_candidates) > max_candidates > 0:
                    ids = rng.choice(len(current_candidates), max_candidates, replace=False)
                    current_candidates = current_candidates[ids]
                    current_dets = current_dets[ids]

                if verbose:
                    print("Optimizing candidates...")
                current_candidates = optimize_trasforms(lattice, current_candidates)
                hh = calculate_interplanar_distances(current_candidates @ lattice).min(axis=-1)

                valid_hh = hh >= d_inner
                if valid_hh.any():
                    if verbose:
                        print("Found improvement")

                    best = current_candidates[valid_hh][
                        current_dets[valid_hh].argmin()
                    ]
                    window_max = det_i(best) - 1
                else:
                    window_min = current_dets.max() + 1
        else:
            if verbose:
                print("Fallback to diagonal (no valid dets)")

    return best @ lattice, best
