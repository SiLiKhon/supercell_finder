from typing import Tuple

import numpy as np


_SL3Z_BASE_GENS = np.array([
    [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
])
SL3Z_BASE = np.concatenate([
    np.eye(3, dtype=int) + _SL3Z_BASE_GENS,
    np.eye(3, dtype=int) - _SL3Z_BASE_GENS,
], axis=0)

EPSILON = 1e-10
MAX_OPT_ITERATIONS = 1000

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

def apply_SL3Z(transforms: np.ndarray) -> np.ndarray:
    """
    Apply every basic SL(3,Z) operation defined above to each transformation in `transforms`.

    Parameters
    ----------
    transforms: np.ndarray
        Array with 3d transformation matrices defined in the last two dimensions

    Returns
    -------
    np.ndarray
    """
    ddim = transforms.ndim - 2
    assert ddim >= 0
    return SL3Z_BASE.reshape((-1,) + (1,) * ddim + (3, 3)) @ transforms[None]

def optimize_trasforms(
    lattices: np.ndarray,
    transforms: np.ndarray,
    depth: int,
    verbose: bool = False,
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
    depth: int
        How many SL(3, Z) modifications to apply at a single step
    verbose: bool
        Controls whether to print the number of iterations taken during optimization

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
    assert depth >= 1

    t_curr = transforms.copy()
    num_iterations = 0
    while True:
        assert num_iterations < MAX_OPT_ITERATIONS
        num_iterations += 1
        t_exp = apply_SL3Z(t_curr)
        for _ in range(1, depth):
            shape = t_exp.shape
            t_exp = apply_SL3Z(t_exp).reshape((len(SL3Z_BASE) * shape[0],) + shape[1:])
        tt = np.concatenate([t_curr[None], t_exp], axis=0)
        hh = calculate_interplanar_distances(tt @ lattices)  # 3 distances per structure
        hh_min = hh.min(axis=-1)  # minimal distance per structure

        # Exit condition: none of the modified transformations are better
        if (hh_min[:1] + EPSILON >= hh_min[1:]).all():
            break

        # Assign positive score to transformations with largest minimal distance
        # and select those with highest sum of distances
        score = np.where(
            np.isclose(hh_min, hh_min.max(axis=0, keepdims=True)),
            hh.sum(axis=-1),
            0,
        )

        ids_best = score.argmax(axis=0, keepdims=True)[..., None, None]
        (t_curr,) = np.take_along_axis(tt, ids_best, axis=0)
    if verbose:
        print(f"Optimized in {num_iterations} steps.")

    assert t_curr.shape == transforms.shape
    return t_curr

def find_optimal_supercell(
    lattice: np.ndarray,
    d_inner: float,
    depth: int = 1,
    max_candidates: int = 1000,
    rng_seed: int = 42,
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
    depth: int
        Depth passed to `optimize_trasforms` calls
    max_candidates: int
        Limits the number of candidate transformations to optimize. Ignored if less than
        or equals to zero.
    rng_seed: int
        Random seed for selecting `max_candidates` transformations
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

    best = np.diag(diagonal_scalings)
    if det_lower_bound != det_upper_bound:  # if so, there may be a better solution than diagonal
        assert det_lower_bound < det_upper_bound

        # We start by generating transformations of various volume changes
        target_dets = np.arange(det_lower_bound, det_upper_bound)
        if verbose:
            print("Optimizing seed...")

        seed_transform = optimize_trasforms(lattice, np.eye(3, dtype=int) * 5, depth=depth, verbose=verbose) / 5.0
        candidates = np.round(
            target_dets[:, None, None]**(1.0 / 3) * seed_transform[None]
        ).astype(int)

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
            candidates = candidates[valid_dets]

            rng = np.random.default_rng(rng_seed)
            if len(candidates) > max_candidates > 0:
                ids = rng.choice(len(candidates), max_candidates, replace=False)
                candidates = candidates[ids]
                dets = dets[ids]

            if verbose:
                print("Optimizing candidates...")
            candidates = optimize_trasforms(lattice, candidates, depth=depth, verbose=verbose)
            hh = calculate_interplanar_distances(candidates @ lattice).min(axis=-1)

            valid_hh = hh >= d_inner
            if valid_hh.any():
                candidates = candidates[valid_hh]
                dets = dets[valid_hh]
                best = candidates[dets.argmin()]
            else:
                if verbose:
                    print("Fallback to diagonal (no valid hh)")
        else:
            if verbose:
                print("Fallback to diagonal (no valid dets)")

    return best @ lattice, best
