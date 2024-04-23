import numpy as np

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
        are stacked in the last axis).
    """
    assert lattices.shape[-2:] == (3, 3)
    vol = np.linalg.det(lattices)
    cross_vecs = np.cross(lattices, np.concatenate([lattices[..., 1:, :], lattices[..., :1, :]], axis=-2))
    result = np.abs(vol[..., None] / np.linalg.norm(cross_vecs, axis=-1))
    return np.concatenate([result[..., 1:], result[..., :1]], axis=-1)
