from typing import Optional

import numpy as np
from pymatgen.core import Structure

from .core import find_optimal_supercell


def make_supercell(
    structure: Structure,
    d_inner: float,
    min_number_of_atoms: Optional[int] = None,
    **kwargs,
) -> Structure:
    if min_number_of_atoms is not None:
        assert "minimal_volume_increase" not in kwargs
        kwargs["minimal_volume_increase"] = int(
            np.ceil(min_number_of_atoms / len(structure))
        )
    _, supercell_transformation = find_optimal_supercell(
        lattice=structure.lattice.matrix,
        d_inner=d_inner,
        **kwargs,
    )
    return structure * supercell_transformation
