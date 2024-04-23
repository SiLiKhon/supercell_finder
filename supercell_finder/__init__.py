from pymatgen.core import Structure

from .core import find_optimal_supercell


def make_supercell(
    structure: Structure,
    d_inner: float,
    **kwargs,
) -> Structure:
    _, supercell_transformation = find_optimal_supercell(
        lattice=structure.lattice.matrix,
        d_inner=d_inner,
        **kwargs,
    )
    return structure * supercell_transformation
