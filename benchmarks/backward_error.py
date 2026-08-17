"""Reports the backward error of the warping solve for a range of sections.

Used to calibrate the tolerance in
:func:`~sectionproperties.analysis.solver.solve_warping_direct`.
"""

from __future__ import annotations

import numpy as np

from sectionproperties.analysis import solver
from sectionproperties.analysis.section import Section
from sectionproperties.pre import Material
from sectionproperties.pre.library import (
    channel_section,
    circular_hollow_section,
    circular_section_by_area,
    elliptical_section,
    rectangular_section,
)

STEEL = Material("Steel", 200e3, 0.3, 7.85e-6, 250, "grey")
TIMBER = Material("Timber", 8e3, 0.35, 6.5e-7, 20, "burlywood")


def composite() -> Section:
    """A two-material section."""
    bottom = rectangular_section(d=50, b=100, material=STEEL)
    top = rectangular_section(d=100, b=100, material=TIMBER).shift_section(y_offset=50)
    geom = bottom + top
    geom.create_mesh(mesh_sizes=[20.0, 20.0])
    return Section(geometry=geom)


def thin_strip() -> Section:
    """A very slender rectangle."""
    geom = rectangular_section(d=0.1, b=100)
    geom.create_mesh(mesh_sizes=[0.05])
    return Section(geometry=geom)


CASES = {
    "solid circle (no warping)": lambda: _mesh(circular_section_by_area(area=np.pi * 2500, n=64), 15.0),
    "solid circle d=150": lambda: _mesh(circular_section_by_area(area=np.pi * 150**2 / 4, n=64), 35.0),
    "chs": lambda: _mesh(circular_hollow_section(d=200, t=10, n=64), 5.0),
    "ellipse": lambda: _mesh(elliptical_section(d_x=100, d_y=50, n=64), 5.0),
    "channel": lambda: _mesh(channel_section(d=200, b=75, t_f=12, t_w=6, r=12, n_r=8), 5.0),
    "rectangle": lambda: _mesh(rectangular_section(d=200, b=100), 10.0),
    "thin strip": thin_strip,
    "composite": composite,
}


def _mesh(geom, mesh_size: float) -> Section:  # noqa: ANN001
    """Meshes a geometry and wraps it in a Section."""
    geom.create_mesh(mesh_sizes=[mesh_size])
    return Section(geometry=geom)


def main() -> None:
    """Print the norm-wise backward error of every warping solve."""
    print(f"pardiso: {solver._HAS_PARDISO}")  # noqa: SLF001
    print(f"{'case':<28}{'nodes':>8}{'backward error (torsion, psi, phi)':>44}")

    for label, factory in CASES.items():
        section = factory()
        section.calculate_geometric_properties()

        batch = section.element_batch(origin=section.get_c())
        k, c, f_torsion = section.assemble_warping(batch=batch)

        ixx, iyy, ixy = section.get_eic() if section.is_composite() else section.get_ic()
        nu = section.get_nu_eff() if section.is_composite() else 0.0

        from sectionproperties.analysis import batch as sp_batch  # noqa: PLC0415

        f_psi_el, f_phi_el = sp_batch.shear_load_vectors(
            batch=batch,
            elastic_modulus=section.element_elastic_modulus,
            ixx=ixx,
            iyy=iyy,
            ixy=ixy,
            nu=nu,
        )
        flat = batch.node_ids.ravel()
        f_psi = np.bincount(flat, weights=f_psi_el.ravel(), minlength=section.num_nodes)
        f_phi = np.bincount(flat, weights=f_phi_el.ravel(), minlength=section.num_nodes)

        f = np.column_stack((f_torsion, f_psi, f_phi))

        n_size = k.shape[0]
        pinned = int(np.argmax(k.diagonal()))
        free = np.ones(n_size, dtype=bool)
        free[pinned] = False
        from scipy.sparse import csc_matrix  # noqa: PLC0415

        k_free = csc_matrix(k.tocsr()[free][:, free])
        u = np.zeros((n_size, f.shape[1]))
        u[free, :] = solver.sp_solve_multi(k=k_free, b=f[free, :])
        u -= (c @ u) / c.sum()

        residual = np.abs(k @ u - f).max(axis=0)
        scale = (abs(k) @ np.abs(u) + np.abs(f)).max(axis=0)
        relative = residual / scale

        errors = "  ".join(f"{value:.2e}" for value in relative)
        print(f"{label:<28}{section.num_nodes:>8}{errors:>44}")
        print(f"{'':<28}{'':>8}  |f|max = " + "  ".join(f"{v:.2e}" for v in np.abs(f).max(axis=0)))
        print(f"{'':<28}{'':>8}  |u|max = " + "  ".join(f"{v:.2e}" for v in np.abs(u).max(axis=0)))


if __name__ == "__main__":
    main()
