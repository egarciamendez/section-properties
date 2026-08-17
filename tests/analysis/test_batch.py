"""Tests that the vectorised kernels reproduce the element-wise ``Tri6`` results.

Several of the mesh-wide integrals (the shear centre and warping integrals in
particular) are sums of large contributions that almost completely cancel.  For
those the only meaningful comparison is against the magnitude that went into the
sum, not against the result, so the tolerances below are expressed relative to
``sum(abs(contributions))``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sectionproperties.analysis import batch
from sectionproperties.analysis.fea import Tri6
from sectionproperties.analysis.section import Section
from sectionproperties.pre import Material
from sectionproperties.pre.library import channel_section, rectangular_section

STEEL = Material(
    name="Steel",
    elastic_modulus=200e3,
    poissons_ratio=0.3,
    density=7.85e-6,
    yield_strength=250,
    color="grey",
)
TIMBER = Material(
    name="Timber",
    elastic_modulus=8e3,
    poissons_ratio=0.35,
    density=6.5e-7,
    yield_strength=20,
    color="burlywood",
)

# relative accuracy expected of a single vectorised operation
RTOL = 1e-10

# accuracy expected of a mesh-wide sum, relative to the magnitude summed
SUM_RTOL = 1e-12


def build_section(composite: bool) -> Section:
    """Builds a small meshed section, optionally with two materials."""
    if composite:
        bottom = rectangular_section(d=50, b=100, material=STEEL)
        top = rectangular_section(d=100, b=100, material=TIMBER).shift_section(y_offset=50)
        geom = bottom + top
        geom.create_mesh(mesh_sizes=[20.0, 20.0])
    else:
        geom = channel_section(d=200, b=75, t_f=12, t_w=6, r=12, n_r=8)
        geom.create_mesh(mesh_sizes=[10.0])

    return Section(geometry=geom)


def element_arrays(section: Section) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the element coordinates, node ids and elastic moduli as arrays."""
    coords = np.array([el.coords for el in section.elements])
    node_ids = np.array([el.node_ids for el in section.elements])
    elastic_modulus = np.array([el.material.elastic_modulus for el in section.elements])

    return coords, node_ids, elastic_modulus


def assert_close(actual: np.ndarray, expected: np.ndarray, rtol: float = RTOL) -> None:
    """Asserts equality with an absolute tolerance derived from the value scale."""
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    assert np.allclose(actual, expected, rtol=rtol, atol=rtol * scale)


def assert_sum_close(actual: np.ndarray, contributions: np.ndarray) -> None:
    """Asserts a mesh-wide sum matches, allowing for cancellation in the sum."""
    expected = contributions.sum(axis=0)
    tolerance = SUM_RTOL * np.abs(contributions).sum(axis=0)
    assert np.all(np.abs(actual - expected) <= tolerance)


@pytest.mark.parametrize("composite", [False, True])
def test_geometric_properties_match(composite: bool) -> None:
    """Batch geometric properties equal the element-wise results."""
    section = build_section(composite=composite)
    coords, node_ids, _ = element_arrays(section)
    element_batch = batch.build_batch(coords=coords, node_ids=node_ids)

    expected = np.array([el.geometric_properties()[:6] for el in section.elements])
    actual = np.column_stack(batch.geometric_properties(element_batch))

    assert_close(actual, expected)


@pytest.mark.parametrize("composite", [False, True])
def test_torsion_properties_match(composite: bool) -> None:
    """Batch torsion matrices and vectors equal the element-wise results."""
    section = build_section(composite=composite)
    coords, node_ids, elastic_modulus = element_arrays(section)
    element_batch = batch.build_batch(coords=coords, node_ids=node_ids)

    k_el, f_el, c_el = batch.torsion_properties(element_batch, elastic_modulus)

    expected = [el.torsion_properties() for el in section.elements]
    assert_close(k_el, np.array([item[0] for item in expected]))
    assert_close(f_el, np.array([item[1] for item in expected]))
    assert_close(c_el, np.array([item[2] for item in expected]))


@pytest.mark.parametrize("composite", [False, True])
def test_shear_load_vectors_match(composite: bool) -> None:
    """Batch shear load vectors equal the element-wise results."""
    section = build_section(composite=composite)
    section.calculate_geometric_properties()
    coords, node_ids, elastic_modulus = element_arrays(section)
    element_batch = batch.build_batch(coords=coords, node_ids=node_ids)

    ixx, iyy, ixy = section.get_eic() if section.is_composite() else section.get_ic()
    nu = section.get_nu_eff() if section.is_composite() else 0.0

    f_psi, f_phi = batch.shear_load_vectors(element_batch, elastic_modulus, ixx, iyy, ixy, nu)

    expected = [el.shear_load_vectors(ixx=ixx, iyy=iyy, ixy=ixy, nu=nu) for el in section.elements]
    assert_close(f_psi, np.array([item[0] for item in expected]))
    assert_close(f_phi, np.array([item[1] for item in expected]))


@pytest.mark.parametrize("composite", [False, True])
def test_element_stress_matches(composite: bool) -> None:
    """Batch element stresses equal the element-wise results."""
    section = build_section(composite=composite)
    section.calculate_geometric_properties()
    section.calculate_warping_properties()

    coords, node_ids, elastic_modulus = element_arrays(section)
    cx, cy = section.get_c()
    shifted_coords = coords.copy()
    shifted_coords[:, 0, :] -= cx
    shifted_coords[:, 1, :] -= cy
    shifted = batch.build_batch(coords=shifted_coords, node_ids=node_ids, n_gauss=6)

    if section.is_composite():
        ea = section.get_ea()
        ixx, iyy, ixy = section.get_eic()
        i11, i22 = section.get_eip()
        j = section.get_ej()
        nu = section.get_nu_eff()
    else:
        ea = section.get_area()
        ixx, iyy, ixy = section.get_ic()
        i11, i22 = section.get_ip()
        j = section.get_j()
        nu = 0.0

    omega = section.section_props.omega
    psi_shear = section.section_props.psi_shear
    phi_shear = section.section_props.phi_shear
    delta_s = section.section_props.delta_s
    assert omega is not None
    assert psi_shear is not None
    assert phi_shear is not None
    assert delta_s is not None

    actions = {
        "n": 1e3,
        "mxx": 2e6,
        "myy": 3e6,
        "m11": 4e6,
        "m22": 5e6,
        "mzz": 6e6,
        "vx": 7e3,
        "vy": 8e3,
    }
    properties = {
        "ixx": ixx,
        "iyy": iyy,
        "ixy": ixy,
        "i11": i11,
        "i22": i22,
        "phi": section.get_phi(),
        "j": j,
        "nu": nu,
        "delta_s": delta_s,
    }

    stresses = batch.element_stress(
        batch=shifted,
        elastic_modulus=elastic_modulus,
        ea=ea,
        omega=omega,
        psi_shear=psi_shear,
        phi_shear=phi_shear,
        **actions,
        **properties,
    )

    expected = [
        el.element_stress(
            ea=ea,
            cx=cx,
            cy=cy,
            omega=omega[el.node_ids],
            psi_shear=psi_shear[el.node_ids],
            phi_shear=phi_shear[el.node_ids],
            **actions,
            **properties,
        )
        for el in section.elements
    ]

    components = (
        "sig_zz_n",
        "sig_zz_mxx",
        "sig_zz_myy",
        "sig_zz_m11",
        "sig_zz_m22",
        "sig_zx_mzz",
        "sig_zy_mzz",
        "sig_zx_vx",
        "sig_zy_vx",
        "sig_zx_vy",
        "sig_zy_vy",
    )

    for index, name in enumerate(components):
        assert_close(getattr(stresses, name), np.array([item[index] for item in expected]))

    assert np.allclose(stresses.weights, expected[0][11], rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("composite", [False, True])
def test_warping_integrals_and_coefficients_match(composite: bool) -> None:
    """Batch warping integrals, shear coefficients and monosymmetry integrals match."""
    section = build_section(composite=composite)
    section.calculate_geometric_properties()
    section.calculate_warping_properties()

    coords, node_ids, elastic_modulus = element_arrays(section)
    cx, cy = section.get_c()
    shifted_coords = coords.copy()
    shifted_coords[:, 0, :] -= cx
    shifted_coords[:, 1, :] -= cy
    shifted = batch.build_batch(coords=shifted_coords, node_ids=node_ids)

    ixx, iyy, ixy = section.get_eic() if section.is_composite() else section.get_ic()
    nu = section.get_nu_eff() if section.is_composite() else 0.0
    omega = section.section_props.omega
    psi_shear = section.section_props.psi_shear
    phi_shear = section.section_props.phi_shear
    phi = section.get_phi()
    assert omega is not None
    assert psi_shear is not None
    assert phi_shear is not None

    integrals = []
    kappas = []
    monos = []

    for idx, el in enumerate(section.elements):
        shifted_el = Tri6(
            el_id=el.el_id,
            coords=shifted_coords[idx],
            node_ids=el.node_ids,
            material=el.material,
        )
        integrals.append(
            shifted_el.shear_warping_integrals(ixx=ixx, iyy=iyy, ixy=ixy, omega=omega[el.node_ids])
        )
        kappas.append(
            shifted_el.shear_coefficients(
                ixx=ixx,
                iyy=iyy,
                ixy=ixy,
                psi_shear=psi_shear[el.node_ids],
                phi_shear=phi_shear[el.node_ids],
                nu=nu,
            )
        )
        monos.append(shifted_el.monosymmetry_integrals(phi=phi))

    assert_sum_close(
        np.array(batch.shear_warping_integrals(shifted, elastic_modulus, ixx, iyy, ixy, omega)),
        np.array(integrals),
    )
    assert_sum_close(
        np.array(
            batch.shear_coefficients(
                shifted, elastic_modulus, ixx, iyy, ixy, psi_shear, phi_shear, nu
            )
        ),
        np.array(kappas),
    )
    assert_sum_close(
        np.array(batch.monosymmetry_integrals(shifted, elastic_modulus, phi)),
        np.array(monos),
    )
