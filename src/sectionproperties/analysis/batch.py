"""Vectorised finite element kernels.

The :class:`~sectionproperties.analysis.fea.Tri6` methods evaluate one element at
a time, which means a mesh with tens of thousands of elements pays the Python
interpreter overhead tens of thousands of times.  The functions in this module
compute exactly the same quantities, but for every element of a mesh in a
handful of array operations.

All kernels operate on an :class:`ElementBatch`, which stores the shape function
data (:math:`N`, :math:`B`, the Jacobian weights and the global Gauss point
coordinates) for the whole mesh.  The maths mirrors
:class:`~sectionproperties.analysis.fea.Tri6` one-to-one; only the loop
structure differs.

Index conventions used throughout:

* ``e`` - element index, ``n_el`` in total
* ``g`` - Gauss point index, ``n_gp`` in total
* ``i``, ``j`` - element node index, 6 in total
* ``k`` - global direction index (``x``, ``y``), 2 in total
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from sectionproperties.analysis.fea import gauss_points, h_inv


@lru_cache(maxsize=None)
def isoparametric_data(
    n: int,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Gauss weights, shape functions and isoparametric derivatives of a ``Tri6``.

    These only depend on the integration rule, not on the element geometry, so
    they are computed once and cached.

    Args:
        n: Number of Gauss points (1, 3, 4 or 6)

    Returns:
        Gauss weights (``[n_gp]``), the shape functions at each Gauss point
        (``[n_gp x 6]``) and the derivatives of the shape functions with respect
        to the isoparametric coordinates (``[n_gp x 3 x 6]``)
    """
    gps = gauss_points(n=n)
    weights = np.ascontiguousarray(gps[:, 0])
    eta, xi, zeta = gps[:, 1], gps[:, 2], gps[:, 3]

    n_shape = np.column_stack(
        (
            eta * (2 * eta - 1),
            xi * (2 * xi - 1),
            zeta * (2 * zeta - 1),
            4 * eta * xi,
            4 * xi * zeta,
            4 * eta * zeta,
        )
    )

    b_iso = np.zeros((len(gps), 3, 6), dtype=float)
    b_iso[:, 0, 0] = 4 * eta - 1
    b_iso[:, 0, 3] = 4 * xi
    b_iso[:, 0, 5] = 4 * zeta
    b_iso[:, 1, 1] = 4 * xi - 1
    b_iso[:, 1, 3] = 4 * eta
    b_iso[:, 1, 4] = 4 * zeta
    b_iso[:, 2, 2] = 4 * zeta - 1
    b_iso[:, 2, 4] = 4 * xi
    b_iso[:, 2, 5] = 4 * eta

    return weights, n_shape, b_iso


@dataclass(frozen=True)
class ElementBatch:
    """Shape function data for every element of a mesh.

    Args:
        node_ids (numpy.ndarray): Global node ids of each element, of size
            ``[n_el x 6]``
        n_shape (numpy.ndarray): Shape functions at each Gauss point, of size
            ``[n_gp x 6]``
        weights (numpy.ndarray): Gauss weight multiplied by the Jacobian, of size
            ``[n_el x n_gp]``
        nx (numpy.ndarray): Global x-coordinate of each Gauss point, of size
            ``[n_el x n_gp]``
        ny (numpy.ndarray): Global y-coordinate of each Gauss point, of size
            ``[n_el x n_gp]``
        b (numpy.ndarray | None): Strain matrix of each element at each Gauss
            point, of size ``[n_el x n_gp x 2 x 6]``. ``None`` when the batch was
            built without it.
    """

    node_ids: npt.NDArray[np.int32]
    n_shape: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    nx: npt.NDArray[np.float64]
    ny: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64] | None = None

    @property
    def num_elements(self) -> int:
        """Number of elements in the batch.

        Returns:
            Number of elements
        """
        return self.weights.shape[0]

    @property
    def num_gauss_points(self) -> int:
        """Number of Gauss points per element.

        Returns:
            Number of Gauss points
        """
        return self.weights.shape[1]

    def strain_matrix_2d(self) -> npt.NDArray[np.float64]:
        """The strain matrix flattened over the Gauss point and direction axes.

        Raises:
            ValueError: If the batch was built without the strain matrix

        Returns:
            Strain matrix of size ``[n_el x (n_gp * 2) x 6]``
        """
        if self.b is None:
            msg = "This ElementBatch was built without the strain matrix."
            raise ValueError(msg)

        return self.b.reshape(self.num_elements, self.num_gauss_points * 2, 6)

    def flatten_directions(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Flattens a per-Gauss-point, per-direction array over its last two axes.

        Args:
            values: Values of size ``[n_el x n_gp x 2]``

        Returns:
            Values of size ``[n_el x (n_gp * 2)]``
        """
        return values.reshape(self.num_elements, self.num_gauss_points * 2)


def build_batch(
    coords: npt.NDArray[np.float64],
    node_ids: npt.NDArray[np.int32],
    *,
    n_gauss: int = 4,
    strain_matrix: bool = True,
) -> ElementBatch:
    """Evaluates the ``Tri6`` shape functions for every element of a mesh.

    This is the vectorised equivalent of calling
    :func:`~sectionproperties.analysis.fea.shape_function` for every element and
    every Gauss point.

    Args:
        coords: Global coordinates of the element nodes, of size ``[n_el x 2 x 6]``
        node_ids: Global node ids of each element, of size ``[n_el x 6]``
        n_gauss: Number of Gauss points. Defaults to ``4``.
        strain_matrix: Whether the strain matrix ``b`` is required. Skipping it
            avoids the Jacobian inversion. Defaults to ``True``.

    Returns:
        The shape function data for the whole mesh
    """
    gauss_weights, n_shape, b_iso = isoparametric_data(n_gauss)
    n_el = coords.shape[0]
    n_gp = len(gauss_weights)

    # coords transposed to [n_el x 6 x 2] so the contractions below are plain
    # batched matrix products
    coords_t = np.ascontiguousarray(coords.transpose(0, 2, 1))

    # global Gauss point coordinates: [n_el x n_gp x 2]
    nxy = n_shape @ coords_t
    nx = np.ascontiguousarray(nxy[:, :, 0])
    ny = np.ascontiguousarray(nxy[:, :, 1])

    # Jacobian matrix of every element at every Gauss point: [n_el x n_gp x 3 x 3]
    # the first column is all ones, the remaining two columns are b_iso @ coords.T
    jac_cols = (b_iso.reshape(n_gp * 3, 6) @ coords_t).reshape(n_el, n_gp, 3, 2)

    j01 = jac_cols[:, :, 0, 0]
    j02 = jac_cols[:, :, 0, 1]
    j11 = jac_cols[:, :, 1, 0]
    j12 = jac_cols[:, :, 1, 1]
    j21 = jac_cols[:, :, 2, 0]
    j22 = jac_cols[:, :, 2, 1]

    # determinant of the Jacobian matrix, expanded along its first (unit) column
    det = (j11 * j22 - j12 * j21) - j01 * (j22 - j12) + j02 * (j21 - j11)
    weights = gauss_weights * (0.5 * det)

    if not strain_matrix:
        return ElementBatch(
            node_ids=node_ids, n_shape=n_shape, weights=weights, nx=nx, ny=ny
        )

    # only the last two rows of the inverse Jacobian are needed, since the strain
    # matrix is b = [[0, 1, 0], [0, 0, 1]] @ inv(j) @ b_iso.  Degenerate elements
    # (zero determinant) divide by infinity, which yields the zero strain matrix
    # used by the element-wise implementation.
    inv_det = 1.0 / np.where(det == 0.0, np.inf, det)

    inv_rows = np.empty((n_el, n_gp, 2, 3), dtype=float)
    inv_rows[:, :, 0, 0] = j12 - j22
    inv_rows[:, :, 0, 1] = j22 - j02
    inv_rows[:, :, 0, 2] = j02 - j12
    inv_rows[:, :, 1, 0] = j21 - j11
    inv_rows[:, :, 1, 1] = j01 - j21
    inv_rows[:, :, 1, 2] = j11 - j01
    inv_rows *= inv_det[:, :, None, None]

    b = np.empty((n_el, n_gp, 2, 6), dtype=float)
    for gp in range(n_gp):
        b[:, gp] = inv_rows[:, gp] @ b_iso[gp]

    return ElementBatch(
        node_ids=node_ids, n_shape=n_shape, weights=weights, nx=nx, ny=ny, b=b
    )


def geometric_properties(
    batch: ElementBatch,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Calculates the geometric properties of every element.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.geometric_properties`.

    Args:
        batch: Shape function data for the mesh

    Returns:
        Per-element geometric properties (``area``, ``qx``, ``qy``, ``ixx``,
        ``iyy``, ``ixy``), each of size ``[n_el]``
    """
    w = batch.weights
    nx = batch.nx
    ny = batch.ny

    return (
        w.sum(axis=1),
        (w * ny).sum(axis=1),
        (w * nx).sum(axis=1),
        (w * ny * ny).sum(axis=1),
        (w * nx * nx).sum(axis=1),
        (w * nx * ny).sum(axis=1),
    )


def torsion_properties(
    batch: ElementBatch,
    elastic_modulus: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Calculates the warping stiffness matrix and torsion load vector per element.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.torsion_properties`.

    Args:
        batch: Shape function data for the mesh
        elastic_modulus: Elastic modulus of every element, of size ``[n_el]``

    Returns:
        Element stiffness matrices (``[n_el x 6 x 6]``), element torsion load
        vectors (``[n_el x 6]``) and element constraint vectors (``[n_el x 6]``)
    """
    w = batch.weights * elastic_modulus[:, None]
    b2d = batch.strain_matrix_2d()

    b_weighted = (batch.b * w[:, :, None, None]).reshape(b2d.shape)
    k_el = np.matmul(b_weighted.transpose(0, 2, 1), b2d)

    load = np.stack((batch.ny, -batch.nx), axis=-1) * w[:, :, None]
    f_el = np.matmul(b2d.transpose(0, 2, 1), batch.flatten_directions(load)[:, :, None])

    c_el = w @ batch.n_shape

    return k_el, f_el[:, :, 0], c_el


def _shear_parameters(
    batch: ElementBatch,
    ixx: float,
    iyy: float,
    ixy: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the ``d`` and ``h`` shear parameter vectors at every Gauss point.

    Vectorised equivalent of
    :func:`~sectionproperties.analysis.fea._shear_parameter`.

    Args:
        batch: Shape function data for the mesh
        ixx: Second moment of area about the centroidal x-axis
        iyy: Second moment of area about the centroidal y-axis
        ixy: Second moment of area about the centroidal xy-axis

    Returns:
        The (``d1``, ``d2``) and (``h1``, ``h2``) parameters, each of size
        ``[n_el x n_gp x 2]``
    """
    nx = batch.nx
    ny = batch.ny

    r = nx**2 - ny**2
    q = 2 * nx * ny

    d = np.stack((ixx * r - ixy * q, ixy * r + ixx * q), axis=-1)
    h = np.stack((-ixy * r + iyy * q, -iyy * r - ixy * q), axis=-1)

    return d, h


def shear_load_vectors(
    batch: ElementBatch,
    elastic_modulus: npt.NDArray[np.float64],
    ixx: float,
    iyy: float,
    ixy: float,
    nu: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the shear load vectors of every element.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.shear_load_vectors`.

    Args:
        batch: Shape function data for the mesh
        elastic_modulus: Elastic modulus of every element, of size ``[n_el]``
        ixx: Second moment of area about the centroidal x-axis
        iyy: Second moment of area about the centroidal y-axis
        ixy: Second moment of area about the centroidal xy-axis
        nu: Effective Poisson's ratio for the cross-section

    Returns:
        Element shear load vectors ``f_psi`` and ``f_phi``, each of size
        ``[n_el x 6]``
    """
    w = batch.weights * elastic_modulus[:, None]
    b2d = batch.strain_matrix_2d()
    d, h = _shear_parameters(batch=batch, ixx=ixx, iyy=iyy, ixy=ixy)

    d_weighted = batch.flatten_directions(d * w[:, :, None])
    h_weighted = batch.flatten_directions(h * w[:, :, None])

    b_transpose = b2d.transpose(0, 2, 1)
    f_psi = nu / 2 * np.matmul(b_transpose, d_weighted[:, :, None])[:, :, 0]
    f_phi = nu / 2 * np.matmul(b_transpose, h_weighted[:, :, None])[:, :, 0]

    f_psi += 2 * (1 + nu) * ((w * (ixx * batch.nx - ixy * batch.ny)) @ batch.n_shape)
    f_phi += 2 * (1 + nu) * ((w * (iyy * batch.ny - ixy * batch.nx)) @ batch.n_shape)

    return f_psi, f_phi


def shear_warping_integrals(
    batch: ElementBatch,
    elastic_modulus: npt.NDArray[np.float64],
    ixx: float,
    iyy: float,
    ixy: float,
    omega: npt.NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """Calculates the shear centre and warping integrals over the whole mesh.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.shear_warping_integrals`.

    Args:
        batch: Shape function data for the mesh
        elastic_modulus: Elastic modulus of every element, of size ``[n_el]``
        ixx: Second moment of area about the centroidal x-axis
        iyy: Second moment of area about the centroidal y-axis
        ixy: Second moment of area about the centroidal xy-axis
        omega: Values of the warping function at every node of the mesh

    Returns:
        Shear centre integrals about the ``x`` and ``y`` axes and warping
        integrals (``sc_xint``, ``sc_yint``, ``q_omega``, ``i_omega``,
        ``i_xomega``, ``i_yomega``)
    """
    w = batch.weights * elastic_modulus[:, None]
    nx = batch.nx
    ny = batch.ny

    n_omega = omega[batch.node_ids] @ batch.n_shape.transpose()
    radius = nx**2 + ny**2

    return (
        float(np.sum(w * (iyy * nx + ixy * ny) * radius)),
        float(np.sum(w * (ixx * ny + ixy * nx) * radius)),
        float(np.sum(w * n_omega)),
        float(np.sum(w * n_omega**2)),
        float(np.sum(w * nx * n_omega)),
        float(np.sum(w * ny * n_omega)),
    )


def shear_coefficients(
    batch: ElementBatch,
    elastic_modulus: npt.NDArray[np.float64],
    ixx: float,
    iyy: float,
    ixy: float,
    psi_shear: npt.NDArray[np.float64],
    phi_shear: npt.NDArray[np.float64],
    nu: float,
) -> tuple[float, float, float]:
    """Calculates the shear deformation coefficients over the whole mesh.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.shear_coefficients`.

    Args:
        batch: Shape function data for the mesh
        elastic_modulus: Elastic modulus of every element, of size ``[n_el]``
        ixx: Second moment of area about the centroidal x-axis
        iyy: Second moment of area about the centroidal y-axis
        ixy: Second moment of area about the centroidal xy-axis
        psi_shear: Values of the psi shear function at every node of the mesh
        phi_shear: Values of the phi shear function at every node of the mesh
        nu: Effective Poisson's ratio for the cross-section

    Returns:
        Shear deformation coefficients (``kappa_x``, ``kappa_y``, ``kappa_xy``)
    """
    w = batch.weights * elastic_modulus[:, None]
    b2d = batch.strain_matrix_2d()
    d, h = _shear_parameters(batch=batch, ixx=ixx, iyy=iyy, ixy=ixy)

    shape = (batch.num_elements, batch.num_gauss_points, 2)
    b_psi = np.matmul(b2d, psi_shear[batch.node_ids][:, :, None]).reshape(shape)
    b_phi = np.matmul(b2d, phi_shear[batch.node_ids][:, :, None]).reshape(shape)

    b_psi_d = b_psi - nu / 2 * d
    b_phi_h = b_phi - nu / 2 * h

    return (
        float(np.sum(w * np.sum(b_psi_d * b_psi_d, axis=2))),
        float(np.sum(w * np.sum(b_phi_h * b_phi_h, axis=2))),
        float(np.sum(w * np.sum(b_psi_d * b_phi_h, axis=2))),
    )


@dataclass(frozen=True)
class ElementStress:
    """Nodal stresses of every element resulting from a specified loading.

    Every field holds the stress at the six nodes of each element, of size
    ``[n_el x 6]``.

    Args:
        sig_zz_n (numpy.ndarray): Normal stress from the axial force
        sig_zz_mxx (numpy.ndarray): Normal stress from bending about the xx-axis
        sig_zz_myy (numpy.ndarray): Normal stress from bending about the yy-axis
        sig_zz_m11 (numpy.ndarray): Normal stress from bending about the 11-axis
        sig_zz_m22 (numpy.ndarray): Normal stress from bending about the 22-axis
        sig_zx_mzz (numpy.ndarray): Shear stress in x from the torsion moment
        sig_zy_mzz (numpy.ndarray): Shear stress in y from the torsion moment
        sig_zx_vx (numpy.ndarray): Shear stress in x from the shear force in x
        sig_zy_vx (numpy.ndarray): Shear stress in y from the shear force in x
        sig_zx_vy (numpy.ndarray): Shear stress in x from the shear force in y
        sig_zy_vy (numpy.ndarray): Shear stress in y from the shear force in y
        weights (numpy.ndarray): Integration weights of the six Gauss points, of
            size ``[6]``
    """

    sig_zz_n: npt.NDArray[np.float64]
    sig_zz_mxx: npt.NDArray[np.float64]
    sig_zz_myy: npt.NDArray[np.float64]
    sig_zz_m11: npt.NDArray[np.float64]
    sig_zz_m22: npt.NDArray[np.float64]
    sig_zx_mzz: npt.NDArray[np.float64]
    sig_zy_mzz: npt.NDArray[np.float64]
    sig_zx_vx: npt.NDArray[np.float64]
    sig_zy_vx: npt.NDArray[np.float64]
    sig_zx_vy: npt.NDArray[np.float64]
    sig_zy_vy: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]


def element_stress(
    batch: ElementBatch,
    elastic_modulus: npt.NDArray[np.float64],
    n: float,
    mxx: float,
    myy: float,
    m11: float,
    m22: float,
    mzz: float,
    vx: float,
    vy: float,
    ea: float,
    ixx: float,
    iyy: float,
    ixy: float,
    i11: float,
    i22: float,
    phi: float,
    j: float,
    nu: float,
    omega: npt.NDArray[np.float64],
    psi_shear: npt.NDArray[np.float64],
    phi_shear: npt.NDArray[np.float64],
    delta_s: float,
) -> ElementStress:
    """Calculates the nodal stresses of every element for a specified loading.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.element_stress`. The batch must be
    built on centroidal coordinates with six Gauss points.

    Args:
        batch: Shape function data for the mesh, in centroidal coordinates
        elastic_modulus: Elastic modulus of every element, of size ``[n_el]``
        n: Axial force
        mxx: Bending moment about the centroidal xx-axis
        myy: Bending moment about the centroidal yy-axis
        m11: Bending moment about the centroidal 11-axis
        m22: Bending moment about the centroidal 22-axis
        mzz: Torsion moment about the centroidal zz-axis
        vx: Shear force acting in the x-direction
        vy: Shear force acting in the y-direction
        ea: Modulus weighted area
        ixx: Second moment of area about the centroidal x-axis
        iyy: Second moment of area about the centroidal y-axis
        ixy: Second moment of area about the centroidal xy-axis
        i11: Second moment of area about the principal 11-axis
        i22: Second moment of area about the principal 22-axis
        phi: Principal bending axis angle
        j: St. Venant torsion constant
        nu: Effective Poisson's ratio for the cross-section
        omega: Values of the warping function at every node of the mesh
        psi_shear: Values of the psi shear function at every node of the mesh
        phi_shear: Values of the phi shear function at every node of the mesh
        delta_s: Cross-section shear factor

    Returns:
        The nodal stresses of every element
    """
    n_el = batch.num_elements
    n_gp = batch.num_gauss_points
    e_mod = elastic_modulus[:, None]

    nx = batch.nx
    ny = batch.ny

    phi_rad = phi * np.pi / 180
    nx_11 = nx * np.cos(phi_rad) + ny * np.sin(phi_rad)
    ny_22 = ny * np.cos(phi_rad) - nx * np.sin(phi_rad)

    determinant = ixx * iyy - ixy**2

    # the axial stress is constant over the element, so it needs no extrapolation
    sig_zz_n = np.repeat((n * elastic_modulus / ea)[:, None], 6, axis=1)

    # normal stresses at the Gauss points
    sig_zz_mxx_gp = e_mod * (
        -(ixy * mxx) / determinant * nx + (iyy * mxx) / determinant * ny
    )
    sig_zz_myy_gp = e_mod * (
        -(ixx * myy) / determinant * nx + (ixy * myy) / determinant * ny
    )
    sig_zz_m11_gp = e_mod * m11 / i11 * ny_22
    sig_zz_m22_gp = e_mod * -m22 / i22 * nx_11

    # shear stresses at the Gauss points, which stay zero for an action that is not
    # applied, exactly as the element-wise implementation leaves them
    shape = (n_el, n_gp, 2)
    torsion = np.zeros(shape)
    shear_x = np.zeros(shape)
    shear_y = np.zeros(shape)

    if mzz != 0 or vx != 0 or vy != 0:
        b2d = batch.strain_matrix_2d()
        d, h = _shear_parameters(batch=batch, ixx=ixx, iyy=iyy, ixy=ixy)

        if mzz != 0:
            b_omega = np.matmul(b2d, omega[batch.node_ids][:, :, None]).reshape(shape)
            torsion = (
                e_mod[:, :, None] * mzz / j * (b_omega - np.stack((ny, -nx), axis=-1))
            )

        if vx != 0:
            b_psi = np.matmul(b2d, psi_shear[batch.node_ids][:, :, None]).reshape(shape)
            shear_x = e_mod[:, :, None] * vx / delta_s * (b_psi - nu / 2 * d)

        if vy != 0:
            b_phi = np.matmul(b2d, phi_shear[batch.node_ids][:, :, None]).reshape(shape)
            shear_y = e_mod[:, :, None] * vy / delta_s * (b_phi - nu / 2 * h)

    # extrapolate the Gauss point results to the element nodes
    extrapolate = h_inv.transpose()

    return ElementStress(
        sig_zz_n=sig_zz_n,
        sig_zz_mxx=sig_zz_mxx_gp @ extrapolate,
        sig_zz_myy=sig_zz_myy_gp @ extrapolate,
        sig_zz_m11=sig_zz_m11_gp @ extrapolate,
        sig_zz_m22=sig_zz_m22_gp @ extrapolate,
        sig_zx_mzz=torsion[:, :, 0] @ extrapolate,
        sig_zy_mzz=torsion[:, :, 1] @ extrapolate,
        sig_zx_vx=shear_x[:, :, 0] @ extrapolate,
        sig_zy_vx=shear_x[:, :, 1] @ extrapolate,
        sig_zx_vy=shear_y[:, :, 0] @ extrapolate,
        sig_zy_vy=shear_y[:, :, 1] @ extrapolate,
        weights=gauss_points(n=6)[:, 0],
    )


def monosymmetry_integrals(
    batch: ElementBatch,
    elastic_modulus: npt.NDArray[np.float64],
    phi: float,
) -> tuple[float, float, float, float]:
    """Calculates the monosymmetry integrals over the whole mesh.

    Vectorised equivalent of
    :meth:`~sectionproperties.analysis.fea.Tri6.monosymmetry_integrals`.

    Args:
        batch: Shape function data for the mesh
        elastic_modulus: Elastic modulus of every element, of size ``[n_el]``
        phi: Principal bending axis angle, in degrees

    Returns:
        Integrals used to evaluate the monosymmetry constants (``int_x``,
        ``int_y``, ``int_11``, ``int_22``)
    """
    w = batch.weights * elastic_modulus[:, None]
    nx = batch.nx
    ny = batch.ny

    phi_rad = phi * np.pi / 180
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    nx_11 = nx * cos_phi + ny * sin_phi
    ny_22 = ny * cos_phi - nx * sin_phi

    return (
        float(np.sum(w * (nx * nx * ny + ny * ny * ny))),
        float(np.sum(w * (ny * ny * nx + nx * nx * nx))),
        float(np.sum(w * (nx_11 * nx_11 * ny_22 + ny_22 * ny_22 * ny_22))),
        float(np.sum(w * (ny_22 * ny_22 * nx_11 + nx_11 * nx_11 * nx_11))),
    )
