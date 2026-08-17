"""Methods used for solving linear systems and displaying info on tasks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
)
from rich.table import Column
from rich.text import Text
from scipy.sparse import csc_matrix as _csc_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cgs, splu, spsolve

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import LinearOperator


try:
    import pypardiso

    sp_solve = pypardiso.spsolve
    _HAS_PARDISO = True
except ImportError:
    sp_solve = spsolve
    _HAS_PARDISO = False


# Whether the pinned warping system is solved with PARDISO when it is installed.
# The Lagrangian multiplier matrix has a completely dense row and column, which
# PARDISO's ordering handles far better than SuperLU's. The pinned system that
# solve_warping_direct() builds does not, and SuperLU is measurably faster on it
# at every mesh size tested, so PARDISO is not used there by default. Set this to
# True to use it anyway.
USE_PARDISO_FOR_WARPING = False


def sp_solve_multi(
    k: csc_matrix,
    b: npt.NDArray[np.float64],
    pardiso: bool | None = None,
    positive_definite: bool = False,
) -> npt.NDArray[np.float64]:
    """Solves a linear system for several right hand sides at once.

    The matrix is factorised once and the factorisation is reused for every
    right hand side, which is considerably cheaper than solving each system
    separately.

    Args:
        k: ``N x N`` matrix of the linear system
        b: ``N x M`` right hand sides of the linear system
        pardiso: Whether to use PARDISO when it is installed. Defaults to using
            it whenever it is available.
        positive_definite: Set to True when the matrix is symmetric positive
            definite, which lets SuperLU order the matrix symmetrically and skip
            partial pivoting. Defaults to ``False``.

    Returns:
        The ``N x M`` solution of the linear system
    """
    use_pardiso = _HAS_PARDISO if pardiso is None else (pardiso and _HAS_PARDISO)

    if use_pardiso:
        return np.asarray(sp_solve(A=k, b=b)).reshape(b.shape)

    options = (
        {
            "permc_spec": "MMD_AT_PLUS_A",
            "diag_pivot_thresh": 0.0,
            "options": {"SymmetricMode": True},
        }
        if positive_definite
        else {}
    )

    return splu(_csc_matrix(k), **options).solve(b)


def solve_cgs(
    k: csc_matrix,
    f: npt.NDArray[np.float64],
    m: LinearOperator,
    tol: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Solves a linear system using the CGS iterative method.

    Args:
        k: ``N x N`` matrix of the linear system
        f: ``N x 1`` right hand side of the linear system
        m: Preconditioner for the linear matrix approximating the inverse of ``k``
        tol: Relative tolerance for the solver to achieve. Defaults to ``1e-5``.

    Returns:
        The solution vector to the linear system of equations

    Raises:
        RuntimeError: If the CGS iterative method does not converge
    """
    u, info = cgs(A=k, b=f, rtol=tol, M=m)

    if info != 0:
        msg = "CGS iterative method did not converge."
        raise RuntimeError(msg)

    return u


def solve_cgs_lagrange(
    k_lg: csc_matrix,
    f: npt.NDArray[np.float64],
    m: LinearOperator,
    tol: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Solves a linear system using the CGS iterative method (Lagrangian multiplier).

    Args:
        k_lg: ``(N+1) x (N+1)`` Lagrangian multiplier matrix of the linear system
        f: ``N x 1`` right hand side of the linear system
        m: Preconditioner for the linear matrix approximating the inverse of ``k``
        tol: Relative tolerance for the solver to achieve. Defaults to ``1e-5``.

    Returns:
        The solution vector to the linear system of equations

    Raises:
        RuntimeError: If the CGS iterative method does not converge or the error from
            the Lagrangian multiplier method exceeds the tolerance
    """
    u, info = cgs(A=k_lg, b=np.append(f, 0), rtol=tol, M=m)

    if info != 0:
        msg = "CGS iterative method did not converge."
        raise RuntimeError(msg)

    # compute error
    err = u[-1] / max(np.absolute(u))

    if err > tol:
        msg = "Lagrangian multiplier method error exceeds tolerance."
        raise RuntimeError(msg)

    return u[:-1]


def solve_direct(
    k: csc_matrix,
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves a linear system using the direct solver method.

    Args:
        k: ``N x N`` matrix of the linear system
        f: ``N x 1`` right hand side of the linear system

    Returns:
        The solution vector to the linear system of equations
    """
    return sp_solve(A=k, b=f)


def solve_direct_lagrange(
    k_lg: csc_matrix,
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves a linear system using the direct solver method (Lagrangian multiplier).

    Args:
        k_lg: ``(N+1) x (N+1)`` Lagrangian multiplier matrix of the linear system
        f: ``N x 1`` right hand side of the linear system

    Returns:
        The solution vector to the linear system of equations

    Raises:
        RuntimeError: If the Lagrangian multiplier method exceeds a relative tolerance
            of ``1e-7`` or absolute tolerance related to your machine's floating point
            precision.
    """
    u = sp_solve(A=k_lg, b=np.append(f, 0))

    # compute error
    multiplier = abs(u[-1])
    rel_error = multiplier / max(np.absolute(u))

    if rel_error > 1e-7 and multiplier > 10.0 * np.finfo(float).eps:
        msg = "Lagrangian multiplier method error exceeds the prescribed tolerance, "
        msg += "consider refining your mesh. If this error is unexpected raise an "
        msg += "issue at https://github.com/robbievanleeuwen/section-properties/issues."
        raise RuntimeError(msg)

    return u[:-1]


def _pin_dofs(
    k: csc_matrix,
    dofs: npt.NDArray[np.int_],
) -> csc_matrix:
    """Replaces the equations of some degrees of freedom with ``u[dof] = 0``.

    Their rows and columns are zeroed and their diagonal entries are set to one.
    Working on the stored values keeps the sparsity structure of the matrix,
    which is much cheaper than slicing the rows and columns away.

    Args:
        k: ``N x N`` matrix in CSC format
        dofs: Indices of the degrees of freedom to fix

    Returns:
        The ``N x N`` matrix with the equations of ``dofs`` replaced

    Raises:
        RuntimeError: If the matrix has no entry on the diagonal of a fixed
            degree of freedom
    """
    if not k.has_sorted_indices:
        k.sort_indices()

    pinned = _csc_matrix(
        (k.data.copy(), k.indices, k.indptr), shape=k.shape, copy=False
    )

    # the rows are every entry whose row index is one of the fixed dofs
    pinned.data[np.isin(pinned.indices, dofs)] = 0.0

    for dof in dofs:
        # the column of a dof is one contiguous block of the data array
        start = int(pinned.indptr[dof])
        stop = int(pinned.indptr[dof + 1])
        offset = int(np.searchsorted(pinned.indices[start:stop], dof))

        if offset >= stop - start or pinned.indices[start + offset] != dof:
            msg = "The stiffness matrix has no entry on the diagonal of the degree of "
            msg += "freedom being fixed, which means the mesh contains an unused node."
            raise RuntimeError(msg)

        pinned.data[start:stop] = 0.0
        pinned.data[start + offset] = 1.0

    return pinned


def solve_warping_direct(
    k: csc_matrix,
    c: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Solves the warping system :math:`Ku = f` subject to :math:`c \cdot u = 0`.

    The stiffness matrix ``k`` is singular: adding a constant to a warping or
    shear function leaves the equations unchanged. :func:`solve_direct_lagrange`
    removes that freedom with a Lagrangian multiplier, which appends a fully
    dense row and column to an otherwise very sparse matrix. Eliminating that
    row costs an enormous amount of fill-in during the factorisation.

    This routine instead fixes a single degree of freedom, solves the resulting
    positive definite system, and then shifts the solution by the constant that
    satisfies the constraint. The result is the same solution, obtained from a
    factorisation that is roughly an order of magnitude cheaper.

    Args:
        k: ``N x N`` stiffness matrix, singular up to a constant
        c: ``N`` constraint vector, the modulus weighted nodal areas
        f: ``N x M`` right hand sides of the linear system

    Returns:
        The ``N x M`` solution of the constrained linear systems
    """
    k = k.tocsc()
    diagonal = k.diagonal()

    # the null space of k is the set of functions that are constant on each
    # connected region of the mesh, so one degree of freedom has to be fixed per
    # region. A warping analysis of a mesh with more than one region is not
    # meaningful, but the analysis is allowed to run and warn rather than fail.
    n_regions, region = connected_components(csgraph=k, directed=False)

    if n_regions == 1:
        pinned = np.array([int(np.argmax(diagonal))])
    else:
        msg = f"\nThe finite element mesh consists of {n_regions} disconnected "
        msg += "regions, which is invalid for a warping analysis. Please revise your "
        msg += "geometry to ensure there is connectivity between all regions. Please "
        msg += "see https://sectionproperties.rtfd.io/en/stable/user_guide/"
        msg += "analysis.html#warping-analysis for more information."
        warnings.warn(msg, stacklevel=2)

        # fix the best conditioned degree of freedom of every region
        pinned = np.array(
            [
                int(np.flatnonzero(region == idx)[np.argmax(diagonal[region == idx])])
                for idx in range(n_regions)
            ]
        )

    # replace the equations of the fixed degrees of freedom with u[pinned] = 0,
    # which keeps the sparsity structure of the matrix intact
    k_pinned = _pin_dofs(k=k, dofs=pinned)

    rhs = f.copy()
    rhs[pinned, :] = 0.0

    u = sp_solve_multi(
        k=k_pinned,
        b=rhs,
        pardiso=USE_PARDISO_FOR_WARPING,
        positive_definite=True,
    )

    # restore the constraint by shifting each solution by a constant
    u -= (c @ u) / c.sum()

    return u


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed in milliseconds."""

    def render(
        self,
        task: Task,
    ) -> Text:
        """Show time remaining.

        Args:
            task: Rich progress task

        Returns:
            Rich text object
        """
        elapsed = task.finished_time if task.finished else task.elapsed

        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")

        elapsed_string = f"[ {elapsed:.4f} s ]"

        return Text(elapsed_string, style="progress.elapsed")


def create_progress() -> Progress:
    """Returns a Rich Progress class.

    Returns:
        Rich Progress class containing a spinner, progress description, percentage and
        time
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=1)
        ),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        CustomTimeElapsedColumn(),
        expand=True,
    )
