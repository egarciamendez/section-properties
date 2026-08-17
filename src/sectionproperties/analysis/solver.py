"""Methods used for solving linear systems and displaying info on tasks."""

from __future__ import annotations

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


def sp_solve_multi(
    k: csc_matrix,
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves a linear system for several right hand sides at once.

    The matrix is factorised once and the factorisation is reused for every
    right hand side, which is considerably cheaper than solving each system
    separately.

    Args:
        k: ``N x N`` matrix of the linear system
        b: ``N x M`` right hand sides of the linear system

    Returns:
        The ``N x M`` solution of the linear system
    """
    if _HAS_PARDISO:
        return np.asarray(sp_solve(A=k, b=b)).reshape(b.shape)

    return splu(_csc_matrix(k)).solve(b)


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


def solve_direct_lagrange_multi(
    k_lg: csc_matrix,
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves several linear systems that share a matrix (Lagrangian multiplier).

    Equivalent to calling :func:`solve_direct_lagrange` once per column of ``f``,
    but the matrix is only factorised once.

    Args:
        k_lg: ``(N+1) x (N+1)`` Lagrangian multiplier matrix of the linear system
        f: ``N x M`` right hand sides of the linear system

    Returns:
        The ``N x M`` solution of the linear systems

    Raises:
        RuntimeError: If the Lagrangian multiplier method exceeds a relative tolerance
            of ``1e-7`` or absolute tolerance related to your machine's floating point
            precision.
    """
    n_size = f.shape[0]
    rhs = np.zeros((n_size + 1, f.shape[1]))
    rhs[:n_size, :] = f

    u = sp_solve_multi(k=k_lg, b=rhs)

    # compute error for each right hand side
    multiplier = np.abs(u[-1, :])
    rel_error = multiplier / np.max(np.absolute(u), axis=0)

    invalid = (rel_error > 1e-7) & (multiplier > 10.0 * np.finfo(float).eps)

    if invalid.any():
        msg = "Lagrangian multiplier method error exceeds the prescribed tolerance, "
        msg += "consider refining your mesh. If this error is unexpected raise an "
        msg += "issue at https://github.com/robbievanleeuwen/section-properties/issues."
        raise RuntimeError(msg)

    return u[:n_size, :]


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
