from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.spatial.distance import cdist


x, y = sp.symbols("x y", real=True)

@dataclass(frozen=True)
class GridSpec:
    """Configuration for a rectangular evaluation grid."""

    x_range: tuple[float, float] = (-2.5, 2.5)
    y_range: tuple[float, float] = (-2.5, 2.5)
    resolution: tuple[int, int] = (300, 300)

@dataclass(frozen=True)
class AnalysisResult:
    """Container for symbolic objects, grid data, and detected candidates."""

    expr: sp.Expr
    dfdx: sp.Expr
    dfdy: sp.Expr
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    DFDX: np.ndarray
    DFDY: np.ndarray
    critical_points: np.ndarray


def create_numeric_function(expr: sp.Expr,
                            variables: tuple[sp.Symbol,
                                             sp.Symbol] = (x, y)
                           ):
    """Convert a symbolic expression into a NumPy-compatible callable."""

    return sp.lambdify(variables, expr, modules="numpy")


def evaluate_on_grid(func,
                     x_range: tuple[float, float],
                     y_range: tuple[float, float],
                     resolution: tuple[int, int],
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a numeric function on a rectangular meshgrid."""

    x_values = np.linspace(*x_range, resolution[0])
    y_values = np.linspace(*y_range, resolution[1])
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.asarray(func(X, Y), dtype=float)
    return X, Y, Z


def compute_scalar_field_data(expr: sp.Expr,
                              variables: tuple[sp.Symbol,
                                               sp.Symbol] = (x, y),
                              grid: GridSpec = GridSpec(),
                             ) -> AnalysisResult:
    """Differentiate a scalar field symbolically and evaluate it numerically."""

    x_var, y_var = variables

    dfdx = sp.diff(expr, x_var)
    dfdy = sp.diff(expr, y_var)

    field_func = create_numeric_function(expr, variables)
    dfdx_func = create_numeric_function(dfdx, variables)
    dfdy_func = create_numeric_function(dfdy, variables)

    X, Y, Z = evaluate_on_grid(field_func, grid.x_range, grid.y_range, grid.resolution)
    _, _, DFDX = evaluate_on_grid(
        dfdx_func, grid.x_range, grid.y_range, grid.resolution
    )
    _, _, DFDY = evaluate_on_grid(
        dfdy_func, grid.x_range, grid.y_range, grid.resolution
    )

    critical_points = find_critical_points_from_arrays(X, Y, DFDX, DFDY)

    return AnalysisResult(
        expr=expr,
        dfdx=dfdx,
        dfdy=dfdy,
        X=X,
        Y=Y,
        Z=Z,
        DFDX=DFDX,
        DFDY=DFDY,
        critical_points=critical_points,
    )


def extract_zero_contour_points(X: np.ndarray,
                                Y: np.ndarray,
                                Z: np.ndarray,
                                *,
                                level: float = 0.0,
                               ) -> np.ndarray:
    """
    Extract sampled points from a level contour.

    This uses Matplotlib's contour machinery under the hood, but returns only the
    vertices so analysis can remain separate from plotting.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        contour_set = ax.contour(X, Y, Z, levels=[level])
        segments = contour_set.allsegs[0]
    finally:
        plt.close(fig)

    if not segments:
        return np.empty((0, 2), dtype=float)

    points = np.vstack(segments).astype(float, copy=False)
    return points


def find_intersections(points1: np.ndarray,
                       points2: np.ndarray,
                       *,
                       threshold: float = 5e-2,
                      ) -> np.ndarray:
    """Find approximate intersections between two point clouds."""

    if points1.size == 0 or points2.size == 0:
        return np.empty((0, 2), dtype=float)

    distances = cdist(points1, points2)
    i_idx, j_idx = np.where(distances < threshold)

    if i_idx.size == 0:
        return np.empty((0, 2), dtype=float)

    intersections = 0.5 * (points1[i_idx] + points2[j_idx])
    return intersections


def deduplicate_points(points: np.ndarray,
                       *,
                       tolerance: float = 7.5e-2
                      ) -> np.ndarray:
    """
    Collapse nearby points to simple centroid representatives.

    This is deliberately lightweight. It avoids turning one true candidate into a
    cloud of nearly identical hits.
    """


    if points.size == 0:
        return np.empty((0, 2), dtype=float)

    remaining = points.copy()
    clusters: list[np.ndarray] = []

    while len(remaining) > 0:
        seed = remaining[0]
        distances = np.linalg.norm(remaining - seed, axis=1)
        mask = distances <= tolerance
        cluster = remaining[mask]
        clusters.append(cluster.mean(axis=0))
        remaining = remaining[~mask]

    return np.vstack(clusters)


def find_critical_points_from_arrays(X: np.ndarray,
                                     Y: np.ndarray,
                                     DFDX: np.ndarray,
                                     DFDY: np.ndarray,
                                     *,
                                     contour_level: float = 0.0,
                                     intersection_threshold: float = 5e-2,
                                     deduplication_tolerance: float = 7.5e-2,
                                    ) -> np.ndarray:
    """Locate approximate critical points from zero-contour intersections."""

    points_dx = extract_zero_contour_points(X, Y, DFDX, level=contour_level)
    points_dy = extract_zero_contour_points(X, Y, DFDY, level=contour_level)

    candidates = find_intersections(
        points_dx,
        points_dy,
        threshold=intersection_threshold,
    )
    return deduplicate_points(candidates, tolerance=deduplication_tolerance)


def analyse_scalar_field(expr: sp.Expr,
                         *,
                         variables: tuple[sp.Symbol, sp.Symbol] = (x, y),
                         grid: GridSpec = GridSpec(),
                         intersection_threshold: float = 5e-2,
                         deduplication_tolerance: float = 7.5e-2,
                        ) -> AnalysisResult:
    """
    End-to-end symbolic and numeric analysis for a 2D scalar field.

    Notes:
        - Critical points are approximate, not exact.
        - Results depend on grid resolution and contour/intersection thresholds.
    """

    x_var, y_var = variables

    dfdx = sp.diff(expr, x_var)
    dfdy = sp.diff(expr, y_var)

    field_func = create_numeric_function(expr, variables)
    dfdx_func = create_numeric_function(dfdx, variables)
    dfdy_func = create_numeric_function(dfdy, variables)

    X, Y, Z = evaluate_on_grid(field_func, grid.x_range, grid.y_range, grid.resolution)
    _, _, DFDX = evaluate_on_grid(
        dfdx_func, grid.x_range, grid.y_range, grid.resolution
    )
    _, _, DFDY = evaluate_on_grid(
        dfdy_func, grid.x_range, grid.y_range, grid.resolution
    )

    critical_points = find_critical_points_from_arrays(
        X,
        Y,
        DFDX,
        DFDY,
        intersection_threshold=intersection_threshold,
        deduplication_tolerance=deduplication_tolerance,
    )

    return AnalysisResult(
        expr=expr,
        dfdx=dfdx,
        dfdy=dfdy,
        X=X,
        Y=Y,
        Z=Z,
        DFDX=DFDX,
        DFDY=DFDY,
        critical_points=critical_points,
    )



if __name__ == "__main__":
    test_expr = x**3 - 3 * x * y**2
    result = analyse_scalar_field(test_expr)

    print("f(x, y) =", result.expr)
    print("df/dx =", result.dfdx)
    print("df/dy =", result.dfdy)
    print("Approximate critical points:")
    print(result.critical_points)
