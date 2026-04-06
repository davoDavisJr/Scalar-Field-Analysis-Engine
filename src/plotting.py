from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_scalar_field(X: np.ndarray,
                      Y: np.ndarray,
                      Z: np.ndarray,
                      *,
                      ax=None,
                      filled: bool = True,
                      levels: int = 200,
                      cmap: str = "viridis",
                      title: str = "Scalar field: $f(x,~y)$",
                     ) -> tuple[plt.ContourSet, plt.Axes]:
    """Plot the scalar field on the supplied axes."""

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots()

    if filled:
        artist = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        cbar = ax.figure.colorbar(artist, ax=ax)
    else:
        artist = ax.contour(X, Y, Z, levels=levels, cmap=cmap)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Independent variable: $x$")
    ax.set_ylabel("Independent variable: $y$")
    ax.set_aspect("equal")

    if created_ax:
        plt.tight_layout()

    return artist, ax


def plot_partial_derivative(X: np.ndarray,
                            Y: np.ndarray,
                            D: np.ndarray,
                            *,
                            derivative_label: str,
                            ax=None,
                            filled: bool = True,
                            levels: int = 200,
                            cmap: str = "plasma",
                            title: str = "Partial derivative field",
                           ) -> tuple[plt.ContourSet, plt.Axes]:
    """Plot a partial derivative field."""

    title = f"Partial derivative: {derivative_label}"
    return plot_scalar_field(
        X,
        Y,
        D,
        ax=ax,
        filled=filled,
        levels=levels,
        cmap=cmap,
        title=title,
    )


def plot_zero_contours(X: np.ndarray,
                       Y: np.ndarray,
                       DFDX: np.ndarray,
                       DFDY: np.ndarray,*,
                       ax=None,
                       level: float = 0.0,
                       colors: tuple[str, str] = ("red", "blue"),
                       labels: tuple[str, str] = ("$\\frac{df}{dx}~\\approx~0$",
                                                  "$\\frac{df}{dy}~\\approx~0$"),
                       title: str = "Zero contours of the gradient components",
                      ):
    """Plot the zero contours of df/dx and df/dy together."""

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots()
        print(type(ax))

    ax.contour(X, Y, DFDX, levels=[level], colors=colors[0])
    ax.contour(X, Y, DFDY, levels=[level], colors=colors[1])

    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label=labels[0]),
        Line2D([0], [0], color=colors[1], lw=2, label=labels[1]),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Independent variable: $x$")
    ax.set_ylabel("Independent variable: $y$")
    ax.set_aspect("equal")

    if created_ax:
        plt.tight_layout()

    return ax


def plot_critical_points(X: np.ndarray,
                         Y: np.ndarray,
                         DFDX: np.ndarray,
                         DFDY: np.ndarray,
                         critical_points: np.ndarray,
                         *,
                         ax=None,
                         level: float = 0.0,
                         colors: tuple[str, str, str] = ("red", "blue", "black"),
                         labels: tuple[str, str, str] = (
                                                         "$\\frac{df\\,}{dx\\,}~\\approx~0$",
                                                         "$\\frac{df\\,}{dy\\,}~\\approx~0$",
                                                         "Critical point candidate",
                                                        ),
                         title: str = "Approximate critical points\nfrom zero-contour intersections",
                         point_size: float = 20,
                        ) -> plt.Axes:
    """Plot zero contours together with detected critical-point candidates."""

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots()

    ax.contour(X, Y, DFDX, levels=[level], colors=colors[0])
    ax.contour(X, Y, DFDY, levels=[level], colors=colors[1])

    if critical_points.size != 0:
        ax.scatter(
            critical_points[:, 0],
            critical_points[:, 1],
            color=colors[2],
            s=point_size,
            zorder=5,
        )

    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label=labels[0]),
        Line2D([0], [0], color=colors[1], lw=2, label=labels[1]),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[2],
            linestyle="None",
            markersize=6,
            label=labels[2],
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Independent variable: $x$")
    ax.set_ylabel("Independent variable: $y$")
    ax.set_aspect("equal")

    if created_ax:
        plt.tight_layout()

    return ax


def plot_gradient_quiver(X: np.ndarray,
                         Y: np.ndarray,
                         DFDX: np.ndarray,
                         DFDY: np.ndarray,
                         *,
                         ax=None,
                         stride: int = 12,
                         title: str = "Gradient vector field",
                        ) -> plt.Axes:
    """Overlay a downsampled gradient vector field."""

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots()

    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        DFDX[::stride, ::stride],
        DFDY[::stride, ::stride],
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Independent variable: $x$")
    ax.set_ylabel("Independent variable: $y$")
    ax.set_aspect("equal")

    if created_ax:
        plt.tight_layout()

    return ax


def plot_analysis_summary(result,
                          *,
                          show_filled_field: bool = True,
                          show_gradient: bool = False,
                         ) -> tuple[plt.Figure, np.ndarray]:
    """
    Produce a compact multi-panel summary of scalar-field analysis.

    Expected result attributes:
        X, Y, Z, DFDX, DFDY, critical_points
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_scalar_field(
        result.X,
        result.Y,
        result.Z,
        ax=axes[0, 0],
        filled=show_filled_field,
        title="Scalar field: $f(x,~y)$",
    )

    plot_partial_derivative(
        result.X,
        result.Y,
        result.DFDX,
        derivative_label="$\\frac{\\!df\\,}{dx\\,}$",
        ax=axes[0, 1],
    )

    plot_partial_derivative(
        result.X,
        result.Y,
        result.DFDY,
        derivative_label="$\\frac{\\!df\\,}{dy\\,}$",
        ax=axes[1, 0],
    )

    plot_critical_points(
        result.X,
        result.Y,
        result.DFDX,
        result.DFDY,
        result.critical_points,
        ax=axes[1, 1],
    )

    if show_gradient:
        plot_gradient_quiver(
            result.X,
            result.Y,
            result.DFDX,
            result.DFDY,
            ax=axes[0, 0],
            title="Scalar field with gradient overlay",
        )

    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    print("This module provides plotting helpers for scalar-field analysis.")
    print("Import it from a notebook or driver script rather than running it directly.")
