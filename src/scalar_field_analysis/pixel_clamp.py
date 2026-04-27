from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from .analysis import AnalysisResult


@dataclass(frozen=True)
class DisplayInfo:
    screen_width_px: Optional[int]
    screen_height_px: Optional[int]
    screen_width_mm: Optional[float]
    screen_height_mm: Optional[float]
    screen_ppi_x: Optional[float]
    screen_ppi_y: Optional[float]
    figure_width_in: float
    figure_height_in: float
    figure_dpi: float
    figure_width_px: int
    figure_height_px: int


def get_display_info() -> DisplayInfo:
    """
    Estimate practical display information relevant to plotting.

    Notes:
        - Screen mm dimensions may be unavailable on some systems.
        - In notebook / VSCode environments, actual displayed pixel usage may
          differ slightly from this estimate due to UI scaling and embedding.
    """

    fig = plt.gcf()

    figure_width_in, figure_height_in = fig.get_size_inches()
    figure_dpi = float(fig.dpi)

    figure_width_px  = int(round(figure_width_in * figure_dpi))
    figure_height_px = int(round(figure_height_in * figure_dpi))

    screen_width_px  = None
    screen_height_px = None
    screen_width_mm  = None
    screen_height_mm = None
    screen_ppi_x     = None
    screen_ppi_y     = None

    # Tkinter is a decent cross-platform fallback for screen metrics.
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        screen_width_px  = int(root.winfo_screenwidth())
        screen_height_px = int(root.winfo_screenheight())

        # These may be 0 or unreliable on some systems / backends.
        screen_width_mm_raw  = root.winfo_screenmmwidth()
        screen_height_mm_raw = root.winfo_screenmmheight()

        if screen_width_mm_raw and screen_width_mm_raw > 0:
            screen_width_mm  = float(screen_width_mm_raw)
        if screen_height_mm_raw and screen_height_mm_raw > 0:
            screen_height_mm = float(screen_height_mm_raw)

        root.destroy()
    except Exception:
        pass

    if (screen_width_px is not None
        and screen_width_mm is not None
        and screen_width_mm > 0):
        screen_ppi_x = screen_width_px / (screen_width_mm / 25.4)

    if (screen_height_px is not None
        and screen_height_mm is not None
        and screen_height_mm > 0):
        screen_ppi_y = screen_height_px / (screen_height_mm / 25.4)

    return DisplayInfo(
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        screen_width_mm=screen_width_mm,
        screen_height_mm=screen_height_mm,
        screen_ppi_x=screen_ppi_x,
        screen_ppi_y=screen_ppi_y,
        figure_width_in=figure_width_in,
        figure_height_in=figure_height_in,
        figure_dpi=figure_dpi,
        figure_width_px=figure_width_px,
        figure_height_px=figure_height_px,
    )


def graph_is_oversampled(result,
                         *,
                         target_display_pixels: tuple[int, int] = (1280, 720),
                         physical_display: DisplayInfo          = get_display_info(),
                         subplot_layout: tuple[int, int]        = (2, 2),
                         tolerance: float                       = 2.0,
                        ) -> bool:
    """
    Return True if the plotting grid is denser than the display can
    meaningfully show for the current summary layout.

    Parameters
    ----------
    result:
        Analysis result object with X/Y/Z-style grid arrays.
    target_display_pixels:
        Pixel 'budget' for the *entire* displayed figure.
        Default is 720p = (1280, 720).
    subplot_layout:
        The summary plot layout as (rows, cols).
        For `plot_analysis_summary()`, this is (2, 2). Currently.
    tolerance:
        Multiplier for how much oversampling to tolerate before calling it
        excessive. Use:
           - 1.0  -> strict
           - 1.25 -> mildly permissive
           - 1.5  -> fairly permissive
           - 2.0  ~> Nyquist limit for anti-aliasing
           - 2.5+ -> very permissive (not recommended)

    For more info on the `Nyquist-Shannon Sampling Theorem`, see:
    > H. Nyquist, "Certain Topics in Telegraph Transmission Theory,"
    Transactions of the A.I.E.E., vol. 47, no. 2, pp. 617-644, Apr. 1928

    Notes
    -----
    This checks *display oversampling*, not whether the numerical analysis
    itself benefited from the denser grid.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive (and non-zero)")

    if physical_display is None:
        physical_display = get_display_info()

    rows, cols = subplot_layout
    if rows <= 0 or cols <= 0:
        raise ValueError("subplot_layout must contain positive integers")

    if target_display_pixels is not None:
        total_width_px, total_height_px = target_display_pixels
    elif physical_display.figure_width_px > 0 and physical_display.figure_height_px > 0:
        total_width_px  = physical_display.figure_width_px
        total_height_px = physical_display.figure_height_px
    elif (
        physical_display.screen_width_px is not None
        and physical_display.screen_height_px is not None
    ):
        total_width_px  = physical_display.screen_width_px
        total_height_px = physical_display.screen_height_px
    else:
        raise ValueError(
            "Could not determine a display pixel budget. Provide target_display_pixels explicitly."
        )

    panel_width_px = total_width_px / cols
    panel_height_px = total_height_px / rows

    # Meshgrid shape is (y_points, x_points)
    y_points, x_points = result.X.shape

    max_useful_x = ceil(panel_width_px * tolerance)
    max_useful_y = ceil(panel_height_px * tolerance)

    return (x_points > max_useful_x) or (y_points > max_useful_y)


def downsample_grid(X: np.ndarray,
                    Y: np.ndarray,
                    *arrays: np.ndarray,
                    max_points: tuple[int, int],
                   ) -> tuple[np.ndarray, ...]:
    """
    Downsample a meshgrid and any number of aligned 2D arrays.

    This matches the established plotting pipeline more naturally than accepting
    a single container array. Pass `Z`, `DFDX`, `DFDY`, etc. as separate args.
    """

    y_points, x_points = X.shape
    max_x, max_y = max_points

    if max_x <= 0 or max_y <= 0:
        raise ValueError("max_points must contain positive integers")

    stride_x = max(1, ceil(x_points / max_x))
    stride_y = max(1, ceil(y_points / max_y))

    Xd = np.asarray(X[::stride_y, ::stride_x], dtype=float)
    Yd = np.asarray(Y[::stride_y, ::stride_x], dtype=float)

    downsampled = tuple(
        np.asarray(arr[::stride_y, ::stride_x], dtype=float) for arr in arrays
    )

    return Xd, Yd, *downsampled


def prepare_plot_result(result: AnalysisResult,
                        *,
                        subplot_layout: tuple[int, int] = (2, 2),
                        tolerance: float = 2.0,
                        anti_oversample: bool = True,
                        ) -> AnalysisResult:
    """
    Return a plotting-safe result object.

    The analysis result is preserved as-is unless the grid meaningfully exceeds
    the practical display budget, in which case only the display arrays are
    downsampled. Critical-point candidates are left untouched.
    """

    if not anti_oversample:
        return result

    if not graph_is_oversampled(
        result,
        physical_display=get_display_info(),
        subplot_layout=subplot_layout,
        tolerance=tolerance,
    ):
        return result

    display_info = get_display_info()
    rows, cols = subplot_layout
    panel_width_px = display_info.figure_width_px / cols
    panel_height_px = display_info.figure_height_px / rows

    max_points = (
        ceil(panel_width_px * tolerance),
        ceil(panel_height_px * tolerance),
    )

    X_plot, Y_plot, Z_plot, DFDX_plot, DFDY_plot = downsample_grid(
        result.X,
        result.Y,
        result.Z,
        result.DFDX,
        result.DFDY,
        max_points=max_points,
    )

    return replace(
        result,
        X=X_plot,
        Y=Y_plot,
        Z=Z_plot,
        DFDX=DFDX_plot,
        DFDY=DFDY_plot,
    )
