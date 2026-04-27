from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scalar_field_analysis.rendering import (
    Surface3DConfig,
    launch_3d_viewer,
    payload_from_analysis_result,
)

if TYPE_CHECKING:
    from field_analysis import AnalysisResult


def launch_viewer_subprocess(
    result: AnalysisResult,
    *,
    backend: str = "pyqt6",
    title: str = "Scalar Field Surface",
    scale_mode: str = "linear",
    z_scale: float = 1.0,
    camera_preset: str = "isometric",
    ground_plane_mode: str = "zero",
    include_critical_points: bool = True,
):
    """Convert an AnalysisResult to a payload and launch the VisPy viewer."""

    critical_points = None
    if include_critical_points and hasattr(result, "critical_points"):
        critical_points = np.asarray(result.critical_points, dtype=np.float32)

    config = Surface3DConfig(
        backend=backend,
        title=title,
        scale_mode=scale_mode,
        z_scale=z_scale,
        camera_preset=camera_preset,
        ground_plane_mode=ground_plane_mode,
    )
    payload = payload_from_analysis_result(
        result,
        config=config,
        critical_points=critical_points,
        include_critical_points=include_critical_points,
    )
    return launch_3d_viewer(payload, config=config)
