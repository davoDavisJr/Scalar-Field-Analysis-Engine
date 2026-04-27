"""Scalar-field analysis and GPU-backed visualization tools."""

from .analysis import (
    AnalysisResult,
    GridSpec,
    analyse_scalar_field,
    compute_scalar_field_data,
    create_numeric_function,
    deduplicate_points,
    evaluate_on_grid,
    extract_zero_contour_points,
    find_critical_points_from_arrays,
    find_intersections,
    x,
    y,
)
from .rendering import (
    RendererError,
    Surface3DConfig,
    SurfacePayload,
    ViewerProcess,
    VispySurfaceRenderer,
    launch_3d_viewer,
    payload_from_analysis_result,
    show_surface_3d,
)

__all__ = [
    "AnalysisResult",
    "GridSpec",
    "analyse_scalar_field",
    "compute_scalar_field_data",
    "create_numeric_function",
    "deduplicate_points",
    "evaluate_on_grid",
    "extract_zero_contour_points",
    "find_critical_points_from_arrays",
    "find_intersections",
    "launch_3d_viewer",
    "payload_from_analysis_result",
    "RendererError",
    "show_surface_3d",
    "Surface3DConfig",
    "SurfacePayload",
    "ViewerProcess",
    "VispySurfaceRenderer",
    "x",
    "y",
]
