from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .payload import CameraPreset, SurfacePayload, prepare_z_for_display


@dataclass(frozen=True, slots=True)
class CameraConfig:
    azimuth: float = 45.0
    elevation: float = 35.0
    fov: float = 60.0
    distance: float | None = None


@dataclass(frozen=True, slots=True)
class SceneStats:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_raw_min: float
    z_raw_max: float
    z_display_min: float
    z_display_max: float
    x_center: float
    y_center: float
    z_center: float
    span_x: float
    span_y: float
    span_z: float
    grid_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray
    z_display: np.ndarray
    stats: SceneStats


@dataclass(frozen=True, slots=True)
class MarkerSpec:
    label: str
    xy: tuple[float, float]
    color: tuple[float, float, float, float]


def create_blue_orange_colormap(n: int = 256) -> np.ndarray:
    colors = np.zeros((n, 4), dtype=np.float32)
    blue = np.array([0.0, 0.4, 1.0, 1.0], dtype=np.float32)
    orange = np.array([1.0, 0.6, 0.0, 1.0], dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    colors[:] = (1.0 - t[:, None]) * blue + t[:, None] * orange
    return colors


def get_colormap_lut(name: str, n: int = 256) -> np.ndarray:
    if name == "blue_orange":
        return create_blue_orange_colormap(n)
    try:
        from matplotlib import colormaps
    except ImportError as exc:
        raise ValueError(
            "Matplotlib is required for named colormaps other than blue_orange."
        ) from exc
    try:
        cmap = colormaps[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported colormap: {name!r}") from exc
    values = cmap(np.linspace(0.0, 1.0, n, dtype=np.float32))
    return np.asarray(values, dtype=np.float32)


def normalize_for_colormap(z: np.ndarray) -> np.ndarray:
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max == z_min:
        return np.zeros_like(z, dtype=np.float32)
    return ((z - z_min) / (z_max - z_min)).astype(np.float32, copy=False)


def compute_scene_stats(payload: SurfacePayload, z_display: np.ndarray) -> SceneStats:
    x_min = float(np.min(payload.x))
    x_max = float(np.max(payload.x))
    y_min = float(np.min(payload.y))
    y_max = float(np.max(payload.y))
    z_raw_min = float(np.min(payload.z))
    z_raw_max = float(np.max(payload.z))
    z_display_min = float(np.min(z_display))
    z_display_max = float(np.max(z_display))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_display_min + z_display_max)
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_display_max - z_display_min
    return SceneStats(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_raw_min=z_raw_min,
        z_raw_max=z_raw_max,
        z_display_min=z_display_min,
        z_display_max=z_display_max,
        x_center=x_center,
        y_center=y_center,
        z_center=z_center,
        span_x=span_x,
        span_y=span_y,
        span_z=span_z,
        grid_shape=payload.x.shape,
    )


def build_mesh_data(payload: SurfacePayload) -> MeshData:
    payload.validate()

    z_display = prepare_z_for_display(payload.z, payload.scale_mode)
    z_display = (z_display * payload.z_scale).astype(np.float32, copy=False)
    z_norm = normalize_for_colormap(z_display)
    cmap = get_colormap_lut(payload.colormap_name)
    colors = cmap[(z_norm * 255).astype(np.uint8)]

    vertices = np.column_stack(
        [payload.x.ravel(), payload.y.ravel(), z_display.ravel()]
    ).astype(np.float32, copy=False)

    n_rows, n_cols = payload.x.shape
    faces = np.empty(((n_rows - 1) * (n_cols - 1) * 2, 3), dtype=np.uint32)

    k = 0
    for i in range(n_rows - 1):
        row = i * n_cols
        next_row = (i + 1) * n_cols
        for j in range(n_cols - 1):
            v0 = row + j
            v1 = row + j + 1
            v2 = next_row + j
            v3 = next_row + j + 1
            faces[k] = (v0, v1, v2)
            faces[k + 1] = (v1, v3, v2)
            k += 2

    return MeshData(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors.reshape(-1, 4).astype(np.float32, copy=False),
        z_display=z_display,
        stats=compute_scene_stats(payload, z_display),
    )


def compute_camera_distance(stats: SceneStats) -> float:
    return 1.5 * max(stats.span_x, stats.span_y, max(stats.span_z, 1e-6))


def camera_config_from_preset(
    preset: CameraPreset, base_distance: float
) -> CameraConfig:
    if preset == "isometric":
        return CameraConfig(azimuth=45.0, elevation=35.0, distance=base_distance)
    if preset == "topdown":
        return CameraConfig(azimuth=0.0, elevation=89.0, distance=base_distance)
    if preset == "front":
        return CameraConfig(azimuth=0.0, elevation=0.0, distance=base_distance)
    if preset == "side":
        return CameraConfig(azimuth=90.0, elevation=0.0, distance=base_distance)
    raise ValueError(f"Unsupported camera preset: {preset!r}")


def get_ground_plane_z(payload: SurfacePayload, stats: SceneStats) -> float:
    if payload.ground_plane_mode == "zero":
        return 0.0
    if payload.ground_plane_mode == "min_z":
        return stats.z_display_min
    raise ValueError(f"Unsupported ground_plane_mode: {payload.ground_plane_mode!r}")


def point_in_domain(x: float, y: float, stats: SceneStats) -> bool:
    return stats.x_min <= x <= stats.x_max and stats.y_min <= y <= stats.y_max


def sample_display_z_nearest(payload: SurfacePayload, x: float, y: float) -> float:
    dx = payload.x - x
    dy = payload.y - y
    idx = np.unravel_index(np.argmin(dx * dx + dy * dy), payload.x.shape)
    z_raw = float(payload.z[idx])
    z_display = float(
        prepare_z_for_display(np.array([z_raw], dtype=np.float32), payload.scale_mode)[
            0
        ]
    )
    return z_display * payload.z_scale


def build_marker_specs(payload: SurfacePayload, stats: SceneStats) -> list[MarkerSpec]:
    markers: list[MarkerSpec] = []

    if payload.show_domain_center:
        markers.append(
            MarkerSpec(
                label="domain center",
                xy=(stats.x_center, stats.y_center),
                color=(1.0, 1.0, 1.0, 1.0),
            )
        )

    if payload.show_origin and point_in_domain(0.0, 0.0, stats):
        markers.append(
            MarkerSpec(
                label="origin",
                xy=(0.0, 0.0),
                color=(0.2, 1.0, 0.2, 1.0),
            )
        )

    if payload.critical_points is not None:
        for i, point in enumerate(
            np.asarray(payload.critical_points, dtype=np.float32), start=1
        ):
            x_pt = float(point[0])
            y_pt = float(point[1])
            if point_in_domain(x_pt, y_pt, stats):
                markers.append(
                    MarkerSpec(
                        label=f"critical {i}",
                        xy=(x_pt, y_pt),
                        color=(1.0, 0.2, 0.2, 1.0),
                    )
                )

    return markers


def build_hud_lines(
    payload: SurfacePayload, stats: SceneStats, backend: str
) -> list[str]:
    return [
        payload.title,
        f"Scale mode: {payload.scale_mode}",
        f"Color: {payload.colormap_name} mapped to displayed z",
        f"Vertical scale: {payload.z_scale:.2f}x",
        f"Ground plane: {payload.ground_plane_mode}",
        f"Camera preset: {payload.camera_preset}",
        (
            f"Domain: [{stats.x_min:.3g}, {stats.x_max:.3g}] x "
            f"[{stats.y_min:.3g}, {stats.y_max:.3g}]"
        ),
        f"Z shown: [{stats.z_display_min:.3g}, {stats.z_display_max:.3g}]",
        f"Raw Z: [{stats.z_raw_min:.3g}, {stats.z_raw_max:.3g}]",
        f"Grid: {stats.grid_shape[0]} x {stats.grid_shape[1]}",
        f"Backend: {backend}",
    ]
