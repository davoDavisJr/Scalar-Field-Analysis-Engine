from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

BackendName = Literal["pyqt6", "pyqt5", "pyside6", "glfw", "sdl2"]
ScaleMode = Literal["linear", "log10"]
CameraPreset = Literal["isometric", "topdown", "front", "side"]
GroundPlaneMode = Literal["zero", "min_z"]

BACKENDS: tuple[BackendName, ...] = ("pyqt6", "pyqt5", "pyside6", "glfw", "sdl2")
SCALE_MODES: tuple[ScaleMode, ...] = ("linear", "log10")
CAMERA_PRESETS: tuple[CameraPreset, ...] = (
    "isometric",
    "topdown",
    "front",
    "side",
)
GROUND_PLANE_MODES: tuple[GroundPlaneMode, ...] = ("zero", "min_z")


@dataclass(frozen=True, slots=True)
class Surface3DConfig:
    """Runtime and display options for the 3D renderer."""

    backend: BackendName = "pyqt6"
    window_size: tuple[int, int] = (1200, 900)
    scale_mode: ScaleMode = "linear"
    z_scale: float = 1.0
    camera_preset: CameraPreset = "isometric"
    colormap: str = "viridis"
    ground_plane_mode: GroundPlaneMode = "zero"
    show_hud: bool = True
    show_markers: bool = True
    show_domain_center: bool = True
    show_origin: bool = True
    title: str = "Scalar Field Surface"
    x_label: str = "x"
    y_label: str = "y"
    z_label: str = "z"

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend!r}")
        if self.scale_mode not in SCALE_MODES:
            raise ValueError(f"Unsupported scale mode: {self.scale_mode!r}")
        if self.camera_preset not in CAMERA_PRESETS:
            raise ValueError(f"Unsupported camera preset: {self.camera_preset!r}")
        if self.ground_plane_mode not in GROUND_PLANE_MODES:
            raise ValueError(
                f"Unsupported ground_plane_mode: {self.ground_plane_mode!r}"
            )
        width, height = self.window_size
        if width <= 0 or height <= 0:
            raise ValueError("window_size must contain positive integers.")
        if not np.isfinite(self.z_scale) or self.z_scale <= 0.0:
            raise ValueError("z_scale must be a positive finite number.")


@dataclass(frozen=True, slots=True)
class SurfacePayload:
    """Serializable scene data for a scalar-field surface."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    title: str = "Scalar Field Surface"
    scale_mode: ScaleMode = "linear"
    x_label: str = "x"
    y_label: str = "y"
    z_label: str = "z"
    colormap_name: str = "viridis"
    z_scale: float = 1.0
    camera_preset: CameraPreset = "isometric"
    ground_plane_mode: GroundPlaneMode = "zero"
    critical_points: np.ndarray | None = None
    show_domain_center: bool = True
    show_origin: bool = True

    def validate(self) -> None:
        if self.x.ndim != 2 or self.y.ndim != 2 or self.z.ndim != 2:
            raise ValueError("x, y, and z must all be 2D arrays.")
        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError("x, y, and z must have identical shapes.")
        if min(self.x.shape) < 2:
            raise ValueError("surface grid must be at least 2x2.")
        if not np.isfinite(self.x).all():
            raise ValueError("x contains non-finite values.")
        if not np.isfinite(self.y).all():
            raise ValueError("y contains non-finite values.")
        if not np.isfinite(self.z).all():
            raise ValueError("z contains non-finite values.")
        if self.scale_mode not in SCALE_MODES:
            raise ValueError(f"Unsupported scale mode: {self.scale_mode!r}")
        if self.camera_preset not in CAMERA_PRESETS:
            raise ValueError(f"Unsupported camera preset: {self.camera_preset!r}")
        if self.ground_plane_mode not in GROUND_PLANE_MODES:
            raise ValueError(
                f"Unsupported ground_plane_mode: {self.ground_plane_mode!r}"
            )
        if not np.isfinite(self.z_scale) or self.z_scale <= 0.0:
            raise ValueError("z_scale must be a positive finite number.")
        if self.critical_points is not None:
            critical_points = np.asarray(self.critical_points, dtype=np.float32)
            if critical_points.ndim != 2 or critical_points.shape[1] != 2:
                raise ValueError("critical_points must have shape (n, 2).")
            if not np.isfinite(critical_points).all():
                raise ValueError("critical_points contains non-finite values.")

    def with_config(self, config: Surface3DConfig) -> SurfacePayload:
        config.validate()
        return replace(
            self,
            title=config.title,
            scale_mode=config.scale_mode,
            x_label=config.x_label,
            y_label=config.y_label,
            z_label=config.z_label,
            colormap_name=config.colormap,
            z_scale=config.z_scale,
            camera_preset=config.camera_preset,
            ground_plane_mode=config.ground_plane_mode,
            show_domain_center=config.show_domain_center,
            show_origin=config.show_origin,
        )

    def to_npz(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        metadata = {
            "title": self.title,
            "scale_mode": self.scale_mode,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "z_label": self.z_label,
            "colormap_name": self.colormap_name,
            "z_scale": self.z_scale,
            "camera_preset": self.camera_preset,
            "ground_plane_mode": self.ground_plane_mode,
            "show_domain_center": self.show_domain_center,
            "show_origin": self.show_origin,
        }
        arrays: dict[str, Any] = {
            "x": np.asarray(self.x, dtype=np.float32),
            "y": np.asarray(self.y, dtype=np.float32),
            "z": np.asarray(self.z, dtype=np.float32),
            "metadata": np.array(json.dumps(metadata)),
        }
        if self.critical_points is not None:
            arrays["critical_points"] = np.asarray(
                self.critical_points, dtype=np.float32
            )
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> SurfacePayload:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            critical_points = None
            if "critical_points" in data:
                critical_points = np.asarray(
                    data["critical_points"], dtype=np.float32
                )
            payload = cls(
                x=np.asarray(data["x"], dtype=np.float32),
                y=np.asarray(data["y"], dtype=np.float32),
                z=np.asarray(data["z"], dtype=np.float32),
                title=metadata.get("title", "Scalar Field Surface"),
                scale_mode=metadata.get("scale_mode", "linear"),
                x_label=metadata.get("x_label", "x"),
                y_label=metadata.get("y_label", "y"),
                z_label=metadata.get("z_label", "z"),
                colormap_name=metadata.get("colormap_name", "viridis"),
                z_scale=float(metadata.get("z_scale", 1.0)),
                camera_preset=metadata.get("camera_preset", "isometric"),
                ground_plane_mode=metadata.get("ground_plane_mode", "zero"),
                critical_points=critical_points,
                show_domain_center=bool(metadata.get("show_domain_center", True)),
                show_origin=bool(metadata.get("show_origin", True)),
            )
        payload.validate()
        return payload


def prepare_z_for_display(z: np.ndarray, scale_mode: ScaleMode) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if scale_mode == "linear":
        return z
    if scale_mode == "log10":
        return np.log10(np.maximum(z, 0.0) + 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported scale mode: {scale_mode!r}")


def payload_from_analysis_result(
    result: Any,
    *,
    config: Surface3DConfig | None = None,
    title: str | None = None,
    scale_mode: ScaleMode | None = None,
    z_scale: float | None = None,
    camera_preset: CameraPreset | None = None,
    ground_plane_mode: GroundPlaneMode | None = None,
    colormap: str | None = None,
    critical_points: np.ndarray | None = None,
    include_critical_points: bool = True,
) -> SurfacePayload:
    """Convert an analysis result object into a render payload."""

    config = config or Surface3DConfig()
    config.validate()

    if critical_points is None and include_critical_points:
        if hasattr(result, "critical_points"):
            critical_points = np.asarray(result.critical_points, dtype=np.float32)

    payload = SurfacePayload(
        x=np.asarray(result.X, dtype=np.float32),
        y=np.asarray(result.Y, dtype=np.float32),
        z=np.asarray(result.Z, dtype=np.float32),
        title=title if title is not None else config.title,
        scale_mode=scale_mode if scale_mode is not None else config.scale_mode,
        x_label=config.x_label,
        y_label=config.y_label,
        z_label=config.z_label,
        colormap_name=colormap if colormap is not None else config.colormap,
        z_scale=z_scale if z_scale is not None else config.z_scale,
        camera_preset=(
            camera_preset if camera_preset is not None else config.camera_preset
        ),
        ground_plane_mode=(
            ground_plane_mode
            if ground_plane_mode is not None
            else config.ground_plane_mode
        ),
        critical_points=critical_points,
        show_domain_center=config.show_domain_center,
        show_origin=config.show_origin,
    )
    payload.validate()
    return payload
