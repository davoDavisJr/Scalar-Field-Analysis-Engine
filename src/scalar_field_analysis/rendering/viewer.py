from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mesh import (
    CameraConfig,
    build_hud_lines,
    build_marker_specs,
    build_mesh_data,
    camera_config_from_preset,
    compute_camera_distance,
    get_ground_plane_z,
    sample_display_z_nearest,
)
from .payload import (
    BackendName,
    Surface3DConfig,
    SurfacePayload,
    payload_from_analysis_result,
)


class RendererError(RuntimeError):
    """Raised when the VisPy renderer cannot initialize or run."""


@dataclass(slots=True)
class ViewerProcess:
    """Subprocess handle plus temporary payload cleanup."""

    process: subprocess.Popen[Any]
    payload_path: Path
    temp_dir: Path

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None, *, cleanup: bool = True) -> int:
        returncode = self.process.wait(timeout=timeout)
        if cleanup:
            self.cleanup()
        return returncode

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def cleanup(self) -> None:
        if self.process.poll() is None:
            return
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.process, name)


def _coerce_payload(
    result_or_payload: Any,
    *,
    config: Surface3DConfig | None = None,
) -> SurfacePayload:
    if isinstance(result_or_payload, SurfacePayload):
        payload = (
            result_or_payload
            if config is None
            else result_or_payload.with_config(config)
        )
    else:
        payload = payload_from_analysis_result(result_or_payload, config=config)
    payload.validate()
    return payload


def _config_from_payload(
    payload: SurfacePayload,
    config: Surface3DConfig | None,
) -> Surface3DConfig:
    if config is not None:
        config.validate()
        return config
    derived = Surface3DConfig(
        scale_mode=payload.scale_mode,
        z_scale=payload.z_scale,
        camera_preset=payload.camera_preset,
        colormap=payload.colormap_name,
        ground_plane_mode=payload.ground_plane_mode,
        show_domain_center=payload.show_domain_center,
        show_origin=payload.show_origin,
        title=payload.title,
        x_label=payload.x_label,
        y_label=payload.y_label,
        z_label=payload.z_label,
    )
    derived.validate()
    return derived


class VispySurfaceRenderer:
    """Native-window GPU renderer for scalar-field surfaces."""

    def __init__(
        self,
        payload: SurfacePayload,
        *,
        config: Surface3DConfig | None = None,
        backend: BackendName | None = None,
        camera: CameraConfig | None = None,
        bgcolor: str = "#101010",
    ) -> None:
        self.config = config or Surface3DConfig()
        if backend is not None:
            self.config = Surface3DConfig(
                backend=backend,
                window_size=self.config.window_size,
                scale_mode=self.config.scale_mode,
                z_scale=self.config.z_scale,
                camera_preset=self.config.camera_preset,
                colormap=self.config.colormap,
                ground_plane_mode=self.config.ground_plane_mode,
                show_hud=self.config.show_hud,
                show_markers=self.config.show_markers,
                show_domain_center=self.config.show_domain_center,
                show_origin=self.config.show_origin,
                title=self.config.title,
                x_label=self.config.x_label,
                y_label=self.config.y_label,
                z_label=self.config.z_label,
            )
        self.config.validate()
        self.payload = payload
        self.camera = camera
        self.bgcolor = bgcolor

    def run(self) -> None:
        try:
            from vispy import app, scene
        except ImportError as exc:
            raise RendererError(
                "VisPy is not installed. Install the render3d extra before using "
                "3D mode."
            ) from exc

        try:
            app.use_app(self.config.backend)
        except Exception as exc:
            raise RendererError(
                f"Failed to initialize requested backend {self.config.backend!r}."
            ) from exc

        try:
            from vispy.scene.cameras import TurntableCamera
            from vispy.scene.visuals import (
                GridLines,
                Markers,
                Mesh,
                Plane,
                Text,
                XYZAxis,
            )
        except ImportError as exc:
            raise RendererError("Failed to import VisPy scene visuals.") from exc

        mesh_data = build_mesh_data(self.payload)
        stats = mesh_data.stats
        ground_plane_z = get_ground_plane_z(self.payload, stats)

        canvas = scene.SceneCanvas(
            title=self.payload.title,
            size=self.config.window_size,
            show=True,
            keys="interactive",
            bgcolor=self.bgcolor,
        )
        view = canvas.central_widget.add_view()
        view.bgcolor = "#121212"

        base_distance = compute_camera_distance(stats)
        camera_cfg = self.camera or camera_config_from_preset(
            self.payload.camera_preset,
            base_distance,
        )
        view.camera = TurntableCamera(
            center=(stats.x_center, stats.y_center, stats.z_center),
            distance=camera_cfg.distance,
            azimuth=camera_cfg.azimuth,
            elevation=camera_cfg.elevation,
            fov=camera_cfg.fov,
        )
        view.camera.set_range(
            x=(stats.x_min, stats.x_max),
            y=(stats.y_min, stats.y_max),
            z=(
                min(stats.z_display_min, ground_plane_z),
                max(stats.z_display_max, ground_plane_z + 1e-6),
            ),
            margin=0.08,
        )

        grid = GridLines(parent=view.scene, color=(0.45, 0.45, 0.45, 0.45))
        grid.transform = scene.transforms.STTransform(
            translate=(0.0, 0.0, ground_plane_z)
        )

        plane = Plane(
            width=max(stats.span_x, 1e-6),
            height=max(stats.span_y, 1e-6),
            direction="+z",
            color=(0.18, 0.18, 0.18, 0.35),
            parent=view.scene,
        )
        plane.transform = scene.transforms.STTransform(
            translate=(stats.x_center, stats.y_center, ground_plane_z)
        )

        Mesh(
            vertices=mesh_data.vertices,
            faces=mesh_data.faces,
            vertex_colors=mesh_data.vertex_colors,
            shading="smooth",
            parent=view.scene,
        )

        axis = XYZAxis(parent=view.scene, width=4)
        axis_origin_x = 0.0 if stats.x_min <= 0.0 <= stats.x_max else stats.x_min
        axis_origin_y = 0.0 if stats.y_min <= 0.0 <= stats.y_max else stats.y_min
        axis.transform = scene.transforms.STTransform(
            translate=(axis_origin_x, axis_origin_y, ground_plane_z)
        )

        if self.config.show_markers:
            self._add_markers(Markers, view, mesh_data.stats)

        if self.config.show_hud:
            self._add_hud(Text, canvas, mesh_data.stats)

        print("Controls:")
        print("  - Left-click + drag to rotate")
        print("  - Right-click drag or scroll wheel to zoom")
        print("  - Shift + left-click drag to pan")
        print("  - Press 'r' to reset the camera")

        app.run()

    def _add_markers(self, markers_cls: Any, view: Any, stats: Any) -> None:
        marker_specs = build_marker_specs(self.payload, stats)
        if not marker_specs:
            return

        marker_positions = []
        marker_colors = []
        for spec in marker_specs:
            z_marker = sample_display_z_nearest(self.payload, spec.xy[0], spec.xy[1])
            marker_positions.append((spec.xy[0], spec.xy[1], z_marker))
            marker_colors.append(spec.color)

        markers = markers_cls(parent=view.scene)
        markers.set_data(
            pos=np.asarray(marker_positions, dtype=np.float32),
            face_color=np.asarray(marker_colors, dtype=np.float32),
            edge_color=np.asarray(marker_colors, dtype=np.float32),
            size=10,
        )

    def _add_hud(self, text_cls: Any, canvas: Any, stats: Any) -> None:
        hud_text = text_cls(
            "\n".join(build_hud_lines(self.payload, stats, self.config.backend)),
            color="white",
            font_size=11,
            anchor_x="left",
            anchor_y="top",
            parent=canvas.scene,
        )

        def _position_overlay(event: Any = None) -> None:
            hud_text.pos = (12, 12)

        canvas.events.resize.connect(_position_overlay)
        _position_overlay()


def show_surface_3d(
    result_or_payload: Any,
    *,
    config: Surface3DConfig | None = None,
) -> None:
    payload = _coerce_payload(result_or_payload, config=config)
    renderer = VispySurfaceRenderer(payload, config=_config_from_payload(payload, config))
    renderer.run()


def launch_3d_viewer(
    result_or_payload: Any,
    *,
    config: Surface3DConfig | None = None,
    python_executable: str | None = None,
    module_name: str = "scalar_field_analysis.rendering.cli",
    capture_output: bool = False,
) -> ViewerProcess:
    payload = _coerce_payload(result_or_payload, config=config)
    config = _config_from_payload(payload, config)
    python_executable = python_executable or sys.executable

    temp_dir = Path(tempfile.mkdtemp(prefix="scalar_field_viewer_"))
    payload_path = payload.to_npz(temp_dir / "surface_payload.npz")

    cmd = [
        python_executable,
        "-m",
        module_name,
        str(payload_path),
        "--backend",
        config.backend,
        "--window-size",
        str(config.window_size[0]),
        str(config.window_size[1]),
    ]
    if not config.show_hud:
        cmd.append("--no-hud")
    if not config.show_markers:
        cmd.append("--no-markers")

    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)
    viewer_process = ViewerProcess(
        process=process,
        payload_path=payload_path,
        temp_dir=temp_dir,
    )
    atexit.register(viewer_process.cleanup)
    return viewer_process


def launch_viewer_subprocess(
    payload: SurfacePayload,
    *,
    backend: BackendName = "pyqt6",
    python_executable: str | None = None,
    module_name: str = "scalar_field_analysis.rendering.cli",
) -> ViewerProcess:
    config = Surface3DConfig(backend=backend)
    return launch_3d_viewer(
        payload,
        config=config,
        python_executable=python_executable,
        module_name=module_name,
    )
