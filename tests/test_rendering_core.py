from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalar_field_analysis.rendering import (
    Surface3DConfig,
    SurfacePayload,
    build_marker_specs,
    build_mesh_data,
    camera_config_from_preset,
    launch_3d_viewer,
    prepare_z_for_display,
)
from scalar_field_analysis.rendering import cli


def sample_payload(**overrides) -> SurfacePayload:
    x_values = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    y_values = np.linspace(-2.0, 2.0, 3, dtype=np.float32)
    x, y = np.meshgrid(x_values, y_values)
    z = x * x + y
    values = {
        "x": x,
        "y": y,
        "z": z,
        "critical_points": np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
    }
    values.update(overrides)
    return SurfacePayload(**values)


class RenderingCoreTests(unittest.TestCase):
    def test_payload_rejects_shape_mismatch(self) -> None:
        payload = sample_payload(z=np.ones((2, 2), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            payload.validate()

    def test_payload_rejects_invalid_critical_point_shape(self) -> None:
        payload = sample_payload(critical_points=np.array([1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "critical_points"):
            payload.validate()

    def test_payload_roundtrips_through_npz(self) -> None:
        payload = sample_payload(title="Roundtrip", colormap_name="blue_orange")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = payload.to_npz(Path(temp_dir) / "payload.npz")
            loaded = SurfacePayload.from_npz(path)

        self.assertEqual(loaded.title, "Roundtrip")
        self.assertEqual(loaded.colormap_name, "blue_orange")
        np.testing.assert_allclose(loaded.x, payload.x)
        np.testing.assert_allclose(loaded.y, payload.y)
        np.testing.assert_allclose(loaded.z, payload.z)
        np.testing.assert_allclose(loaded.critical_points, payload.critical_points)

    def test_log_scale_clamps_negative_values(self) -> None:
        z = np.array([-3.0, 0.0, 99.0], dtype=np.float32)

        actual = prepare_z_for_display(z, "log10")

        np.testing.assert_allclose(actual, np.array([0.0, 0.0, 2.0]))

    def test_mesh_shape_and_face_order_are_deterministic(self) -> None:
        mesh = build_mesh_data(sample_payload(colormap_name="blue_orange"))

        self.assertEqual(mesh.vertices.shape, (12, 3))
        self.assertEqual(mesh.faces.shape, (12, 3))
        self.assertEqual(mesh.vertex_colors.shape, (12, 4))
        np.testing.assert_array_equal(mesh.faces[0], np.array([0, 1, 4]))
        np.testing.assert_array_equal(mesh.faces[1], np.array([1, 5, 4]))

    def test_marker_specs_include_only_domain_points(self) -> None:
        mesh = build_mesh_data(sample_payload(colormap_name="blue_orange"))
        markers = build_marker_specs(sample_payload(), mesh.stats)

        labels = [marker.label for marker in markers]
        self.assertIn("domain center", labels)
        self.assertIn("origin", labels)
        self.assertIn("critical 1", labels)
        self.assertNotIn("critical 2", labels)

    def test_camera_presets(self) -> None:
        camera = camera_config_from_preset("side", 10.0)

        self.assertEqual(camera.azimuth, 90.0)
        self.assertEqual(camera.elevation, 0.0)
        self.assertEqual(camera.distance, 10.0)

    def test_launch_uses_inherited_output_by_default(self) -> None:
        class FakePopen:
            pid = 123
            returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self, timeout=None) -> int:
                return 0

        popen = Mock(return_value=FakePopen())
        config = Surface3DConfig(backend="pyqt6", window_size=(640, 480))

        with patch("scalar_field_analysis.rendering.viewer.subprocess.Popen", popen):
            viewer = launch_3d_viewer(sample_payload(), config=config)
            viewer.cleanup()

        cmd = popen.call_args.args[0]
        kwargs = popen.call_args.kwargs
        self.assertIn("scalar_field_analysis.rendering.cli", cmd)
        self.assertIn("--backend", cmd)
        self.assertIn("pyqt6", cmd)
        self.assertIsNone(kwargs["stdout"])
        self.assertIsNone(kwargs["stderr"])


class CliTests(unittest.TestCase):
    def test_cli_reports_missing_payload(self) -> None:
        self.assertEqual(cli.main(["missing-payload.npz"]), 1)

    def test_cli_reports_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payload.npz"
            path.write_text("not an npz", encoding="utf-8")

            self.assertEqual(cli.main([str(path)]), 1)

    def test_cli_loads_payload_and_invokes_renderer(self) -> None:
        payload = sample_payload()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = payload.to_npz(Path(temp_dir) / "payload.npz")
            renderer = Mock()
            renderer.return_value.run.return_value = None

            with patch(
                "scalar_field_analysis.rendering.cli.VispySurfaceRenderer",
                renderer,
            ):
                returncode = cli.main([str(path), "--no-hud", "--no-markers"])

        self.assertEqual(returncode, 0)
        renderer.return_value.run.assert_called_once_with()


@unittest.skipUnless(
    os.environ.get("RUN_VISPY_GPU_SMOKE") == "1",
    "Set RUN_VISPY_GPU_SMOKE=1 to run the optional VisPy/OpenGL smoke test.",
)
class VispyGpuSmokeTests(unittest.TestCase):
    def test_hidden_scene_canvas_can_be_constructed(self) -> None:
        from vispy import app, scene

        app.use_app("pyqt6")
        canvas = scene.SceneCanvas(show=False, size=(320, 240))
        self.assertEqual(tuple(canvas.size), (320, 240))
        canvas.close()


if __name__ == "__main__":
    unittest.main()
