from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .payload import BACKENDS, Surface3DConfig, SurfacePayload
from .viewer import RendererError, VispySurfaceRenderer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone 3D scalar-field surface renderer."
    )
    parser.add_argument("payload", type=Path, help="Path to a .npz render payload.")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="pyqt6",
        help="VisPy backend to use for the native window.",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1200, 900),
        help="Native viewer window size in pixels.",
    )
    parser.add_argument("--no-hud", action="store_true", help="Hide HUD text.")
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Hide origin/domain/critical-point markers.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.payload.exists():
        print(f"Error: payload file not found: {args.payload}", file=sys.stderr)
        return 1

    try:
        payload = SurfacePayload.from_npz(args.payload)
        config = Surface3DConfig(
            backend=args.backend,
            window_size=tuple(args.window_size),
            scale_mode=payload.scale_mode,
            z_scale=payload.z_scale,
            camera_preset=payload.camera_preset,
            colormap=payload.colormap_name,
            ground_plane_mode=payload.ground_plane_mode,
            show_hud=not args.no_hud,
            show_markers=not args.no_markers,
            show_domain_center=payload.show_domain_center,
            show_origin=payload.show_origin,
            title=payload.title,
            x_label=payload.x_label,
            y_label=payload.y_label,
            z_label=payload.z_label,
        )
    except Exception as exc:
        print(f"Error: failed to load payload: {exc}", file=sys.stderr)
        return 1

    try:
        renderer = VispySurfaceRenderer(payload, config=config)
        renderer.run()
    except RendererError as exc:
        print(f"Error: renderer failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: renderer failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
