"""Compatibility CLI for the legacy ``python -m Renderer.cli`` entrypoint."""

from scalar_field_analysis.rendering.cli import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    raise SystemExit(main())
