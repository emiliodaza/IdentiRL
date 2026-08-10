"""Backward-compatible entry point. Prefer ``python -m identirl``."""

from identirl.cli import main


if __name__ == "__main__":
    main()
