"""Run with: python -m demo --config demo/config.local.json."""

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .catalog import read_config
from .files import ResultError


def main():
    parser = argparse.ArgumentParser(description="Read-only Talktopia results viewer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        read_config(args.config.expanduser().resolve())
    except ResultError as error:
        parser.error(str(error))
    uvicorn.run(create_app(args.config), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
