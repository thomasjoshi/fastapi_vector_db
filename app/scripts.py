"""
Script functions for Poetry commands.
These functions are referenced in pyproject.toml and provide 
command-line functionality through Poetry.
"""

import subprocess
import sys
from pathlib import Path


def start():
    """Start the FastAPI application in production mode."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


def dev():
    """Start the FastAPI application in development mode with auto-reload."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


def lint():
    """Run linting checks on the codebase."""
    subprocess.run(["ruff", "check", "."], cwd=Path(__file__).parent.parent)


def format():
    """Format the codebase using black and ruff."""
    repo_root = Path(__file__).parent.parent
    print("Running black...")
    subprocess.run(["black", "app", "tests"], cwd=repo_root)

    print("Running ruff with auto-fixes...")
    subprocess.run(["ruff", "check", ".", "--fix"], cwd=repo_root)


def test():
    """Run the test suite."""
    import pytest

    sys.exit(pytest.main(["tests"]))


if __name__ == "__main__":
    # Allow direct execution
    command = sys.argv[1] if len(sys.argv) > 1 else "dev"

    if command == "start":
        start()
    elif command == "dev":
        dev()
    elif command == "test":
        test()
    elif command == "lint":
        lint()
    elif command == "format":
        format()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: start, dev, test, lint, format")
        sys.exit(1)
