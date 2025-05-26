#!/usr/bin/env python3
"""
Script to generate requirements.txt from pyproject.toml
This script extracts dependencies from pyproject.toml and writes them
to requirements.txt for use in Docker builds.
"""
import sys
from pathlib import Path

import tomlkit


def generate_requirements(pyproject_path: Path, output_path: Path) -> None:
    """Generate requirements.txt from pyproject.toml"""
    # Read pyproject.toml
    with open(pyproject_path) as f:
        pyproject = tomlkit.parse(f.read())

    # Extract dependencies
    if "tool" not in pyproject or "poetry" not in pyproject["tool"]:
        print("Error: pyproject.toml doesn't contain poetry configuration")
        sys.exit(1)

    dependencies = pyproject["tool"]["poetry"]["dependencies"]

    # Format dependencies
    formatted_deps = []
    for name, version in dependencies.items():
        # Skip python dependency
        if name == "python":
            continue

        # Handle different version formats
        if isinstance(version, str):
            formatted_deps.append(f"{name}=={version}" if version != "*" else name)
        elif isinstance(version, dict) and "version" in version:
            formatted_deps.append(f"{name}=={version['version']}")
        else:
            formatted_deps.append(name)

    # Write requirements.txt
    with open(output_path, "w") as f:
        f.write("\n".join(formatted_deps))

    print(f"Generated {output_path} with {len(formatted_deps)} dependencies")


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    output_path = root_dir / "requirements.txt"

    generate_requirements(pyproject_path, output_path)
