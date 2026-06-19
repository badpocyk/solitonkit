from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_release_version.py <release-tag>", file=sys.stderr)
        return 2

    tag = sys.argv[1]
    release_version = tag[1:] if tag.startswith("v") else tag

    with Path("pyproject.toml").open("rb") as pyproject_file:
        package_version = tomllib.load(pyproject_file)["project"]["version"]

    if release_version != package_version:
        print(
            f"release tag {tag!r} does not match package version "
            f"{package_version!r}",
            file=sys.stderr,
        )
        return 1

    print(f"release tag {tag!r} matches package version {package_version!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
