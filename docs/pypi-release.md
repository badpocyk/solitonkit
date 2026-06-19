# Publishing To PyPI

The repository builds binary wheels with `cibuildwheel` and publishes them with
PyPI Trusted Publishing. No long-lived `PYPI_TOKEN` GitHub secret is required.

## One-Time Setup

1. Choose and add a public `LICENSE` before announcing the package as open
   source. The repository intentionally does not invent a license on the
   maintainer's behalf.
2. Create a GitHub environment named `pypi` under **Settings > Environments**.
   Requiring manual approval for this environment is recommended.
3. Configure a PyPI Trusted Publisher:

   - PyPI project name: `solitonkit`
   - GitHub owner: `badpocyk`
   - GitHub repository: `solitonkit`
   - Workflow filename: `publish-pypi.yml`
   - Environment name: `pypi`

For the first release, PyPI supports creating a pending publisher before the
project exists. For later releases, configure the publisher from the existing
project's publishing settings.

## Dry Run

Open **Actions > Publish Python Package > Run workflow**.

The manual workflow builds and verifies:

- a source distribution,
- CPython 3.9-3.14 wheels,
- Linux x86_64 wheels,
- Windows x86_64 wheels,
- macOS Intel and Apple Silicon wheels.

Manual runs never execute the PyPI publish job.

## Release

1. Update `project.version` in `pyproject.toml`.
2. Update the fallback `__version__` in `python/solitonkit/__init__.py` if the
   source-tree fallback should change.
3. Commit and merge the version change.
4. Create a tag matching the version, conventionally `v0.1.0`.
5. Create and publish a GitHub Release from that tag.

The workflow verifies that the release tag matches `project.version`, builds
all distributions, checks their metadata, and publishes through OpenID Connect.

PyPI versions are immutable. If a release fails after any files were uploaded,
increment the package version instead of trying to overwrite the same release.

## Local Packaging Check

```powershell
python -m pip install -U build twine
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
python -c "import solitonkit as sk; print(sk.__version__)"
```
