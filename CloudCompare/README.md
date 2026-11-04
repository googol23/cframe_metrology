# CloudComPy installation: Debian 12, no sudo

All paths are relative to the directory holding `install.sh`, which should be
`<repo>/CloudCompare/`. The installation does not modify system Python.

## Install

1. Download `CloudComPy_Conda312_Linux64_20260528-1022.tar.xz` from
   https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/17-production-cloudcompy-binaries/4-linux-cloudcompy-binaries/126-cloudcompy-conda312-linux64-20260526
2. Put the archive in `<repo>/CloudCompare/` alongside these scripts.
3. Run:

   ```bash
   cd <repo>/CloudCompare
   chmod +x install.sh run.sh
   ./install.sh
   ```

   If a genuine direct archive URL is available, alternatively:

   ```bash
   CLOUDCOMPY_URL='https://.../archive.tar.xz' ./install.sh
   ```

The script installs `uv` in `.tools/`, Python 3.12 in `.python/`, a virtual
environment in `.venv/`, and unpacks CloudComPy under `CloudComPy312/`.
It verifies a Python import. Network access to Astral's uv installer, Python
distributions, and PyPI is required unless already cached.

Verified on Debian 12 (x86_64, GLIBC 2.36) with the repository-local Python
3.12 environment. The packaged environment script is not compatible with Bash
`nounset`, so the wrappers disable `nounset` only while sourcing that script.

## Export

From the repository:

```bash
./CloudCompare/run.sh ./CloudCompare/export_clouds.py ./input/u3dl/data/scene.bin
```

Or choose an output directory:

```bash
./CloudCompare/run.sh ./CloudCompare/export_clouds.py ./input/u3dl/data/scene.bin --output ./input/u3dl/output/point_clouds
```

`run.sh` preserves the caller's working directory, so relative script, BIN, and
output paths resolve from the directory where the command is invoked.

`importFile` exports standalone point clouds returned by the API. Mesh-attached
clouds and specialized hierarchy nodes are not guaranteed to be included.
Existing text outputs are not overwritten by default; unique suffixes are
used. Output columns depend on the CloudComPy ASCII writer. In the tested
`examples/data/DetectorVertical.bin` fixture, each TXT row contained
space-delimited XYZ values with 8 decimal places and no header.

## Limitations

The published release is documented for x86_64 Linux and GLIBC >= 2.28,
but Debian 12 is not one of the distributions that the upstream authors report
testing. Some headless servers may lack shared libraries required by Qt or
CloudComPy; no-sudo installation cannot always remedy unavailable OS libraries.
The tested headless import emitted Qt thread-shutdown and missing fontconfig
configuration warnings, but completed successfully.
