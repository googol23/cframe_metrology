# CloudComPy BIN exporter

Self-contained, no-root CloudComPy installation for Debian 12. It exports each
standalone point cloud returned from a CloudCompare `.bin` project to a separate
`.txt` file named after the cloud entity.

Everything is installed and run inside this directory. System Python is not
modified.

## Requirements

- Debian 12 on x86_64 Linux (GLIBC 2.28 or newer)
- `bash`, `curl`, and `tar`
- Network access for the initial Python dependency installation
- The official CloudComPy archive:
  `CloudComPy_Conda312_Linux64_20260528-1022.tar.xz`

Download the archive from the
[official CloudComPy release page](https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/17-production-cloudcompy-binaries/4-linux-cloudcompy-binaries/126-cloudcompy-conda312-linux64-20260526)
and place it in this directory beside `install.sh`.

## 1. Install

From this directory:

```bash
cd /path/to/pc_metrology/CloudCompare
chmod +x install.sh run.sh
./install.sh
```

The installer creates a local Python 3.12 environment, installs the required
Python packages, extracts CloudComPy, and verifies `import cloudComPy`.
Installation is rerunnable.

If you have a verified direct archive URL, the installer can download it:

```bash
CLOUDCOMPY_URL='https://example.com/CloudComPy_Conda312_Linux64_20260528-1022.tar.xz' ./install.sh
```

Do not use the release page itself as `CLOUDCOMPY_URL`; it must point directly
to the `.tar.xz` file.

## 2. Export a BIN project

Run the exporter from this directory:

```bash
./run.sh ./export_clouds.py /path/to/project.bin
```

By default, TXT files are written beside the BIN file. To select an output
directory:

```bash
./run.sh ./export_clouds.py /path/to/project.bin --output /path/to/output
```

Relative paths are resolved from the directory where `run.sh` is invoked, so
the launcher can also be called from the repository root:

```bash
./CloudCompare/run.sh \
  ./CloudCompare/export_clouds.py \
  ./input/u3dl/U3DL.bin \
  --output ./input/u3dl/output/point_clouds
```

Use `--overwrite` to replace existing TXT files:

```bash
./run.sh ./export_clouds.py /path/to/project.bin \
  --output /path/to/output --overwrite
```

Without `--overwrite`, existing filenames are preserved and new files receive
numeric suffixes such as `_2`.

## Output and limitations

- One `.txt` file is written per standalone cloud returned by
  `cloudComPy.importFile`.
- Filenames are derived from cloud entity names and sanitized for filesystem
  safety.
- Mesh-owned or hierarchy-nested clouds may not be included.
- TXT columns are selected by CloudComPy's ASCII writer. The tested
  `examples/data/DetectorVertical.bin` fixture produced space-delimited XYZ
  values with eight decimal places and no header.
- Headless execution uses Qt's `offscreen` platform. Qt thread-shutdown or
  missing fontconfig warnings may appear even when import and export succeed.
- Missing host shared libraries cannot always be fixed without administrator
  access.
