#!/usr/bin/env bash
# Install CloudComPy locally under this script's directory (no sudo).
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TOOLS="$ROOT/.tools"
CACHE="$ROOT/.cache"
PYTHONS="$ROOT/.python"
VENV="$ROOT/.venv"
ARCHIVE_NAME="CloudComPy_Conda312_Linux64_20260528-1022.tar.xz"
ARCHIVE="${CLOUDCOMPY_ARCHIVE:-$ROOT/$ARCHIVE_NAME}"
DOWNLOAD_URL="${CLOUDCOMPY_URL:-}"
log() { printf '\n==> %s\n' "$*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }
command -v curl >/dev/null || fail 'curl is required; ask your administrator or copy the release archive and the uv executable manually.'
command -v tar >/dev/null || fail 'tar is required.'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This release requires x86_64 Linux.'
[[ -r /etc/os-release ]] && { . /etc/os-release; echo "System: ${PRETTY_NAME:-Linux}"; }
if command -v ldd >/dev/null; then
  GLIBC="$(ldd --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | tail -n1 || true)"
  if [[ -n "$GLIBC" ]] && [[ "$(printf '%s\n' 2.28 "$GLIBC" | sort -V | head -n1)" != '2.28' ]]; then
    fail "GLIBC $GLIBC is too old; minimum 2.28."
  fi
fi
mkdir -p "$TOOLS" "$CACHE" "$PYTHONS"
export UV_CACHE_DIR="$CACHE/uv"
export UV_PYTHON_INSTALL_DIR="$PYTHONS"
export UV_PYTHON_BIN_DIR="$TOOLS/python-bin"
export UV_NO_MODIFY_PATH=1
export UV_INSTALL_DIR="$TOOLS/bin"
export UV_PYTHON_PREFERENCE=only-managed
UV="$TOOLS/bin/uv"
if [[ ! -x "$UV" ]]; then
  log 'Installing the uv tool inside CloudCompare/.tools/'
  # Official installer. All uv paths are explicitly redirected into this directory.
  curl --fail --location --show-error --silent https://astral.sh/uv/install.sh | sh
fi
[[ -x "$UV" ]] || fail "uv not available at $UV"
log 'Creating isolated Python 3.12 virtual environment'
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV" venv --python 3.12 --python-preference only-managed "$VENV"
fi
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail 'Virtual environment must use Python 3.12.'
log 'Installing CloudComPy Python dependencies'
"$UV" pip install --python "$VENV/bin/python" numpy scipy requests psutil matplotlib numpy-quaternion cmake
if [[ ! -f "$ARCHIVE" ]]; then
  if [[ -n "$DOWNLOAD_URL" ]]; then
    log 'Downloading the CloudComPy archive from the provided direct URL'
    ARCHIVE="$ROOT/$ARCHIVE_NAME"
    curl --fail --location --show-error --retry 3 --output "$ARCHIVE.part" "$DOWNLOAD_URL"
    mv "$ARCHIVE.part" "$ARCHIVE"
  else
    cat >&2 <<MSG

CloudComPy archive is missing:
  $ARCHIVE

Download '$ARCHIVE_NAME' from the official release page:
  https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/17-production-cloudcompy-binaries/4-linux-cloudcompy-binaries/126-cloudcompy-conda312-linux64-20260526

Place it alongside install.sh, then rerun. Alternatively set
  CLOUDCOMPY_URL='DIRECT_ARCHIVE_URL' ./install.sh
The page itself is NOT a direct archive URL.
MSG
    exit 2
  fi
fi
log 'Checking downloaded archive'
tar -tJf "$ARCHIVE" > "$CACHE/archive-contents.txt" || fail 'Invalid .tar.xz archive.'
if grep -Eq '(^|/)\.\./|^/' "$CACHE/archive-contents.txt"; then
  fail 'Archive contains unsafe absolute or parent-relative paths.'
fi
if [[ ! -f "$ROOT/CloudComPy312/bin/envCloudComPy.sh" ]]; then
  log 'Extracting CloudComPy to CloudCompare/'
  tar -xJf "$ARCHIVE" -C "$ROOT" --no-same-owner
fi
ENV_SCRIPT="$ROOT/CloudComPy312/bin/envCloudComPy.sh"
[[ -f "$ENV_SCRIPT" ]] || fail "Expected environment setup script missing at $ENV_SCRIPT. Inspect extracted directory structure."
log 'Testing CloudComPy Python import'
# Source packaged environment in a subshell; it may adjust PATH, PYTHONPATH and libraries.
(
  source "$VENV/bin/activate"
  cd "$ROOT/CloudComPy312"
  # The upstream environment script reads optional variables without defaults.
  set +u
  source "$ENV_SCRIPT" activate
  set -u
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
  "$VENV/bin/python" -c 'import cloudComPy as cc; print("CloudComPy import: OK", cc.__file__)'
) || fail 'Import failed; inspect missing shared libraries or Qt compatibility. The binary has not been validated on this server.'
log 'Installation complete'
echo "Run: $ROOT/run.sh $ROOT/export_clouds.py /path/to/project.bin"
