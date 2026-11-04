#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run install.sh first' >&2; exit 1; }
[[ $# -ge 1 ]] || { echo "Usage: $0 SCRIPT.py [args ...]" >&2; exit 2; }
CALLER_CWD="$PWD"
SCRIPT="$1"
if [[ "$SCRIPT" != /* ]]; then
  SCRIPT="$CALLER_CWD/$SCRIPT"
fi
[[ -f "$SCRIPT" ]] || { echo "Python script not found: $SCRIPT" >&2; exit 2; }
shift
source "$ROOT/.venv/bin/activate"
# The upstream environment script reads optional variables without defaults.
set +u
source "$ROOT/CloudComPy312/bin/envCloudComPy.sh" activate
set -u
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$ROOT/.cache/runtime}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
exec "$ROOT/.venv/bin/python" "$SCRIPT" "$@"
