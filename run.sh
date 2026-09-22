#!/usr/bin/env bash
# One command to set up and start laya-snake on Linux or macOS.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
exec "$PY" "$DIR/play.py" "$@"
