#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$VIS_DIR/../../.." && pwd)"
DATA_FILE="$VIS_DIR/data/nn_overview.json"
MEDIA_DIR="exercise3/visualizations/renders"
SCENE_FILE="exercise3/visualizations/manim/nn_overview_scene.py"
SCENE_NAME="Exercise3NNOverview"

if [[ ! -f "$DATA_FILE" ]]; then
  printf 'Missing visualization data: %s\n' "$DATA_FILE" >&2
  printf 'Generate it first with:\n' >&2
  printf '  python3 exercise3/visualization_data.py\n' >&2
  exit 1
fi

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$REPO_ROOT:/work" \
  -w /work \
  manimcommunity/manim:stable \
  manim -qh "$SCENE_FILE" "$SCENE_NAME" --media_dir "$MEDIA_DIR"

printf '\nRendered Manim output under %s/%s\n' "$REPO_ROOT" "$MEDIA_DIR"
