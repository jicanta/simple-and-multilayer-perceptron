#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$VIS_DIR/../.." && pwd)"
DATA_FILE="$VIS_DIR/data/final_model_overview.json"
MEDIA_DIR="exercise1/visualizations/renders"
SCENE_FILE="exercise1/visualizations/manim/final_model_scene.py"
SCENE_NAME="Exercise1FinalModelOverview"

if [[ ! -f "$DATA_FILE" ]]; then
  printf 'Missing visualization data: %s\n' "$DATA_FILE" >&2
  printf 'Generate it first with: python3 exercise1/train.py\n' >&2
  exit 1
fi

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$REPO_ROOT:/work" \
  -w /work \
  manimcommunity/manim:stable \
  manim -qh "$SCENE_FILE" "$SCENE_NAME" --media_dir "$MEDIA_DIR"

printf '\nRendered Manim output under %s/%s\n' "$REPO_ROOT" "$MEDIA_DIR"
