#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MEDIA_DIR="validation/renders"
SCENE_FILE="validation/manim/validation_scene.py"
SCENE_NAME="ValidationRegressionScene"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$REPO_ROOT:/work" \
  -w /work \
  manimcommunity/manim:stable \
  manim -qh "$SCENE_FILE" "$SCENE_NAME" --media_dir "$MEDIA_DIR"

printf '\nRendered to %s/%s\n' "$REPO_ROOT" "$MEDIA_DIR"
