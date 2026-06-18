#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/linsixu/simtoolreal}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/handoff_packages}"
PKG_NAME="franka_brainco_revo2_aerial_env_${STAMP}"
PKG_DIR="$OUT_ROOT/$PKG_NAME"

cd "$ROOT"
mkdir -p "$PKG_DIR"

copy_path() {
  local rel="$1"
  local src="$ROOT/$rel"
  local dst="$PKG_DIR/$rel"
  if [ ! -e "$src" ]; then
    echo "[skip] missing $rel"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "[copy] $rel"
}

copy_glob() {
  local pattern="$1"
  local matched=0
  while IFS= read -r rel; do
    matched=1
    copy_path "$rel"
  done < <(find . -path "./$pattern" -type f | sed 's#^\./##' | sort)
  if [ "$matched" = "0" ]; then
    echo "[skip] no matches for $pattern"
  fi
}

copy_path "docs/handoff/franka_brainco_revo2_aerial_env"
copy_path "docs/brainco_revo2_embodiment.md"
copy_path "docs/project_page/assets/images/falling_baton_game_schematic.png"
copy_path "docs/project_page/assets/images/aerial_object_catch_reference.png"

copy_path "scripts/build_franka_hand_asset.py"
copy_path "scripts/prepare_franka_brainco_revo2_asset.sh"
copy_path "scripts/inspect_robot_asset_urdf.py"
copy_path "scripts/preview_v33_franka_inspire_env.py"
copy_path "scripts/preview_dg_franka_brainco_revo2_env.sh"
copy_path "scripts/preview_franka_brainco_revo2_aerial_envs.sh"
copy_path "scripts/run_revo2_scripted_grasp_sanity.py"

copy_path "assets/generated/franka_brainco_revo2_right"
if [ -d "$PKG_DIR/assets/generated/franka_brainco_revo2_right/hand" ]; then
  find "$PKG_DIR/assets/generated/franka_brainco_revo2_right/hand" -path "*left*" -type f -delete
  find "$PKG_DIR/assets/generated/franka_brainco_revo2_right/hand" -type d -empty -delete
  echo "[prune] left-hand Revo2 files from package copy"
fi
copy_path "assets/urdf/dextoolbench/marker"
copy_path "assets/urdf/dextoolbench/screwdriver"
copy_path "assets/affordance_labels/dextoolbench/marker"
copy_path "assets/affordance_labels/dextoolbench/screwdriver"
copy_glob "assets/affordance_labels/analysis/*clean_v2*"
copy_glob "assets/affordance_labels/visualizations/*clean_v2*"

copy_path "isaacgymenvs/tasks/simtoolreal/env.py"
copy_path "isaacgymenvs/cfg/task"

find "$PKG_DIR" -type f | sed "s#^$PKG_DIR/##" | sort > "$PKG_DIR/PACKAGE_CONTENTS.txt"

TARBALL="$OUT_ROOT/$PKG_NAME.tar.gz"
tar -czf "$TARBALL" -C "$OUT_ROOT" "$PKG_NAME"

echo
echo "Package directory: $PKG_DIR"
echo "Tarball: $TARBALL"
echo "Overlay extract command:"
echo "  tar -xzf $TARBALL --strip-components=1 -C /path/to/simtoolreal"
du -h "$TARBALL"
