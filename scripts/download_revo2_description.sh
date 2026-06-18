#!/usr/bin/env bash
set -euo pipefail

cd /data1/linsixu/simtoolreal

REPO_ZIP_URL="${REVO2_DESCRIPTION_ZIP_URL:-https://github.com/BrainCoTech/revo2_description/archive/refs/heads/main.zip}"
OUTPUT_DIR="${REVO2_DESCRIPTION_OUTPUT_DIR:-/data1/linsixu/simtoolreal/assets/generated/revo2_description}"
TMP_ZIP="${REVO2_DESCRIPTION_TMP_ZIP:-/tmp/brainco_revo2_description_main.zip}"
TMP_DIR="$(mktemp -d /tmp/revo2_description_extract.XXXXXX)"

if [ -e "$OUTPUT_DIR" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "$OUTPUT_DIR already exists. Set FORCE=1 to overwrite."
  exit 0
fi

curl -L "$REPO_ZIP_URL" -o "$TMP_ZIP"
unzip -q -o "$TMP_ZIP" -d "$TMP_DIR"

if [ "${FORCE:-0}" = "1" ]; then
  rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
cp -a "$TMP_DIR"/revo2_description-* "$OUTPUT_DIR"

echo "Downloaded BrainCo Revo2 description to $OUTPUT_DIR"
echo "Right hand URDF: $OUTPUT_DIR/urdf/revo2_right_hand.urdf"
echo "Left hand URDF:  $OUTPUT_DIR/urdf/revo2_left_hand.urdf"
