#!/bin/bash
set -e
set -x

BASE_DIR=/devel/tmp/artifact
DATA_DIR="$BASE_DIR/cgo26_stenso"
EXTRACT_DIR="$BASE_DIR/extracted"
TARBALL_NAME="artifact.tar.gz"

echo "--- 1. Staging files into $DATA_DIR ---"
rm -rf "$BASE_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$EXTRACT_DIR"

cp -r build_tools mlirSynth stenso tensorize CMakeLists.txt Dockerfile requirements.txt test run_all.sh "$DATA_DIR"

echo "--- 2. Creating $TARBALL_NAME from $DATA_DIR ---"
tar -czf "$TARBALL_NAME" -C "$BASE_DIR" "$(basename "$DATA_DIR")"

echo "--- 3. Archive verification ---"
echo -n "Archive Size: "
du -sh "$TARBALL_NAME" | cut -f1

echo "--- 4. Extracting $TARBALL_NAME into $EXTRACT_DIR ---"
tar -xzf "$TARBALL_NAME" -C "$EXTRACT_DIR"

echo "Extraction complete. Contents are in $EXTRACT_DIR/data"