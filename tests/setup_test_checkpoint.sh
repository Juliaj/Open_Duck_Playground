#!/bin/bash
# Setup script to copy a checkpoint to test_data directory for integration tests

set -e

CHECKPOINT_SOURCE="${1:-/home/juliajia/dev/Open_Duck_Playground/checkpoints/2025_12_26_160751_0}"
TEST_DATA_DIR="$(dirname "$0")/test_data/checkpoints"

if [ ! -d "$CHECKPOINT_SOURCE" ]; then
    echo "Error: Checkpoint source directory not found: $CHECKPOINT_SOURCE"
    echo "Usage: $0 [checkpoint_path]"
    exit 1
fi

echo "Setting up test checkpoint..."
echo "  Source: $CHECKPOINT_SOURCE"
echo "  Destination: $TEST_DATA_DIR"

mkdir -p "$TEST_DATA_DIR"

# Copy checkpoint
cp -r "$CHECKPOINT_SOURCE" "$TEST_DATA_DIR/"

echo "✓ Checkpoint copied to test_data directory"
echo ""
echo "To use this checkpoint in tests, set:"
echo "  export TEST_CHECKPOINT_PATH=$TEST_DATA_DIR/$(basename "$CHECKPOINT_SOURCE")"
echo ""
echo "Or tests will use the default path if available."

