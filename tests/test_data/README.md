# Test Data Directory

This directory contains test artifacts used by integration tests.

## Checkpoints

The `checkpoints/` subdirectory should contain checkpoint files for integration testing.

### Setting Up Test Checkpoint

Use the setup script to copy a checkpoint:

```bash
./tests/setup_test_checkpoint.sh [checkpoint_path]
```

Or manually:
```bash
cp -r /path/to/checkpoint tests/test_data/checkpoints/
```

### Running Integration Tests

For local testing, you can:
1. Use the checkpoint in `tests/test_data/checkpoints/` (if available)
2. Set the `TEST_CHECKPOINT_PATH` environment variable to point to your checkpoint

Example:
```bash
export TEST_CHECKPOINT_PATH=/path/to/your/checkpoint
uv run python -m unittest tests.test_export_onnx_jax_integration
```

Or run with default path (if checkpoint exists at default location):
```bash
uv run python -m unittest tests.test_export_onnx_jax_integration
```

### CI/CD

For CI/CD pipelines, checkpoint files should be:
- Stored as test artifacts
- Downloaded before running integration tests
- Referenced via the `TEST_CHECKPOINT_PATH` environment variable

The integration tests will skip if no checkpoint is available, so they won't fail in CI if checkpoints aren't provided.
