# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.pkl` in `playground/<robot>/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run playground/<robot>/runner.py 
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx>
```

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

### ONNX Export Options

By default, the runner uses TensorFlow-based ONNX export. For newer GPUs (e.g., RTX 5090) or to avoid TensorFlow dependencies, use the direct JAX-to-ONNX export:

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000 --use_jax_to_onnx
```

The `--use_jax_to_onnx` flag enables direct JAX-to-ONNX conversion, which:
- Avoids TensorFlow dependency
- Resolves CUDA compatibility issues on newer GPUs
- Uses the same ONNX output format (opset 11, compatible with Isaac Lab)


## Dependencies

Sync dependencies:

```bash
uv sync
```

### Updating Dependencies with Nightly Builds

This project uses PyTorch nightly builds for RTX 5090 support. To update PyTorch packages or add new dependencies that require the nightly index:

```bash
uv lock --upgrade-package torch --upgrade-package torchvision --upgrade-package torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match
uv sync
```

The `--index-strategy unsafe-best-match` flag allows uv to check all indexes (PyPI and PyTorch nightly) to find the best matching versions. This is needed because some packages may only be available on PyPI while PyTorch nightly builds are on a separate index.

To add a new package that requires the nightly index:

```bash
uv lock --upgrade-package warp-lang --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match
uv lock --upgrade-package <package_name> --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match
uv sync
```

## Running Tests

Run all unit tests:

```bash
uv run python -m unittest discover tests
```

Run a specific test file:

```bash
uv run python -m unittest tests.test_export_onnx_jax
```

Run with verbose output:

```bash
uv run python -m unittest tests.test_export_onnx_jax -v
```
