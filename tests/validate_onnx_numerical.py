# Copyright (C) 2025 Julia Jia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Validate ONNX model numerical correctness by comparing with JAX inference.

Usage:
    uv run python tests/validate_onnx_numerical.py --onnx checkpoints/model.onnx
    uv run python tests/validate_onnx_numerical.py --checkpoints-dir checkpoints
"""

import argparse
import json
import os
import numpy as np
import onnxruntime as ort
from orbax import checkpoint as ocp

from playground.common.export_jax_to_onnx import make_jax_inference_fn
from mujoco_playground.config import locomotion_params


def load_metadata(onnx_path):
    """Load metadata from .metadata.json file."""
    metadata_path = f"{onnx_path}.metadata.json"
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def create_test_inputs(obs_size, num_inputs=5):
    """Create test inputs for validation."""
    return [
        np.random.randn(1, obs_size).astype(np.float32),  # Random
        np.zeros((1, obs_size), dtype=np.float32),  # Zero
        np.ones((1, obs_size), dtype=np.float32),  # Unit
        np.random.randn(1, obs_size).astype(np.float32) * 2.0,  # Large
        np.random.randn(1, obs_size).astype(np.float32) * 0.5,  # Small
    ][:num_inputs]


def validate_single_model(onnx_path, mae_threshold=1e-4, mse_threshold=1e-8, max_error_threshold=1e-3):
    """Validate a single ONNX model against JAX inference.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX file
    mae_threshold : float
        Maximum allowed mean absolute error (default: 1e-4)
    mse_threshold : float
        Maximum allowed mean squared error (default: 1e-8)
    max_error_threshold : float
        Maximum allowed per-element error (default: 1e-3)
    
    Returns:
    --------
    dict
        Validation results with pass/fail status and error metrics
    """
    # Load metadata
    metadata = load_metadata(onnx_path)
    if not metadata:
        return {
            "status": "FAIL",
            "error": "Metadata file not found",
            "onnx_path": onnx_path
        }
    
    obs_size = metadata["obs_size"]
    act_size = metadata["act_size"]
    checkpoint_path = metadata.get("checkpoint_path")
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {
            "status": "FAIL",
            "error": f"Checkpoint not found: {checkpoint_path}",
            "onnx_path": onnx_path
        }
    
    try:
        # Load checkpoint and create JAX inference function
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(checkpoint_path)
        if isinstance(params, list):
            params = tuple(params)
        
        ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
        jax_inference_fn = make_jax_inference_fn(params, act_size, ppo_params, obs_size)
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Generate test inputs
        test_inputs = create_test_inputs(obs_size, num_inputs=5)
        
        # Compare outputs
        all_errors = []
        for test_input in test_inputs:
            # JAX inference
            import jax.numpy as jnp
            jax_output = np.array(jax_inference_fn(jnp.array(test_input)))
            
            # ONNX inference
            onnx_output = session.run([output_name], {input_name: test_input})[0]
            
            # Compute errors
            error = np.abs(jax_output - onnx_output)
            all_errors.append(error)
        
        # Aggregate errors
        all_errors = np.concatenate(all_errors)
        mae = np.mean(all_errors)
        mse = np.mean(all_errors ** 2)
        max_error = np.max(all_errors)
        
        # Check thresholds
        passed = (
            mae < mae_threshold and
            mse < mse_threshold and
            max_error < max_error_threshold
        )
        
        return {
            "status": "PASS" if passed else "FAIL",
            "onnx_path": onnx_path,
            "mae": float(mae),
            "mse": float(mse),
            "max_error": float(max_error),
            "mae_threshold": mae_threshold,
            "mse_threshold": mse_threshold,
            "max_error_threshold": max_error_threshold,
            "num_test_inputs": len(test_inputs)
        }
        
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "onnx_path": onnx_path
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX model numerical correctness"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="Path to ONNX file or directory containing ONNX files"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints and ONNX files (alternative to --onnx)"
    )
    parser.add_argument(
        "--mae-threshold",
        type=float,
        default=1e-4,
        help="Maximum allowed mean absolute error (default: 1e-4)"
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=1e-8,
        help="Maximum allowed mean squared error (default: 1e-8)"
    )
    parser.add_argument(
        "--max-error-threshold",
        type=float,
        default=1e-3,
        help="Maximum allowed per-element error (default: 1e-3)"
    )
    
    args = parser.parse_args()
    
    if args.onnx:
        target_path = args.onnx
    else:
        target_path = args.checkpoints_dir
    
    if not os.path.exists(target_path):
        print(f"Error: Path does not exist: {target_path}")
        return
    
    # Find ONNX files
    if os.path.isdir(target_path):
        onnx_files = [
            os.path.join(target_path, f) 
            for f in os.listdir(target_path) 
            if f.endswith(".onnx")
        ]
    else:
        onnx_files = [target_path]
    
    if not onnx_files:
        print(f"No ONNX files found in {target_path}")
        return
    
    print(f"Validating {len(onnx_files)} ONNX model(s)...")
    print(f"Thresholds: MAE < {args.mae_threshold}, MSE < {args.mse_threshold}, Max Error < {args.max_error_threshold}")
    print()
    
    results = []
    for onnx_path in sorted(onnx_files):
        result = validate_single_model(
            onnx_path,
            args.mae_threshold,
            args.mse_threshold,
            args.max_error_threshold
        )
        results.append(result)
        
        status = result["status"]
        filename = os.path.basename(onnx_path)
        
        if status == "PASS":
            print(f"✓ {filename}")
            print(f"  MAE: {result['mae']:.2e}, MSE: {result['mse']:.2e}, Max Error: {result['max_error']:.2e}")
        else:
            print(f"✗ {filename}")
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  MAE: {result['mae']:.2e}, MSE: {result['mse']:.2e}, Max Error: {result['max_error']:.2e}")
        print()
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    
    print(f"Summary: {passed}/{len(results)} passed, {failed} failed")


if __name__ == "__main__":
    main()

