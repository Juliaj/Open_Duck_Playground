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
Generate metadata for existing ONNX models.

This script scans a checkpoints directory, finds all .onnx files,
matches them to their checkpoint directories, and generates metadata.

Usage:
    uv run python tests/generate_metadata_for_existing.py --checkpoints-dir checkpoints
"""

import argparse
import os
import re
from pathlib import Path
from orbax import checkpoint as ocp

from playground.common.export_jax_to_onnx import (
    export_onnx_jax, make_jax_inference_fn, extract_norm_params
)
from mujoco_playground.config import locomotion_params


def parse_checkpoint_name(onnx_filename):
    """Extract checkpoint directory name from ONNX filename.
    
    Examples:
        "2025_12_26_163941_85852160.onnx" -> "2025_12_26_163941_85852160"
        "2025_12_26_163220_0.onnx" -> "2025_12_26_163220_0"
    """
    return os.path.splitext(onnx_filename)[0]


def find_checkpoint_dir(checkpoints_dir, checkpoint_name):
    """Find checkpoint directory matching the name."""
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        # Convert to absolute path (required by orbax)
        return os.path.abspath(checkpoint_path)
    return None


def extract_training_step(checkpoint_name):
    """Extract training step from checkpoint name.
    
    Examples:
        "2025_12_26_163941_85852160" -> 85852160
        "2025_12_26_163220_0" -> 0
    """
    parts = checkpoint_name.split("_")
    if len(parts) >= 6:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


def generate_metadata_for_onnx(onnx_path, checkpoints_dir, reward=None, reward_std=None):
    """Generate metadata for a single ONNX file.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX file
    checkpoints_dir : str
        Directory containing checkpoints
    reward : float, optional
        Training reward value
    reward_std : float, optional
        Training reward std value
    """
    onnx_filename = os.path.basename(onnx_path)
    checkpoint_name = parse_checkpoint_name(onnx_filename)
    checkpoint_dir = find_checkpoint_dir(checkpoints_dir, checkpoint_name)
    
    if not checkpoint_dir:
        print(f"Warning: Could not find checkpoint directory for {onnx_filename}")
        return False
    
    print(f"Processing {onnx_filename}...")
    
    try:
        # Load checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(checkpoint_dir)
        
        if isinstance(params, list):
            params = tuple(params)
        
        # Get config
        ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
        
        # Detect sizes from checkpoint
        mean, std = extract_norm_params(params)
        obs_size = mean.shape[0] if len(mean.shape) == 1 else mean.shape[-1]
        
        # Detect act_size from checkpoint
        if len(params) >= 2:
            policy_params_raw = params[1]
            if isinstance(policy_params_raw, dict) and 'params' in policy_params_raw:
                nested = policy_params_raw['params']
                if isinstance(nested, dict) and len(nested) > 0:
                    layer_names = sorted(nested.keys())
                    last_layer_name = layer_names[-1]
                    last_layer = nested[last_layer_name]
                    if isinstance(last_layer, dict) and 'kernel' in last_layer:
                        kernel = last_layer['kernel']
                        output_size = kernel.shape[-1]
                        act_size = output_size // 2
                    else:
                        print(f"Warning: Could not detect act_size from {onnx_filename}")
                        return False
                else:
                    print(f"Warning: Could not detect act_size from {onnx_filename}")
                    return False
            else:
                print(f"Warning: Could not detect act_size from {onnx_filename}")
                return False
        else:
            print(f"Warning: Could not detect act_size from {onnx_filename}")
            return False
        
        # Extract training step
        training_step = extract_training_step(checkpoint_name)
        
        # Generate metadata using the same logic as export_onnx_jax
        from playground.common import export_jax_to_onnx
        
        # Use the _save_metadata function directly
        export_jax_to_onnx._save_metadata(
            onnx_path,
            params,
            act_size,
            obs_size,
            ppo_params,
            checkpoint_dir,
            training_step,
            reward,
            reward_std
        )
        
        print(f"✓ Generated metadata for {onnx_filename}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to generate metadata for {onnx_filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata for existing ONNX models"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="Path to specific ONNX file to process"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints and ONNX files"
    )
    parser.add_argument(
        "--reward",
        type=float,
        help="Training reward value (optional)"
    )
    parser.add_argument(
        "--reward-std",
        type=float,
        help="Training reward std value (optional)"
    )
    
    args = parser.parse_args()
    
    if args.onnx:
        # Process single file
        if not os.path.exists(args.onnx):
            print(f"Error: ONNX file does not exist: {args.onnx}")
            return
        checkpoints_dir = os.path.dirname(args.onnx) or args.checkpoints_dir
        onnx_files = [args.onnx]
    else:
        # Process all files in directory
        checkpoints_dir = args.checkpoints_dir
        if not os.path.exists(checkpoints_dir):
            print(f"Error: Checkpoints directory does not exist: {checkpoints_dir}")
            return
        
        onnx_files = []
        for filename in os.listdir(checkpoints_dir):
            if filename.endswith(".onnx"):
                onnx_path = os.path.join(checkpoints_dir, filename)
                onnx_files.append(onnx_path)
        
        if not onnx_files:
            print(f"No ONNX files found in {checkpoints_dir}")
            return
    
    print(f"Found {len(onnx_files)} ONNX file(s)")
    if args.reward is not None:
        print(f"Using reward: {args.reward}, reward_std: {args.reward_std}")
    print()
    
    success_count = 0
    for onnx_path in sorted(onnx_files):
        if generate_metadata_for_onnx(onnx_path, checkpoints_dir, args.reward, args.reward_std):
            success_count += 1
    
    print()
    print(f"Completed: {success_count}/{len(onnx_files)} metadata files generated")


if __name__ == "__main__":
    main()

