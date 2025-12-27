#!/usr/bin/env python3
"""
Debug script to inspect checkpoint structure and test ONNX export.

This script is for debugging purposes only and should not be run as part of
the test suite. It requires a real checkpoint file to function.
"""

import sys
from pathlib import Path
from orbax import checkpoint as ocp
import jax

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.common.export_jax_to_onnx import export_onnx_jax
from mujoco_playground.config import locomotion_params


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint using PyTreeCheckpointer (same as how it's saved)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_data = checkpointer.restore(checkpoint_path)
    
    print("\n=== Checkpoint Structure ===")
    print(f"Type: {type(ckpt_data)}")
    
    if isinstance(ckpt_data, (list, tuple)):
        print(f"Length: {len(ckpt_data)}")
        for i, p in enumerate(ckpt_data):
            print(f"\n  Element [{i}] type: {type(p)}")
            if hasattr(p, 'mean'):
                print(f"    Has 'mean' attribute")
                try:
                    if hasattr(p.mean, 'keys'):
                        print(f"    mean keys: {list(p.mean.keys())}")
                    if 'state' in p.mean:
                        print(f"    mean['state'] shape: {p.mean['state'].shape}, dtype: {p.mean['state'].dtype}")
                except:
                    pass
            if hasattr(p, 'std'):
                print(f"    Has 'std' attribute")
                try:
                    if hasattr(p.std, 'keys'):
                        print(f"    std keys: {list(p.std.keys())}")
                    if 'state' in p.std:
                        print(f"    std['state'] shape: {p.std['state'].shape}, dtype: {p.std['state'].dtype}")
                except:
                    pass
            if isinstance(p, dict):
                print(f"    Dict keys: {list(p.keys())[:20]}")
                if 'mean' in p:
                    print(f"    Has 'mean' key")
                    if isinstance(p['mean'], dict) and 'state' in p['mean']:
                        mean = p['mean']['state']
                        print(f"    mean['state'] shape: {mean.shape}, dtype: {mean.dtype}")
                if 'std' in p:
                    print(f"    Has 'std' key")
                    if isinstance(p['std'], dict) and 'state' in p['std']:
                        std = p['std']['state']
                        print(f"    std['state'] shape: {std.shape}, dtype: {std.dtype}")
                if 'policy' in p:
                    print(f"    Has 'policy' key")
                    policy = p['policy']
                    print(f"    policy type: {type(policy)}")
                    if isinstance(policy, dict):
                        print(f"    policy keys: {list(policy.keys())}")
                        if 'params' in policy:
                            print(f"    policy['params'] type: {type(policy['params'])}")
                            if isinstance(policy['params'], dict):
                                print(f"    policy['params'] keys: {list(policy['params'].keys())[:10]}...")
                                # Check first layer
                                if len(policy['params']) > 0:
                                    first_key = list(policy['params'].keys())[0]
                                    print(f"    First key: {first_key}")
                                    first_layer = policy['params'][first_key]
                                    print(f"    {first_key} type: {type(first_layer)}")
                                    if isinstance(first_layer, dict):
                                        print(f"    {first_key} keys: {list(first_layer.keys())}")
                elif 'params' in p:
                    print(f"    Has 'params' key (not 'policy')")
                    params = p['params']
                    print(f"    params type: {type(params)}")
                    if isinstance(params, dict):
                        print(f"    params keys: {list(params.keys())[:10]}...")
                        if len(params) > 0:
                            first_key = list(params.keys())[0]
                            first_layer = params[first_key]
                            print(f"    First key: {first_key}")
                            if isinstance(first_layer, dict):
                                print(f"    {first_key} keys: {list(first_layer.keys())}")
                                if 'kernel' in first_layer:
                                    print(f"    {first_key}['kernel'] shape: {first_layer['kernel'].shape}")
                                if 'bias' in first_layer:
                                    print(f"    {first_key}['bias'] shape: {first_layer['bias'].shape}")
    
    return ckpt_data


def test_export(checkpoint_path: str, output_path: str = "test_export.onnx"):
    """Test ONNX export with the checkpoint."""
    print(f"\n=== Testing ONNX Export ===")
    
    # Load checkpoint using PyTreeCheckpointer
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(checkpoint_path)
    
    # Convert list to tuple if needed
    if isinstance(params, list):
        params = tuple(params)
        print(f"Converted params from list to tuple, length: {len(params)}")
    
    # Get config
    ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
    
    # Get actual sizes from checkpoint
    if len(params) >= 1:
        norm_params = params[0]
        if isinstance(norm_params, dict) and 'mean' in norm_params and 'state' in norm_params['mean']:
            mean = norm_params['mean']['state']
            obs_size = mean.shape[0] if len(mean.shape) == 1 else mean.shape[-1]
            print(f"Detected obs_size from checkpoint: {obs_size} (mean shape: {mean.shape})")
        else:
            obs_size = 46
            print(f"Could not detect obs_size, using default: {obs_size}")
    else:
        obs_size = 46
        print(f"Using default obs_size: {obs_size}")
    
    # Detect act_size from checkpoint parameters
    # The last layer output should be act_size * 2 (loc + log_std)
    if len(params) >= 2:
        policy_params_raw = params[1]
        if isinstance(policy_params_raw, dict) and 'params' in policy_params_raw:
            nested = policy_params_raw['params']
            if isinstance(nested, dict) and len(nested) > 0:
                # Get the last layer (usually named hidden_N or output)
                layer_names = sorted(nested.keys())
                last_layer_name = layer_names[-1]
                last_layer = nested[last_layer_name]
                if isinstance(last_layer, dict) and 'kernel' in last_layer:
                    kernel = last_layer['kernel']
                    # The kernel shape is (input_size, output_size)
                    # output_size should be act_size * 2
                    output_size = kernel.shape[-1]
                    act_size = output_size // 2
                    print(f"Detected act_size from checkpoint: {act_size} (last layer output size: {output_size})")
                else:
                    act_size = 10
                    print(f"Could not detect act_size from last layer, using default: {act_size}")
            else:
                act_size = 10
                print(f"Could not detect act_size, using default: {act_size}")
        else:
            act_size = 10
            print(f"Could not detect act_size, using default: {act_size}")
    else:
        act_size = 10
        print(f"Could not detect act_size, using default: {act_size}")
    
    # Test if the inference function can be traced before attempting ONNX export
    print("\n=== Testing Function Traceability ===")
    try:
        from playground.common.export_jax_to_onnx import make_jax_inference_fn
        import jax.numpy as jnp
        
        # Create inference function
        jax_inference_fn = make_jax_inference_fn(params, act_size, ppo_params, obs_size)
        
        # Test with a concrete input
        test_input = jnp.zeros((1, obs_size), dtype=jnp.float32)
        test_output = jax_inference_fn(test_input)
        print(f"✓ Function executes successfully")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {test_output.shape}")
        
        # Test JIT compilation (this will reveal tracing issues)
        jax_inference_fn_jit = jax.jit(jax_inference_fn)
        test_output_jit = jax_inference_fn_jit(test_input)
        print(f"✓ Function can be JIT compiled (traceable)")
        
    except Exception as e:
        print(f"✗ Function traceability test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Fix traceability issues before attempting ONNX export")
        return
    
    # Now attempt ONNX export
    print("\n=== Attempting ONNX Export ===")
    try:
        export_onnx_jax(
            params,
            act_size,
            ppo_params,
            obs_size,
            output_path=output_path
        )
        print(f"✓ ONNX export successful! Saved to: {output_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()


# Note: This script is not meant to be run directly.
# To use it, import the functions and call them manually:
#
# from tests.debug_checkpoint import inspect_checkpoint, test_export
# checkpoint_path = "/path/to/checkpoint"
# inspect_checkpoint(checkpoint_path)
# test_export(checkpoint_path)

