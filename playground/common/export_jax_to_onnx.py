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

"""JAX to ONNX export using direct conversion (jax2onnx)."""

import functools
import inspect
from typing import Callable
import os
import warnings
import json
import numpy as np
from datetime import datetime

import jax
import jax.numpy as jnp
# import RL policies from Brax. Brax is also a physics engine for RL but not used for this project.
from brax.training.agents.ppo import networks as ppo_networks
from jax2onnx import to_onnx


def export_onnx_jax(
    params, act_size, ppo_params, obs_size, output_path="open_duck_mini_v2_jax.onnx",
    checkpoint_path=None, training_step=None, reward=None, reward_std=None
)->str:
    """
    Export JAX policy to ONNX format using jax2onnx (direct conversion).
    
    This function directly converts JAX inference function to ONNX without
    going through TensorFlow, avoiding CUDA compatibility issues on new GPUs like the RTX 5090.
    
    Note: jax2onnx may fail with certain Flax network operations that use dynamic shapes
    during tracing. If this occurs, consider using the TensorFlow-based export as a fallback.
    
    Parameters:
    -----------
    params : tuple
        Policy parameters containing normalization stats and model weights.
        Structure: (normalization_params, model_params)
        - normalization_params: Contains mean/std for state normalization
        - model_params: JAX model parameters for the policy network
    act_size : int
        Action size (number of action dimensions)
    ppo_params : object
        PPO configuration parameters containing network factory settings
    obs_size : int
        Observation size (number of observation dimensions)
    output_path : str, optional
        Path to save the ONNX model file (default: "open_duck_mini_v2_jax.onnx")
    
    Returns:
    --------
    None
        The ONNX model is saved to output_path.
    
    Requirements:
    -------------
    - Opset 11 (Isaac Lab compatibility)
    - Input name: "obs"
    - Output name: "continuous_actions"
    - Batch size: (1, obs_size) for MuJoCo
    """
    if obs_size <= 0 or act_size <= 0:
        raise ValueError(f"obs_size and act_size must be positive, got obs_size={obs_size} and act_size={act_size}")
    
    print(" === Exporting to ONNX directly from JAX === ")
    
    # Suppress jax2onnx warnings about float64 truncation to float32
    # jax2onnx internally uses float64 in some operations, but JAX defaults to float32.
    # These warnings are excessive and don't affect functionality.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Explicitly requested dtype float64.*",
            category=UserWarning,
            module="jax2onnx.*"
        )
        
        # Create JAX inference function with normalization and output processing
        jax_inference_fn = make_jax_inference_fn(params, act_size, ppo_params, obs_size)
        
        # Pre-compile the function with JIT to ensure it's fully traceable
        # This helps catch any tracing issues before jax2onnx tries to convert
        # We warm up the function with a concrete input to ensure all shapes are resolved
        input_spec = jax.ShapeDtypeStruct(shape=(1, obs_size), dtype=jnp.float32)
        concrete_input = jnp.zeros((1, obs_size), dtype=jnp.float32)
        
        # Warm up the function to ensure it's fully compiled
        try:
            _ = jax_inference_fn(concrete_input)
            # Also try JIT compilation to ensure traceability
            jax_inference_fn_jit = jax.jit(jax_inference_fn)
            _ = jax_inference_fn_jit(concrete_input)
        except Exception as warmup_error:
            raise ValueError(
                f"Function warmup failed, cannot proceed with ONNX export: {warmup_error}"
            ) from warmup_error
        
        # Convert to ONNX using jax2onnx
        # Try with the original function first, then with JIT if needed
        try:
            model_proto = to_onnx(
                jax_inference_fn,
                inputs=[input_spec],
                opset=11,
                model_name="open_duck_policy"
            )
        except Exception as e:
            error_msg = str(e)
            # If tracing fails, try with JIT compiled version
            if "JitTracer" in error_msg or "concrete value" in error_msg.lower():
                print("Initial conversion failed, trying with JIT-compiled function...")
                try:
                    model_proto = to_onnx(
                        jax_inference_fn_jit,
                        inputs=[input_spec],
                        opset=11,
                        model_name="open_duck_policy"
                    )
                except Exception as e2:
                    # If JIT version also fails, provide helpful error
                    raise ValueError(
                        f"Failed to convert to ONNX even with JIT compilation: {e2}\n"
                        "This error typically occurs when jax2onnx encounters dynamic shapes "
                        "in Flax network operations during tracing. The function executes correctly, "
                        "but jax2onnx's tracing mechanism cannot handle certain Flax operations.\n"
                        "Consider using the TensorFlow-based ONNX export as a fallback "
                        "(disable --use_jax_to_onnx flag)."
                    ) from e2
            else:
                raise ValueError(f"Failed to convert to ONNX: {e}") from e
        
        # Set input and output names on the ONNX model proto
        # jax2onnx may generate generic names, so we set them explicitly
        if len(model_proto.graph.input) > 0:
            # Find the actual input name used in the graph nodes
            old_input_name = model_proto.graph.input[0].name
            actual_input_name = None
            # Check what the first node(s) use as input
            for node in model_proto.graph.node:
                for inp in node.input:
                    # If input is not from a previous node and not an initializer, it's the graph input
                    if (inp not in [n_out for n in model_proto.graph.node for n_out in n.output] and
                        inp not in [init.name for init in model_proto.graph.initializer]):
                        actual_input_name = inp
                        break
                if actual_input_name:
                    break
            
            # Use the actual input name found, or fall back to graph input name
            input_name_to_replace = actual_input_name if actual_input_name else old_input_name
            
            # Rename the graph input
            model_proto.graph.input[0].name = "obs"
            # Update all node references to use "obs"
            for node in model_proto.graph.node:
                for i, inp in enumerate(node.input):
                    if inp == input_name_to_replace:
                        node.input[i] = "obs"
        
        if len(model_proto.graph.output) > 0:
            # Get the actual output name from the last node
            if len(model_proto.graph.node) > 0:
                last_node = model_proto.graph.node[-1]
                if len(last_node.output) > 0:
                    actual_output_name = last_node.output[0]
                    # Rename the last node's output to "continuous_actions"
                    last_node.output[0] = "continuous_actions"
                    # Update the graph output name
                    model_proto.graph.output[0].name = "continuous_actions"
            else:
                model_proto.graph.output[0].name = "continuous_actions"

    # Save ONNX model
    # Create the directory if it doesn't exist before saving the model
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
    except Exception as e:
        raise ValueError(f"Failed to save ONNX model: {e}") from e

    # Generate and save metadata (always in local format)
    try:
        _save_metadata(
            output_path, params, act_size, obs_size, ppo_params,
            checkpoint_path, training_step, reward, reward_std
        )
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}")
        print("ONNX model was saved successfully, but metadata generation was skipped.")

    return output_path

def extract_norm_params(params) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract normalization parameters from Brax PPO checkpoint.
    
    Parameters:
    -----------
    params : tuple or list
        Policy parameters tuple/list of length 2 or 3: 
        - (normalization_params, model_params) for length 2
        - (normalization_params, policy_params, value_params) for length 3
    
    Returns:
    --------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing (mean, std) for state normalization
    
    Raises:
    ------
    ValueError
        If params has fewer than 2 elements
    TypeError
        If normalization parameters cannot be extracted
    """
    if len(params) < 2:
        raise ValueError(f"params must have at least 2 elements, got {len(params)}")
    
    norm_params = params[0]
    try:
        # Handle both dict and object with attributes
        if isinstance(norm_params, dict):
            mean: jnp.ndarray = norm_params["mean"]["state"]
        else:
            mean: jnp.ndarray = norm_params.mean["state"]
    except (AttributeError, KeyError, TypeError) as e:
        raise TypeError(f"Failed to extract mean parameters: {e}") from e
    try:
        # Handle both dict and object with attributes
        if isinstance(norm_params, dict):
            std: jnp.ndarray = norm_params["std"]["state"]
        else:
            std: jnp.ndarray = norm_params.std["state"]
    except (AttributeError, KeyError, TypeError) as e:
        raise TypeError(f"Failed to extract std parameters: {e}") from e
    return mean, std


def extract_model_params(model_params_obj) -> dict:
    """
    Extract and validate model parameters from Brax PPO checkpoint.
    
    Parameters:
    -----------
    model_params_obj : dict
        Model parameters from params[1]. Can be either:
        - A dictionary with nested 'params' containing raw network layers (e.g., {'params': {'hidden_0': ...}})
        - A dictionary with PPO structure (policy, value, aux, etc.)
    
    Returns:
    --------
    dict
        Model parameters dictionary with 'policy' key containing the policy network parameters
    
    Raises:
    ------
    TypeError
        If model_params_obj is neither a dict nor has expected attributes
    KeyError
        If 'policy' key is missing from model_params
    """
    # Handle dict input
    if isinstance(model_params_obj, dict):
        # Check for nested 'params' with raw network layers (e.g., {'params': {'hidden_0': ...}})
        # This is the structure from Brax checkpoints: params[1] = {'params': {'hidden_0': ..., 'hidden_1': ...}}
        if 'params' in model_params_obj and isinstance(model_params_obj['params'], dict):
            nested = model_params_obj['params']
            # Check if nested dict has 'policy' key (PPO structure)
            if 'policy' in nested:
                # Already has PPO structure
                model_params = nested
            else:
                # Raw network layers - wrap in PPO structure
                # The 'params' dict contains the actual network layers (hidden_0, hidden_1, etc.)
                model_params = {'policy': {'params': nested}}
        elif 'policy' in model_params_obj:
            # Already has 'policy' key at top level
            model_params = model_params_obj
        else:
            # No 'params' or 'policy' key - assume it's the policy params directly
            # Wrap in PPO structure
            model_params = {'policy': {'params': model_params_obj}}
    else:
        raise TypeError(
            f"model_params_obj must be a dict, got {type(model_params_obj)}"
        )
    
    # Validate structure
    if not isinstance(model_params, dict):
        raise TypeError(f"Model parameters must be a dictionary, got {type(model_params)}")
    
    if 'policy' not in model_params:
        raise KeyError(f"Model parameters must contain 'policy' key, got keys: {list(model_params.keys())}")
    
    return model_params


def _has_observation_keyword(param_name: str) -> bool:
    """Check if parameter name contains observation-related keywords."""
    param_lower = param_name.lower()
    return 'obs' in param_lower or 'observation' in param_lower


def _has_action_keyword(param_name: str) -> bool:
    """Check if parameter name contains action-related keywords."""
    param_lower = param_name.lower()
    return 'act' in param_lower or 'action' in param_lower


def _inspect_network_factory_signature(network_factory, obs_size: int, act_size: int) -> tuple[list[str], inspect.Signature]:
    """
    Inspect network_factory signature and extract parameter names.
    
    Parameters:
    -----------
    network_factory : callable
        Network factory function (make_ppo_networks or partial)
    obs_size : int
        Observation size (for error messages)
    act_size : int
        Action size (for error messages)
    
    Returns:
    --------
    tuple[list[str], inspect.Signature]
        Tuple of (parameter_names, signature)
    
    Raises:
    ------
    ValueError
        If the signature cannot be determined or is invalid
    """
    
    try:
        sig = inspect.signature(network_factory)
        param_names = list(sig.parameters.keys())
    except (AttributeError, ValueError) as e:
        raise ValueError(
            f"Could not inspect network_factory signature: {e}. "
            f"Brax's make_ppo_networks typically uses (observation_size, action_size). "
            f"obs_size={obs_size}, act_size={act_size}"
        ) from e
    
    if len(param_names) < 2:
        raise ValueError(
            f"network_factory signature has fewer than 2 parameters. "
            f"Expected at least (observation_size, action_size). "
            f"Found parameters: {param_names}"
        )
    
    return param_names, sig


def verify_network_factory_signature(network_factory, obs_size: int, act_size: int) -> None:
    """
    Verify that network_factory signature can be determined and is valid.
    
    Parameters:
    -----------
    network_factory : callable
        Network factory function (make_ppo_networks or partial)
    obs_size : int
        Observation size
    act_size : int
        Action size
    
    Raises:
    ------
    ValueError
        If the signature cannot be determined or is invalid
    """
    param_names, _ = _inspect_network_factory_signature(network_factory, obs_size, act_size)
    
    first_param = param_names[0]
    second_param = param_names[1]
    
    # Verify we can determine order from parameter names
    can_determine_order = (
        _has_observation_keyword(first_param) or
        _has_action_keyword(first_param) or
        _has_observation_keyword(second_param) or
        _has_action_keyword(second_param)
    )
    
    if not can_determine_order:
        raise ValueError(
            f"Could not determine argument order from parameter names. "
            f"First parameter: '{first_param}', Second parameter: '{second_param}'. "
            f"Brax's make_ppo_networks typically uses (observation_size, action_size). "
            f"obs_size={obs_size}, act_size={act_size}"
        )


def get_network_factory_argument_order(network_factory, obs_size: int, act_size: int) -> tuple[int, int]:
    """
    Get the correct argument order for network_factory.
    
    Parameters:
    -----------
    network_factory : callable
        Network factory function (make_ppo_networks or partial)
    obs_size : int
        Observation size
    act_size : int
        Action size
    
    Returns:
    --------
    tuple[int, int]
        Correct argument order (first_arg, second_arg)
    
    Raises:
    ------
    ValueError
        If the signature cannot be determined
    """
    param_names, _ = _inspect_network_factory_signature(network_factory, obs_size, act_size)
    
    first_param = param_names[0]
    second_param = param_names[1]
    
    # Determine order based on parameter names
    # If first param has observation keyword, order is (obs_size, act_size)
    # If first param has action keyword, order is (act_size, obs_size)
    # If second param has observation keyword, first must be action, so (act_size, obs_size)
    # If second param has action keyword, first must be observation, so (obs_size, act_size)
    if _has_observation_keyword(first_param) or _has_action_keyword(second_param):
        return (obs_size, act_size)
    if _has_action_keyword(first_param) or _has_observation_keyword(second_param):
        return (act_size, obs_size)
    
    raise ValueError(
        f"Could not determine argument order from parameter names. "
        f"First parameter: '{first_param}', Second parameter: '{second_param}'"
    )


def verify_policy_network_signature(policy_network) -> None:
    """
    Verify the argument signature of the policy network apply function.
    
    Parameters:
    -----------
    policy_network : object
        Policy network object with an 'apply' method
    
    Raises:
    ------
    AttributeError
        If policy_network doesn't have an 'apply' method
    
    Note:
    -----
    This is a lenient check - we only verify the apply method exists.
    The actual signature may vary and we handle it in the inference function.
    """
    if not hasattr(policy_network, 'apply'):
        raise AttributeError("policy_network must have an 'apply' method")
    
    # Optional: Try to inspect signature if available, but don't fail if not
    if hasattr(policy_network.apply, 'signature'):
        try:
            signature_params = policy_network.apply.signature.parameters
            # Just log for debugging, don't enforce strict requirements
            print(f"DEBUG: Policy network apply signature has parameters: {list(signature_params.keys())[:5]}...")
        except Exception:
            pass  # Signature inspection failed, but that's okay


def make_jax_inference_fn(params, act_size, ppo_params, obs_size) -> Callable:
    """
    Make a JAX inference function for the policy with normalization and output processing.
    
    This function creates a complete inference pipeline that:
    1. Normalizes observations using mean/std from params[0]
    2. Runs the policy network with params[1]
    3. Processes output (split + tanh) to match TensorFlow version
    
    Parameters:
    -----------
    params : tuple
        Policy parameters: (normalization_params, model_params)
        - normalization_params: Contains mean/std for state normalization
        - model_params: JAX model parameters for the policy network
    act_size : int
        Action size (number of action dimensions)
    ppo_params : object
        PPO configuration parameters containing network factory settings
    obs_size : int
        Observation size (number of observation dimensions)
    
    Returns:
    --------
    Callable
        JAX inference function that takes obs and returns actions
    """
    # Handle both 2-element and 3-element params tuples/lists
    # Brax checkpoints can have: (norm_params, policy_params, value_params)
    # We only need the first 2 elements (norm_params and policy_params)
    if len(params) < 2:
        raise ValueError(f"params must have at least 2 elements, got {len(params)}")
    if len(params) > 2:
        # Extract first 2 elements (discard value network params if present)
        params = params[:2]

    # Extract normalization parameters
    mean, std = extract_norm_params(params)

    # Extract model parameters
    model_params = extract_model_params(params[1])

    # Recreate the network structure using the same factory as training
    # PPO parameters contains configuration and hyperparameters for the network
    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
    else:
        network_factory = ppo_networks.make_ppo_networks
    
    # Create networks (we only need the policy network) and ignore the value network
    # Verify signature can be determined, then get correct argument order
    verify_network_factory_signature(network_factory, obs_size, act_size)
    arg1, arg2 = get_network_factory_argument_order(network_factory, obs_size, act_size)
    networks = network_factory(arg1, arg2)
    # PPONetworks is an object with policy_network and value_network attributes
    policy_network = networks.policy_network

    # Verify the argument order of the policy network apply function
    verify_policy_network_signature(policy_network)
    
    # Extract policy parameters from model_params
    # model_params structure from extract_model_params: {'policy': {'params': {...}}}
    # where {...} contains the actual network layers (hidden_0, hidden_1, etc.)
    policy_params = model_params.get('policy')
    if policy_params is None:
        raise KeyError("model_params must contain 'policy' key with policy network parameters")
    
    # Brax policy network apply signature is: (processor_params, policy_params, obs)
    # processor_params: normalization/preprocessing params (empty since we normalize manually)
    # policy_params: the actual network parameters dict with layers (hidden_0, hidden_1, etc.)
    if isinstance(policy_params, dict):
        processor_params = policy_params.get('processor', {})
        if 'params' in policy_params:
            # Extract the actual network parameters (dict with hidden_0, hidden_1, etc.)
            actual_policy_params = policy_params['params']
        else:
            # If no 'params' key, the whole dict might be the policy params directly
            actual_policy_params = policy_params
    else:
        processor_params = {}
        actual_policy_params = policy_params
    
    # Validate that actual_policy_params has the expected structure (dict with layer names)
    if not isinstance(actual_policy_params, dict):
        raise TypeError(
            f"Policy parameters must be a dict with layer names, got {type(actual_policy_params)}"
        )
    if len(actual_policy_params) == 0:
        raise ValueError("Policy parameters dict is empty")

    # Create inference function with normalization and output processing
    # Brax policy network apply signature: (processor_params, policy_params, obs)
    # We do normalization manually, so processor_params can be empty
    # Ensure mean/std are float32 to match input dtype (even when x64 is enabled)
    mean_f32 = jnp.asarray(mean, dtype=jnp.float32)
    std_f32 = jnp.asarray(std, dtype=jnp.float32)
    
    # Convert act_size to concrete int to avoid tracing issues with jax2onnx
    act_size_concrete = int(act_size)
    
    def jax_inference_fn(obs):
        """
        JAX inference function that:
        1. Normalizes observations: (obs - mean) / std
        2. Runs policy network forward pass
        3. Splits output and applies tanh: tanh(split(logits)[0])
        """
        # Normalize observations (matching TensorFlow version)
        # Use float32 mean/std to match input dtype
        normalized_obs = (obs - mean_f32) / std_f32
        
        # Call apply with correct signature: (processor_params, policy_params, obs)
        # Flax networks expect parameters wrapped in {'params': {...}} structure
        # processor_params: empty dict since we normalize manually
        # policy_params: must be wrapped as {'params': actual_policy_params} for Flax
        logits = policy_network.apply(
            processor_params,
            {'params': actual_policy_params},
            normalized_obs
        )
        
        # Split logits into loc (mean action, location parameter) and log_std (discarded)
        # Use concrete act_size to avoid dynamic shape issues during jax2onnx tracing
        # The logits shape should be (batch, act_size * 2), so we split at act_size
        loc = logits[..., :act_size_concrete]
        return jnp.tanh(loc)
        
    return jax_inference_fn


def _generate_example_data(jax_inference_fn, obs_size, num_examples=5):
    """Generate example observations and actions for metadata.
    
    Parameters:
    -----------
    jax_inference_fn : Callable
        JAX inference function
    obs_size : int
        Observation size
    num_examples : int
        Number of examples to generate (default: 5)
    
    Returns:
    --------
    dict
        Dictionary with 'example_observations' and 'example_actions' lists
    """
    examples = []
    
    # Generate different types of inputs
    test_inputs = [
        np.random.randn(1, obs_size).astype(np.float32),  # Random
        np.zeros((1, obs_size), dtype=np.float32),  # Zero
        np.ones((1, obs_size), dtype=np.float32),  # Unit
        np.random.randn(1, obs_size).astype(np.float32) * 2.0,  # Large values
        np.random.randn(1, obs_size).astype(np.float32) * 0.5,  # Small values
    ]
    
    for i, test_input in enumerate(test_inputs[:num_examples]):
        obs_jax = jnp.array(test_input)
        action_jax = jax_inference_fn(obs_jax)
        
        examples.append({
            "observation": test_input.tolist()[0],  # Remove batch dimension
            "action": np.array(action_jax).tolist()[0]  # Remove batch dimension
        })
    
    return examples


def _save_metadata(
    onnx_path, params, act_size, obs_size, ppo_params,
    checkpoint_path, training_step, reward, reward_std
):
    """Save metadata for ONNX model in local format.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX file
    params : tuple
        Policy parameters
    act_size : int
        Action size
    obs_size : int
        Observation size
    ppo_params : object
        PPO configuration
    checkpoint_path : str, optional
        Path to checkpoint directory
    training_step : int, optional
        Training step number
    reward : float, optional
        Training reward
    reward_std : float, optional
        Training reward std
    """
    # Extract normalization params
    mean, std = extract_norm_params(params)
    
    # Generate example data
    jax_inference_fn = make_jax_inference_fn(params, act_size, ppo_params, obs_size)
    examples = _generate_example_data(jax_inference_fn, obs_size, num_examples=5)
    
    # Build metadata dict
    metadata = {
        "model_type": "ppo_policy",
        "obs_size": int(obs_size),
        "act_size": int(act_size),
        "opset_version": 11,
        "framework": "onnx",
        "source_framework": "jax",
        "conversion_tool": "jax2onnx",
        "original_architecture": "brax_ppo",
        "conversion_reason": "RTX 5090 CUDA compatibility",
        "training_repo": "https://github.com/apirrone/Open_Duck_Playground",
        "license": "apache-2.0",
        "normalization": {
            "mean": np.array(mean).tolist(),
            "std": np.array(std).tolist()
        },
        "example_observations": [ex["observation"] for ex in examples],
        "example_actions": [ex["action"] for ex in examples],
        "export_timestamp": datetime.now().isoformat()
    }
    
    if checkpoint_path:
        metadata["checkpoint_path"] = str(checkpoint_path)
    if training_step is not None:
        metadata["training_step"] = int(training_step)
    if reward is not None:
        metadata["reward"] = float(reward)
    if reward_std is not None:
        metadata["reward_std"] = float(reward_std)
    
    # Save metadata in local format: {onnx}.metadata.json
    metadata_path = f"{onnx_path}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

