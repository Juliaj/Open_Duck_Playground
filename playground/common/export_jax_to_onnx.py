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

import jax.numpy as jnp
# import RL policies from Brax. Brax is also a physics engine for RL but not used for this project.
from brax.training.agents.ppo import networks as ppo_networks
from jax2onnx import to_onnx


def export_onnx_jax(
    params, act_size, ppo_params, obs_size, output_path="open_duck_mini_v2_jax.onnx"
)->str:
    """
    Export JAX policy to ONNX format using jax2onnx (direct conversion).
    
    This function directly converts JAX inference function to ONNX without
    going through TensorFlow, avoiding CUDA compatibility issues on new GPUs like the RTX 5090.
    
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
    
    # Create JAX inference function with normalization and output processing
    jax_inference_fn = make_jax_inference_fn(params, act_size, ppo_params, obs_size)
    
    # Convert to ONNX using jax2onnx
    input_shapes = {"obs": (1, obs_size)}
    try:
        model_proto = to_onnx(
            jax_inference_fn,
            input_shapes=input_shapes,
            opset=11,
            output_names=["continuous_actions"]
        )
    except Exception as e:
        raise ValueError(f"Failed to convert to ONNX: {e}") from e

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

    return output_path

def extract_norm_params(params) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract normalization parameters from Brax PPO checkpoint.
    
    Parameters:
    -----------
    params : tuple
        Policy parameters tuple of length 2: (normalization_params, model_params)
    
    Returns:
    --------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing (mean, std) for state normalization
    
    Raises:
    ------
    ValueError
        If params is not a tuple of length 2
    TypeError
        If normalization parameters cannot be extracted
    """
    if len(params) != 2:
        raise ValueError(f"params must be a tuple of length 2, got {len(params)}")
    
    norm_params = params[0]
    try:
        mean: jnp.ndarray = norm_params.mean["state"]
    except AttributeError as e:
        raise TypeError(f"Failed to extract mean parameters: {e}") from e
    try:
        std: jnp.ndarray = norm_params.std["state"]
    except AttributeError as e:
        raise TypeError(f"Failed to extract std parameters: {e}") from e
    return mean, std


def extract_model_params(model_params_obj) -> dict:
    """
    Extract and validate model parameters from Brax PPO checkpoint.
    
    Parameters:
    -----------
    model_params_obj : dict or object
        Model parameters from params[1]. Can be either:
        - A dictionary directly containing the parameters
        - An object with a 'policy' attribute containing parameters
    
    Returns:
    --------
    dict
        Validated model parameters dictionary
    
    Raises:
    ------
    TypeError
        If model_params_obj is neither a dict nor has expected attributes
    KeyError
        If required keys are missing from model_params
    """
    try:
        if isinstance(model_params_obj, dict):
            model_params: dict = model_params_obj
        else:
            if not hasattr(model_params_obj, 'policy'):
                raise TypeError(
                    f"model_params_obj must be a dict or have 'policy' attribute, "
                    f"got {type(model_params_obj)}"
                )
            model_params: dict = model_params_obj.policy.get('params')
            if model_params is None:
                raise KeyError("model_params_obj.policy.get('params') returned None")
    except AttributeError as e:
        raise TypeError(f"Failed to extract model parameters: {e}") from e
    
    if not isinstance(model_params, dict):
        raise TypeError(f"Model parameters must be a dictionary, got {type(model_params)}")
    
    required_keys = ["policy", "value", "aux", "policy_aux", "value_aux", "aux_aux"]
    missing_keys = [key for key in required_keys if key not in model_params]
    if missing_keys:
        raise KeyError(f"Model parameters missing required keys: {missing_keys}")
    
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
        If policy_network doesn't have an 'apply' method or signature
    KeyError
        If required parameters are missing from the signature
    
    Expected parameters:
    - params: model parameters
    - obs: observations
    - deterministic: whether to use deterministic action
    - rng: random number generator
    - step_type: step type
    - episode_length: episode length
    - episode_return: episode return
    
    Note:
    -----
    This verification ensures the network signature matches expectations, but the
    actual apply call may need to handle optional/default parameters.
    """
    if not hasattr(policy_network, 'apply'):
        raise AttributeError("policy_network must have an 'apply' method")
    
    if not hasattr(policy_network.apply, 'signature'):
        raise AttributeError("policy_network.apply must have a 'signature' attribute")
    
    # Access signature.parameters safely after verifying signature exists
    signature_params = policy_network.apply.signature.parameters
    required_params = ['params', 'obs', 'deterministic', 'rng', 'step_type', 'episode_length', 'episode_return']
    
    missing_params = [param for param in required_params if param not in signature_params]
    if missing_params:
        raise KeyError(
            f"Policy network apply signature missing required parameters: {missing_params}. "
            f"Found parameters: {list(signature_params.keys())}"
        )


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
    if len(params) != 2:
        raise ValueError(f"params must be a tuple of length 2, got {len(params)}")

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
    policy_network, _ = network_factory(arg1, arg2)

    # Verify the argument order of the policy network apply function
    verify_policy_network_signature(policy_network)
    
    # Extract policy parameters from model_params (similar to TensorFlow version)
    policy_params = model_params.get('policy')
    if policy_params is None:
        raise KeyError("model_params must contain 'policy' key with policy network parameters")

    # Create inference function with normalization and output processing
    # For inference, we provide:
    # - params, obs (required positional args)
    # - deterministic=True (for deterministic inference)
    # - Other params (rng, step_type, etc.) use their defaults if they have them
    def jax_inference_fn(obs):
        """
        JAX inference function that:
        1. Normalizes observations: (obs - mean) / std
        2. Runs policy network forward pass
        3. Splits output and applies tanh: tanh(split(logits)[0])
        """
        # Normalize observations (matching TensorFlow version)
        normalized_obs = (obs - mean) / std
        
        # Call apply with required params and deterministic=True
        # Other parameters (rng, step_type, episode_length, episode_return) 
        # will use their default values from the signature
        logits = policy_network.apply(
            policy_params,
            normalized_obs,
            deterministic=True
        )
        
        # Split logits into loc (mean action, location parameter) and log_std (discarded)
        loc, _ = jnp.split(logits, 2, axis=-1)
        return jnp.tanh(loc)
        
    return jax_inference_fn