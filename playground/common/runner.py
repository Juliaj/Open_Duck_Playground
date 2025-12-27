"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
import warnings
import traceback
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.export_onnx import export_onnx
from playground.common.export_jax_to_onnx import export_onnx_jax


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        self.use_jax_to_onnx = getattr(args, 'use_jax_to_onnx', False)
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"
        
        # Set up warning filter to capture JAX overflow warnings with context
        # self._setup_jax_overflow_debugging()

    def _setup_jax_overflow_debugging(self) -> None:
        """Set up debugging for JAX overflow warnings."""
        def jax_overflow_warning_handler(message, category, filename, lineno, file=None, line=None):
            """Custom handler for JAX overflow warnings."""
            if "overflow encountered in cast" in str(message):
                # Get stack trace to see where the warning originated
                stack = traceback.extract_stack()
                # Find the relevant frame (skip internal JAX frames)
                relevant_frames = []
                for frame in reversed(stack[:-5]):  # Skip the warning handler frames
                    if 'jax' not in frame.filename.lower() or 'abstract_arrays' in frame.filename:
                        continue
                    relevant_frames.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
                    if len(relevant_frames) >= 3:  # Show top 3 relevant frames
                        break
                
                print(f"DEBUG JAX Overflow Warning:")
                print(f"  Message: {message}")
                print(f"  Location: {filename}:{lineno}")
                if relevant_frames:
                    print(f"  Call stack (relevant frames):")
                    print("\n".join(relevant_frames))
                print()
            
            # Still show the original warning
            return warnings.formatwarning(message, category, filename, lineno, line)
        
        # Install custom warning handler
        warnings.showwarning = jax_overflow_warning_handler

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        print("-----------")
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )
        print("-----------")

    def _export_onnx_model(self, params, output_path: str) -> None:
        """Export policy to ONNX format.
        
        Args:
            params: Policy parameters from Brax training. May be a tuple of length 2 or 3.
                   For JAX export, extracts first 2 elements: (normalization_params, model_params).
                   For TensorFlow export, passes params as-is (it accesses params[0] and params[1]).
            output_path: Path to save the ONNX model file.
        """
        if self.use_jax_to_onnx:
            print("Using JAX-to-ONNX direct export")
            # JAX export requires exactly 2 elements: (normalization_params, model_params)
            # Brax may pass 3 elements (e.g., including optimizer state), so extract first 2
            if not isinstance(params, tuple):
                raise TypeError(f"params must be a tuple, got {type(params)}")
            
            if len(params) < 2:
                raise ValueError(f"params tuple must have at least 2 elements, got {len(params)}")
            
            if len(params) > 2:
                print(f"DEBUG: params tuple has {len(params)} elements, extracting first 2 (discarding element(s) {list(range(2, len(params)))})")
                params = params[:2]
            
            export_onnx_jax(
                params,
                self.action_size,
                self.ppo_params,
                self.obs_size,
                output_path=output_path
            )
        else:
            print("Using TensorFlow-based ONNX export")
            # TensorFlow export accesses params[0] and params[1] directly, so pass as-is
            export_onnx(
                params,
                self.action_size,
                self.ppo_params,
                self.obs_size,
                output_path=output_path
            )

    def policy_params_fn(self, current_step, make_policy, params):
        """Save checkpoint and export ONNX model.
        
        This callback is invoked during training to save checkpoints
        and export the policy to ONNX format for inference.
        """
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        try:
            self._export_onnx_model(params, onnx_export_path)
            print(f"ONNX model exported to: {onnx_export_path}")
        except Exception as e:
            print(f"Warning: Failed to export ONNX model: {e}")
            print("Training will continue, but ONNX export was skipped.")

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
