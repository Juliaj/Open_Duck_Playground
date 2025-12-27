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
Validate JAX model directly in MuJoCo simulation (bypassing ONNX).

Tests if the JAX model can walk forward without falling.
This helps determine if issues are from ONNX export or the model itself.

Usage:
    uv run python tests/validate_jax_simulation.py --checkpoint checkpoints/2025_12_26_165635_300482560
"""

import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from etils import epath
from orbax import checkpoint as ocp

from playground.common.export_jax_to_onnx import make_jax_inference_fn
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2 import base
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from mujoco_playground.config import locomotion_params


class ForwardWalkCommand:
    """Command sequence for forward walking."""
    
    def __init__(self, linear_vel_x=0.15):
        self.linear_vel_x = linear_vel_x
    
    def get_command(self, step):
        return [self.linear_vel_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class HeadlessSimulation(MJInferBase):
    """Headless MuJoCo simulation using JAX inference."""
    
    def __init__(self, model_path, reference_data, jax_inference_fn, standing=False):
        super().__init__(model_path)
        
        self.standing = standing
        self.jax_inference_fn = jax_inference_fn
        
        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)
            self.imitation_i = 0
        else:
            self.PRM = None
            self.imitation_i = 0
        
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        
        self.max_motor_velocity = 5.24
        self.action_scale = 0.25
        self.dof_vel_scale = 0.05
        
        self.phase_frequency_factor = 1.0
        self.imitation_phase = np.array([0, 0])
    
    def get_obs(self, data, command):
        """Get observation from simulation data."""
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3
        
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        
        contacts = self.get_feet_contacts(data)
        contacts_array = np.array([float(contacts[0]), float(contacts[1])], dtype=np.float32)
        
        if not self.standing and self.PRM:
            self.imitation_i += 1.0 * self.phase_frequency_factor
            self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
            self.imitation_phase = np.array([
                np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
            ])
        
        # Ensure all arrays are 1D
        obs = np.concatenate([
            gyro.flatten(),
            accelerometer.flatten(),
            np.array(command).flatten(),
            (joint_angles - self.default_actuator).flatten(),
            (joint_vel * self.dof_vel_scale).flatten(),
            self.last_action.flatten(),
            self.last_last_action.flatten(),
            self.last_last_last_action.flatten(),
            self.motor_targets.flatten(),
            contacts_array.flatten(),
            self.imitation_phase.flatten(),
        ])
        
        return obs
    
    def run_headless(self, command_sequence, duration_seconds=120, fall_height_threshold=0.3, fall_duration_steps=500, use_viewer=False):
        """Run simulation with JAX inference (headless or with viewer).
        
        Parameters:
        -----------
        command_sequence : ForwardWalkCommand
            Command sequence to execute
        duration_seconds : float
            Maximum simulation duration in seconds
        fall_height_threshold : float
            Body height below which is considered "laying flat" (meters)
        fall_duration_steps : int
            Number of consecutive steps below threshold to consider "fallen"
        use_viewer : bool
            If True, show MuJoCo viewer for visual inspection
        """
        max_steps = int(duration_seconds / self.sim_dt)
        step = 0
        counter = 0
        
        start_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        start_z = self.data.qpos[2] if len(self.data.qpos) > 2 else 0.0
        body_heights = []
        fall_steps_below_threshold = 0
        total_steps_below_threshold = 0  # Track total time below, not just consecutive
        fall_detected = False
        fall_step = None
        
        if use_viewer:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                while step < max_steps:
                    step_start = time.time()
                    mujoco.mj_step(self.model, self.data)
                    step += 1
                    counter += 1
                    
                    if counter % self.decimation == 0:
                        command = command_sequence.get_command(step)
                        obs = self.get_obs(self.data, command)
                        
                        import jax.numpy as jnp
                        action = np.array(self.jax_inference_fn(jnp.array(obs.reshape(1, -1))))
                        
                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()
                        
                        self.motor_targets = self.default_actuator + action * self.action_scale
                        
                        self.motor_targets = np.clip(
                            self.motor_targets,
                            self.prev_motor_targets - self.max_motor_velocity * (self.sim_dt * self.decimation),
                            self.prev_motor_targets + self.max_motor_velocity * (self.sim_dt * self.decimation),
                        )
                        self.prev_motor_targets = self.motor_targets.copy()
                        
                        self.data.ctrl = self.motor_targets.copy()
                    
                    current_z = self.data.qpos[2] if len(self.data.qpos) > 2 else 0.0
                    body_heights.append(current_z)
                    
                    if current_z < fall_height_threshold:
                        fall_steps_below_threshold += 1
                        total_steps_below_threshold += 1
                        if fall_steps_below_threshold >= fall_duration_steps and not fall_detected:
                            fall_detected = True
                            fall_step = step
                            print(f"\nFall detected at step {step} (height: {current_z:.3f}m)")
                            break
                    else:
                        fall_steps_below_threshold = 0
                    
                    viewer.sync()
                    
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while step < max_steps:
                mujoco.mj_step(self.model, self.data)
                step += 1
                counter += 1
                
                if counter % self.decimation == 0:
                    command = command_sequence.get_command(step)
                    obs = self.get_obs(self.data, command)
                    
                    import jax.numpy as jnp
                    action = np.array(self.jax_inference_fn(jnp.array(obs.reshape(1, -1))))
                    
                    self.last_last_last_action = self.last_last_action.copy()
                    self.last_last_action = self.last_action.copy()
                    self.last_action = action.copy()
                    
                    self.motor_targets = self.default_actuator + action * self.action_scale
                    
                    self.motor_targets = np.clip(
                        self.motor_targets,
                        self.prev_motor_targets - self.max_motor_velocity * (self.sim_dt * self.decimation),
                        self.prev_motor_targets + self.max_motor_velocity * (self.sim_dt * self.decimation),
                    )
                    self.prev_motor_targets = self.motor_targets.copy()
                    
                    self.data.ctrl = self.motor_targets.copy()
                
                current_z = self.data.qpos[2] if len(self.data.qpos) > 2 else 0.0
                body_heights.append(current_z)
                
                if current_z < fall_height_threshold:
                    fall_steps_below_threshold += 1
                    total_steps_below_threshold += 1
                    if fall_steps_below_threshold >= fall_duration_steps and not fall_detected:
                        fall_detected = True
                        fall_step = step
                        break
                else:
                    fall_steps_below_threshold = 0
        
        end_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        forward_distance = end_x - start_x
        episode_length = step
        stability_score = 1.0 if not fall_detected else (fall_step / max_steps) if fall_step else 0.0
        
        passed = not fall_detected and forward_distance > 0.1
        fraction_below_threshold = total_steps_below_threshold / episode_length if episode_length > 0 else 0.0
        
        return {
            "status": "PASS" if passed else "FAIL",
            "fall_detected": fall_detected,
            "fall_step": fall_step,
            "stability_score": float(stability_score),
            "forward_distance": float(forward_distance),
            "episode_length": int(episode_length),
            "max_steps": max_steps,
            "final_body_height": float(body_heights[-1]) if body_heights else 0.0,
            "total_steps_below_threshold": int(total_steps_below_threshold),
            "fraction_below_threshold": float(fraction_below_threshold),
        }


def validate_jax_model(checkpoint_path, duration_seconds=120, use_viewer=False, fall_height_threshold=0.3, fall_duration_steps=500):
    """Validate JAX model directly in simulation."""
    # Convert to absolute path (required by orbax)
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        return {
            "status": "FAIL",
            "error": f"Checkpoint not found: {checkpoint_path}",
            "checkpoint_path": checkpoint_path
        }
    
    try:
        # Load checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(checkpoint_path)
        if isinstance(params, list):
            params = tuple(params)
        
        # Get config and detect sizes
        ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
        
        # Detect obs_size and act_size from checkpoint
        from playground.common.export_jax_to_onnx import extract_norm_params
        mean, std = extract_norm_params(params)
        obs_size = mean.shape[0] if len(mean.shape) == 1 else mean.shape[-1]
        
        # Detect act_size
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
                        return {"status": "FAIL", "error": "Could not detect act_size"}
                else:
                    return {"status": "FAIL", "error": "Could not detect act_size"}
            else:
                return {"status": "FAIL", "error": "Could not detect act_size"}
        else:
            return {"status": "FAIL", "error": "Could not detect act_size"}
        
        # Create JAX inference function
        jax_inference_fn = make_jax_inference_fn(params, act_size, ppo_params, obs_size)
        
        # Initialize simulation
        model_path = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        reference_data = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
        
        sim = HeadlessSimulation(
            model_path=model_path,
            reference_data=reference_data,
            jax_inference_fn=jax_inference_fn,
            standing=False
        )
        
        # Create forward walk command
        command_seq = ForwardWalkCommand(linear_vel_x=0.15)
        
        # Run simulation
        result = sim.run_headless(
            command_seq, 
            duration_seconds=duration_seconds,
            use_viewer=use_viewer,
            fall_height_threshold=fall_height_threshold,
            fall_duration_steps=fall_duration_steps
        )
        result["checkpoint_path"] = checkpoint_path
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "FAIL",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "checkpoint_path": checkpoint_path
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate JAX model directly in MuJoCo simulation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Maximum simulation duration in seconds (default: 120)"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show MuJoCo viewer for visual inspection"
    )
    parser.add_argument(
        "--fall-height-threshold",
        type=float,
        default=0.3,
        help="Body height threshold for fall detection in meters (default: 0.3)"
    )
    parser.add_argument(
        "--fall-duration-steps",
        type=int,
        default=500,
        help="Number of consecutive steps below threshold to consider fallen (default: 500)"
    )
    
    args = parser.parse_args()
    
    print(f"Validating JAX model from checkpoint...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Duration: {args.duration} seconds")
    print(f"Viewer: {'Enabled' if args.viewer else 'Disabled'}")
    print(f"Fall threshold: {args.fall_height_threshold}m for {args.fall_duration_steps} steps")
    print()
    
    result = validate_jax_model(
        args.checkpoint, 
        args.duration,
        use_viewer=args.viewer,
        fall_height_threshold=args.fall_height_threshold,
        fall_duration_steps=args.fall_duration_steps
    )
    
    status = result["status"]
    checkpoint_name = os.path.basename(args.checkpoint)
    
    if status == "PASS":
        print(f"✓ {checkpoint_name}")
        print(f"  Stability: {result['stability_score']:.2f}, Forward: {result['forward_distance']:.2f}m, Steps: {result['episode_length']}")
    else:
        print(f"✗ {checkpoint_name}")
        if "error" in result:
            print(f"  Error: {result['error']}")
            if "traceback" in result:
                print(f"  Traceback:\n{result['traceback']}")
        else:
            fall_info = f" (fell at step {result['fall_step']})" if result.get('fall_detected') else ""
            fraction_info = f", Below threshold: {result.get('fraction_below_threshold', 0):.1%}" if 'fraction_below_threshold' in result else ""
            print(f"  Stability: {result['stability_score']:.2f}, Forward: {result['forward_distance']:.2f}m, Steps: {result['episode_length']}{fall_info}{fraction_info}")


if __name__ == "__main__":
    main()

