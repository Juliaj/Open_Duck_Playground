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
Validate ONNX model in MuJoCo simulation.

Tests if the model can walk forward without falling.

Usage:
    uv run python tests/validate_onnx_simulation.py --onnx checkpoints/model.onnx
    uv run python tests/validate_onnx_simulation.py --checkpoints-dir checkpoints
"""

import argparse
import json
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from etils import epath

from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2 import base
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase


class ForwardWalkCommand:
    """Command sequence for forward walking."""
    
    def __init__(self, linear_vel_x=0.15):
        """Initialize forward walk command.
        
        Parameters:
        -----------
        linear_vel_x : float
            Forward velocity (default: 0.15, max forward)
        """
        self.linear_vel_x = linear_vel_x
    
    def get_command(self, step):
        """Get command for current step.
        
        Parameters:
        -----------
        step : int
            Current simulation step
        
        Returns:
        --------
        list
            Command vector [lin_vel_x, lin_vel_y, ang_vel, neck_pitch, head_pitch, head_yaw, head_roll]
        """
        return [self.linear_vel_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class HeadlessSimulation(MJInferBase):
    """Headless MuJoCo simulation for validation."""
    
    def __init__(
        self,
        model_path,
        reference_data,
        onnx_model_path,
        standing=False
    ):
        """Initialize headless simulation.
        
        Parameters:
        -----------
        model_path : str
            Path to MuJoCo XML model
        reference_data : str
            Path to reference motion data
        onnx_model_path : str
            Path to ONNX model
        standing : bool
            Whether to use standing mode
        """
        # Initialize base class (loads model, sets up joints, etc.)
        super().__init__(model_path)
        
        self.standing = standing
        
        # Load policy
        self.policy = OnnxInfer(onnx_model_path, awd=True)
        
        # Reference motion (if not standing)
        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)
            self.imitation_i = 0
        else:
            self.PRM = None
            self.imitation_i = 0
        
        # Action history
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        
        # Motor limits
        self.max_motor_velocity = 5.24  # rad/s
        self.action_scale = 0.25
        self.dof_vel_scale = 0.05
        
        # Phase
        self.phase_frequency_factor = 1.0
        self.imitation_phase = np.array([0, 0])
    
    def get_obs(self, data, command):
        """Get observation from simulation data.
        
        Reuses logic from mujoco_infer.py
        """
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3
        
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        
        contacts = self.get_feet_contacts(data)
        contacts_array = np.array([float(contacts[0]), float(contacts[1])], dtype=np.float32)
        
        # Update imitation phase if not standing
        if not self.standing and self.PRM:
            self.imitation_i += 1.0 * self.phase_frequency_factor
            self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
            self.imitation_phase = np.array([
                np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
            ])
        
        # Build observation (matching mujoco_infer.py format)
        obs = np.concatenate([
            gyro,
            accelerometer,
            command,
            joint_angles - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            self.motor_targets,
            contacts_array,
            self.imitation_phase,
        ])
        
        return obs
    
    def run_headless(self, command_sequence, duration_seconds=120, fall_height_threshold=0.3, fall_duration_steps=500, use_viewer=False):
        """Run simulation (headless or with viewer).
        
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
        
        Returns:
        --------
        dict
            Validation results with metrics
        """
        max_steps = int(duration_seconds / self.sim_dt)
        step = 0
        counter = 0
        
        # Track metrics
        start_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        start_z = self.data.qpos[2] if len(self.data.qpos) > 2 else 0.0
        body_heights = []
        fall_steps_below_threshold = 0
        total_steps_below_threshold = 0
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
                        action = self.policy.infer(obs)
                        
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
                    action = self.policy.infer(obs)
                    
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
        
        # Calculate metrics
        end_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        forward_distance = end_x - start_x
        episode_length = step
        stability_score = 1.0 if not fall_detected else (fall_step / max_steps) if fall_step else 0.0
        fraction_below_threshold = total_steps_below_threshold / episode_length if episode_length > 0 else 0.0
        
        # Determine pass/fail
        passed = not fall_detected and forward_distance > 0.1
        
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


def validate_single_model(onnx_path, duration_seconds=120, use_viewer=False, fall_height_threshold=0.3, fall_duration_steps=500):
    """Validate a single ONNX model in simulation.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX file
    duration_seconds : float
        Maximum simulation duration in seconds
    use_viewer : bool
        If True, show MuJoCo viewer
    fall_height_threshold : float
        Body height threshold for fall detection
    fall_duration_steps : int
        Number of consecutive steps below threshold to consider fallen
    
    Returns:
    --------
    dict
        Validation results
    """
    # Load metadata to get model info
    metadata_path = f"{onnx_path}.metadata.json"
    if not os.path.exists(metadata_path):
        return {
            "status": "FAIL",
            "error": "Metadata file not found",
            "onnx_path": onnx_path
        }
    
    try:
        # Default paths
        model_path = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        reference_data = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
        
        # Initialize simulation
        sim = HeadlessSimulation(
            model_path=model_path,
            reference_data=reference_data,
            onnx_model_path=onnx_path,
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
        result["onnx_path"] = onnx_path
        
        return result
        
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "onnx_path": onnx_path
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX model in MuJoCo simulation"
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
    
    print(f"Validating {len(onnx_files)} ONNX model(s) in simulation...")
    print(f"Duration: {args.duration} seconds")
    print(f"Viewer: {'Enabled' if args.viewer else 'Disabled'}")
    print(f"Fall threshold: {args.fall_height_threshold}m for {args.fall_duration_steps} steps")
    print()
    
    results = []
    for onnx_path in sorted(onnx_files):
        print(f"Testing {os.path.basename(onnx_path)}...")
        result = validate_single_model(
            onnx_path, 
            args.duration,
            use_viewer=args.viewer,
            fall_height_threshold=args.fall_height_threshold,
            fall_duration_steps=args.fall_duration_steps
        )
        results.append(result)
        
        status = result["status"]
        filename = os.path.basename(onnx_path)
        
        if status == "PASS":
            print(f"✓ {filename}")
            print(f"  Stability: {result['stability_score']:.2f}, Forward: {result['forward_distance']:.2f}m, Steps: {result['episode_length']}")
        else:
            print(f"✗ {filename}")
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                fall_info = f" (fell at step {result['fall_step']})" if result.get('fall_detected') else ""
                fraction_info = f", Below threshold: {result.get('fraction_below_threshold', 0):.1%}" if 'fraction_below_threshold' in result else ""
                print(f"  Stability: {result['stability_score']:.2f}, Forward: {result['forward_distance']:.2f}m, Steps: {result['episode_length']}{fall_info}{fraction_info}")
        print()
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    
    print(f"Summary: {passed}/{len(results)} passed, {failed} failed")


if __name__ == "__main__":
    main()

