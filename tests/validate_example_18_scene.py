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
Validate example_18 MuJoCo scene.xml against Python reference.

This script validates that:
1. The example_18 scene.xml file can be loaded correctly
2. Keyframe values match the Python reference (scene_flat_terrain.xml)
3. Simulation parameters match (timestep, friction, etc.)
4. The model can run a basic simulation step

Usage:
    uv run python tests/validate_example_18_scene.py
    uv run python tests/validate_example_18_scene.py --compare-keyframes
    uv run python tests/validate_example_18_scene.py --run-simulation
    uv run python tests/validate_example_18_scene.py --run-simulation --viewer
    uv run python tests/validate_example_18_scene.py --onnx checkpoints/model.onnx --viewer
    uv run python tests/validate_example_18_scene.py --onnx checkpoints/model.onnx --walk-duration 60 --viewer
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    from etils import epath
except ImportError:
    print("Error: mujoco and etils packages required.")
    print("Install with: pip install mujoco etils")
    sys.exit(1)

# Add playground to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from playground.open_duck_mini_v2 import base
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion


def get_example_18_assets():
    """Build assets dictionary for example_18 MuJoCo model loading.
    
    Returns:
    --------
    dict
        Assets dictionary mapping filenames to file contents
    """
    assets = {}
    
    # Get base assets from open_duck_mini_v2
    base_assets = base.get_assets()
    assets.update(base_assets)
    
    # Add example_18 specific assets from ROS2 example_18 for full validation
    # The scene.xml includes open_duck_mini_v2.xml which should use base assets
    # But we also need the actual assets used in ROS2 example_18
    assets_dir = Path("/home/juliajia/ros2_ws/src/ros-controls/ros2_control_demos/example_18/description/assets")
    
    if assets_dir.exists():
        for stl_file in assets_dir.glob("*.stl"):
            if stl_file.name not in assets:
                assets[stl_file.name] = stl_file.read_bytes()
    
    return assets


def load_reference_keyframe():
    """Load the "home" keyframe from Python reference scene.
    
    Returns:
    --------
    tuple
        (qpos, ctrl) arrays from reference keyframe
    """
    # Path to Python reference scene
    ref_scene_path = Path(__file__).parent.parent / "playground" / "open_duck_mini_v2" / "xmls" / "scene_flat_terrain.xml"
    
    if not ref_scene_path.exists():
        print(f"Warning: Reference scene not found at {ref_scene_path}")
        print("Skipping keyframe comparison.")
        return None, None
    
    try:
        ref_model = mujoco.MjModel.from_xml_string(
            epath.Path(ref_scene_path).read_text(),
            assets=base.get_assets()
        )
        
        home_keyframe = ref_model.keyframe("home")
        ref_qpos = np.array(home_keyframe.qpos)
        ref_ctrl = np.array(home_keyframe.ctrl)
        
        return ref_qpos, ref_ctrl
    except Exception as e:
        print(f"Error loading reference keyframe: {e}")
        return None, None


def validate_scene_xml(scene_path, compare_keyframes=False):
    """Validate the example_18 scene.xml file.
    
    Parameters:
    -----------
    scene_path : Path
        Path to scene.xml file
    compare_keyframes : bool
        If True, compare keyframe values with Python reference
        
    Returns:
    --------
    dict
        Validation results
    """
    results = {
        "scene_loaded": False,
        "model_created": False,
        "keyframe_found": False,
        "keyframe_match": None,
        "timestep_match": False,
        "friction_match": False,
        "errors": []
    }
    
    # Check if file exists
    if not scene_path.exists():
        results["errors"].append(f"Scene file not found: {scene_path}")
        return results
    
    # Load scene XML
    try:
        scene_xml = scene_path.read_text()
        results["scene_loaded"] = True
    except Exception as e:
        results["errors"].append(f"Failed to read scene.xml: {e}")
        return results
    
    # Build assets dictionary
    assets = get_example_18_assets()
    
    # Load MuJoCo model
    try:
        model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
        results["model_created"] = True
    except Exception as e:
        results["errors"].append(f"Failed to create MuJoCo model: {e}")
        import traceback
        results["errors"].append(f"Traceback: {traceback.format_exc()}")
        return results
    
    # Check timestep
    expected_timestep = 0.002
    actual_timestep = model.opt.timestep
    results["timestep_match"] = abs(actual_timestep - expected_timestep) < 1e-6
    if not results["timestep_match"]:
        results["errors"].append(
            f"Timestep mismatch: expected {expected_timestep}, got {actual_timestep}"
        )
    
    # Check friction (first geom with friction)
    # Reference uses friction="0.6" on floor geom
    expected_friction = 0.6
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id >= 0:
        actual_friction = model.geom_friction[floor_geom_id, 0]  # sliding friction
        results["friction_match"] = abs(actual_friction - expected_friction) < 1e-3
        if not results["friction_match"]:
            results["errors"].append(
                f"Floor friction mismatch: expected {expected_friction}, got {actual_friction:.4f}"
            )
    
    # Check for "home" keyframe
    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_keyframe_id >= 0:
        results["keyframe_found"] = True
        home_keyframe = model.keyframe(home_keyframe_id)
        example_qpos = np.array(home_keyframe.qpos)
        example_ctrl = np.array(home_keyframe.ctrl)
        
        if compare_keyframes:
            ref_qpos, ref_ctrl = load_reference_keyframe()
            if ref_qpos is not None and ref_ctrl is not None:
                # Compare qpos
                qpos_match = np.allclose(example_qpos, ref_qpos, atol=1e-6)
                ctrl_match = np.allclose(example_ctrl, ref_ctrl, atol=1e-6)
                results["keyframe_match"] = qpos_match and ctrl_match
                
                if not qpos_match:
                    max_diff = np.max(np.abs(example_qpos - ref_qpos))
                    max_idx = np.argmax(np.abs(example_qpos - ref_qpos))
                    results["errors"].append(
                        f"qpos mismatch: max diff {max_diff:.6f} at index {max_idx}"
                    )
                    results["errors"].append(
                        f"  example_18: {example_qpos[max_idx]:.6f}, reference: {ref_qpos[max_idx]:.6f}"
                    )
                
                if not ctrl_match:
                    max_diff = np.max(np.abs(example_ctrl - ref_ctrl))
                    max_idx = np.argmax(np.abs(example_ctrl - ref_ctrl))
                    results["errors"].append(
                        f"ctrl mismatch: max diff {max_diff:.6f} at index {max_idx}"
                    )
                    results["errors"].append(
                        f"  example_18: {example_ctrl[max_idx]:.6f}, reference: {ref_ctrl[max_idx]:.6f}"
                    )
                
                # Print detailed comparison
                print("\nKeyframe Comparison:")
                print(f"  qpos match: {qpos_match}")
                print(f"  ctrl match: {ctrl_match}")
                if qpos_match and ctrl_match:
                    print("  ✓ Keyframe values match reference")
                else:
                    print("\n  Detailed qpos comparison:")
                    for i, (ex, ref) in enumerate(zip(example_qpos, ref_qpos)):
                        if abs(ex - ref) > 1e-6:
                            print(f"    [{i}] example_18: {ex:.6f}, reference: {ref:.6f}, diff: {abs(ex-ref):.6f}")
                    
                    print("\n  Detailed ctrl comparison:")
                    for i, (ex, ref) in enumerate(zip(example_ctrl, ref_ctrl)):
                        if abs(ex - ref) > 1e-6:
                            print(f"    [{i}] example_18: {ex:.6f}, reference: {ref:.6f}, diff: {abs(ex-ref):.6f}")
    else:
        results["errors"].append("'home' keyframe not found in model")
    
    return results


def run_simulation_test(scene_path, num_steps=100, use_viewer=False):
    """Run a basic simulation test to verify the model works.
    
    Parameters:
    -----------
    scene_path : Path
        Path to scene.xml file
    num_steps : int
        Number of simulation steps to run
    use_viewer : bool
        If True, show MuJoCo viewer for visual inspection
        
    Returns:
    --------
    dict
        Test results
    """
    results = {
        "simulation_ran": False,
        "initial_height": None,
        "final_height": None,
        "height_change": None,
        "errors": []
    }
    
    try:
        import time
        
        # Load model
        scene_xml = scene_path.read_text()
        assets = get_example_18_assets()
        model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
        data = mujoco.MjData(model)
        
        # Initialize to "home" keyframe
        home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_keyframe_id >= 0:
            home_keyframe = model.keyframe(home_keyframe_id)
            data.qpos[:] = home_keyframe.qpos
            data.ctrl[:] = home_keyframe.ctrl
        
        # Get initial body height (assuming first body is the robot base)
        # Floating base z-position is typically at qpos[2]
        results["initial_height"] = data.qpos[2] if len(data.qpos) > 2 else 0.0
        
        if use_viewer:
            with mujoco.viewer.launch_passive(
                model,
                data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                step = 0
                while step < num_steps:
                    step_start = time.time()
                    mujoco.mj_step(model, data)
                    step += 1
                    
                    viewer.sync()
                    
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            # Run simulation steps
            for _ in range(num_steps):
                mujoco.mj_step(model, data)
        
        results["final_height"] = data.qpos[2] if len(data.qpos) > 2 else 0.0
        results["height_change"] = results["final_height"] - results["initial_height"]
        results["simulation_ran"] = True
        
    except Exception as e:
        results["errors"].append(f"Simulation test failed: {e}")
        import traceback
        results["errors"].append(f"Traceback: {traceback.format_exc()}")
    
    return results


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


class Example18Simulation(MJInferBase):
    """Simulation for example_18 scene with ONNX model control."""
    
    def __init__(self, scene_path, onnx_model_path, reference_data=None):
        """Initialize example_18 simulation.
        
        Parameters:
        -----------
        scene_path : Path
            Path to example_18 scene.xml file
        onnx_model_path : Path
            Path to ONNX model file
        reference_data : Path, optional
            Path to reference motion data (polynomial_coefficients.pkl)
        """
        # Initialize base class with scene path (as string)
        # MJInferBase will load the model, but we need to provide assets
        # We'll override the model loading to use our assets
        scene_xml = scene_path.read_text()
        assets = get_example_18_assets()
        
        # Create model with our assets
        self.model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
        self.data = mujoco.MjData(self.model)
        
        # Call parent __init__ but we've already created model
        # We need to manually set up what MJInferBase would set up
        # Actually, let's just replicate the initialization we need
        self.sim_dt = 0.002
        self.decimation = 10
        self.model.opt.timestep = self.sim_dt
        
        # Initialize step (MJInferBase does this)
        mujoco.mj_step(self.model, self.data)
        
        # Get joint/actuator info (from MJInferBase)
        self.num_dofs = self.model.nu
        self.floating_base_name = [
            self.model.jnt(k).name
            for k in range(0, self.model.njnt)
            if self.model.jnt(k).type == 0
        ][0]
        self.actuator_names = [
            self.model.actuator(k).name for k in range(0, self.model.nu)
        ]
        self.joint_names = [
            self.model.jnt(k).name for k in range(0, self.model.njnt)
        ]
        self.backlash_joint_names = [
            j
            for j in self.joint_names
            if j not in self.actuator_names and j not in self.floating_base_name
        ]
        
        # Get all joint addresses (simplified - using helper methods)
        self.all_joint_ids = [self.get_joint_id_from_name(n) for n in self.joint_names]
        self.all_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.joint_names
        ]
        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.actuator_names
        ]
        
        # Get default actuator positions from keyframe
        home_keyframe = self.model.keyframe("home")
        self.default_actuator = np.array(home_keyframe.ctrl)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        
        # Set initial pose
        self.data.qpos[:] = home_keyframe.qpos
        self.data.ctrl[:] = self.default_actuator
        
        # Load ONNX model
        self.policy = OnnxInfer(str(onnx_model_path), awd=True)
        
        # Reference motion
        if reference_data and Path(reference_data).exists():
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
        
        # Initialize sensor addresses (required by MJInferBase methods)
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        if self.gyro_id < 0:
            # Try alternative name
            self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if self.gyro_id >= 0:
            self.gyro_addr = self.model.sensor_adr[self.gyro_id]
            self.gyro_dimensions = 3
        else:
            self.gyro_addr = 0
            self.gyro_dimensions = 0
        
        self.accelerometer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")
        if self.accelerometer_id < 0:
            # Try alternative name
            self.accelerometer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if self.accelerometer_id >= 0:
            self.accelerometer_addr = self.model.sensor_adr[self.accelerometer_id]
            self.accelerometer_dimensions = 3
        else:
            self.accelerometer_addr = 0
            self.accelerometer_dimensions = 0
        
        self.linvel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel")
        if self.linvel_id >= 0:
            self.linvel_dimensions = 3
        else:
            self.linvel_id = -1
            self.linvel_dimensions = 0
        
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        
        self.gravity_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
        if self.gravity_id >= 0:
            self.gravity_dimensions = 3
        else:
            self.gravity_id = -1
            self.gravity_dimensions = 0
        
        # Initialize backlash joint info
        self.backlash_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]
        self.backlash_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.backlash_joint_names
        ]
        
        # Initialize floating base addresses
        self._floating_base_qpos_addr = self.model.jnt_qposadr[
            np.where(self.model.jnt_type == 0)
        ][0]
        self._floating_base_qvel_addr = self.model.jnt_dofadr[
            np.where(self.model.jnt_type == 0)
        ][0]
        self._floating_base_id = self.model.joint(self.floating_base_name).id
        
        # Initialize all joint addresses
        self.all_qvel_addr = np.array(
            [self.model.jnt_dofadr[jad] for jad in self.all_joint_ids]
        )
        self.actuator_qvel_addr = np.array(
            [self.model.jnt_dofadr[jad] for jad in self.actuator_joint_ids]
        )
        
        # Initialize actuator joint dict
        self.actuator_joint_dict = {
            n: self.get_joint_id_from_name(n) for n in self.actuator_names
        }
        
        # Initialize all joint no backlash IDs
        all_idx = self.backlash_joint_ids + list(
            range(self._floating_base_qpos_addr, self._floating_base_qpos_addr + 7)
        )
        all_idx.sort()
        self.all_joint_no_backlash_ids = [idx for idx in all_idx]
    
    def get_joint_id_from_name(self, name):
        """Get joint ID from name."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    
    def get_joint_addr_from_name(self, name):
        """Get joint qpos address from name."""
        joint_id = self.get_joint_id_from_name(name)
        if joint_id >= 0:
            return self.model.jnt_qposadr[joint_id]
        return -1
    
    def get_obs(self, data, command):
        """Get observation from simulation data."""
        # Use parent class methods
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3
        
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        
        contacts = self.get_feet_contacts(data)
        contacts_array = np.array([float(contacts[0]), float(contacts[1])], dtype=np.float32)
        
        # Update imitation phase if not standing
        if self.PRM:
            self.imitation_i += 1.0 * self.phase_frequency_factor
            self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
            self.imitation_phase = np.array([
                np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
            ])
        else:
            self.imitation_phase = np.array([0, 0])
        
        # Build observation
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
    
    def run_walking(self, command_sequence, duration_seconds=30, use_viewer=False):
        """Run walking simulation.
        
        Parameters:
        -----------
        command_sequence : ForwardWalkCommand
            Command sequence to execute
        duration_seconds : float
            Maximum simulation duration in seconds
        use_viewer : bool
            If True, show MuJoCo viewer
            
        Returns:
        --------
        dict
            Results with metrics
        """
        import time
        
        max_steps = int(duration_seconds / self.sim_dt)
        step = 0
        counter = 0
        
        # Track metrics
        start_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        start_z = self.data.qpos[2] if len(self.data.qpos) > 2 else 0.0
        
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
                        
                        # Update action history
                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()
                        
                        # Convert action to motor targets
                        self.motor_targets = self.default_actuator + action * self.action_scale
                        
                        # Apply velocity limits
                        self.motor_targets = np.clip(
                            self.motor_targets,
                            self.prev_motor_targets - self.max_motor_velocity * (self.sim_dt * self.decimation),
                            self.prev_motor_targets + self.max_motor_velocity * (self.sim_dt * self.decimation),
                        )
                        self.prev_motor_targets = self.motor_targets.copy()
                        
                        # Apply control
                        self.data.ctrl[:] = self.motor_targets.copy()
                    
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
                    
                    # Update action history
                    self.last_last_last_action = self.last_last_action.copy()
                    self.last_last_action = self.last_action.copy()
                    self.last_action = action.copy()
                    
                    # Convert action to motor targets
                    self.motor_targets = self.default_actuator + action * self.action_scale
                    
                    # Apply velocity limits
                    self.motor_targets = np.clip(
                        self.motor_targets,
                        self.prev_motor_targets - self.max_motor_velocity * (self.sim_dt * self.decimation),
                        self.prev_motor_targets + self.max_motor_velocity * (self.sim_dt * self.decimation),
                    )
                    self.prev_motor_targets = self.motor_targets.copy()
                    
                    # Apply control
                    self.data.ctrl[:] = self.motor_targets.copy()
        
        # Calculate metrics
        end_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        forward_distance = end_x - start_x
        
        return {
            "forward_distance": float(forward_distance),
            "episode_length": int(step),
            "final_height": float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0.0,
        }


def run_walking_simulation(scene_path, onnx_path, duration_seconds=30, use_viewer=False):
    """Run walking simulation with ONNX model.
    
    Parameters:
    -----------
    scene_path : Path
        Path to scene.xml file
    onnx_path : Path
        Path to ONNX model file
    duration_seconds : float
        Simulation duration in seconds
    use_viewer : bool
        If True, show MuJoCo viewer
        
    Returns:
    --------
    dict
        Results
    """
    results = {
        "simulation_ran": False,
        "errors": []
    }
    
    try:
        # Get reference data path
        ref_data_path = Path(__file__).parent.parent / "playground" / "open_duck_mini_v2" / "data" / "polynomial_coefficients.pkl"
        
        # Create simulation
        sim = Example18Simulation(
            scene_path=scene_path,
            onnx_model_path=onnx_path,
            reference_data=str(ref_data_path) if ref_data_path.exists() else None
        )
        
        # Create forward walk command
        command_seq = ForwardWalkCommand(linear_vel_x=0.15)
        
        # Run walking simulation
        walk_results = sim.run_walking(
            command_seq,
            duration_seconds=duration_seconds,
            use_viewer=use_viewer
        )
        
        results.update(walk_results)
        results["simulation_ran"] = True
        
    except Exception as e:
        results["errors"].append(f"Walking simulation failed: {e}")
        import traceback
        results["errors"].append(f"Traceback: {traceback.format_exc()}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate example_18 MuJoCo scene.xml"
    )
    parser.add_argument(
        "--compare-keyframes",
        action="store_true",
        help="Compare keyframe values with Python reference"
    )
    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Run a basic simulation test"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of simulation steps (default: 100)"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show MuJoCo viewer for visual inspection"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="Path to ONNX model file (enables walking simulation)"
    )
    parser.add_argument(
        "--walk-duration",
        type=float,
        default=30.0,
        help="Walking simulation duration in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent.parent
    scene_path = script_dir / "playground" / "open_duck_mini_v2_example_18" / "scene.xml"
    
    print(f"Validating example_18 scene.xml")
    print(f"  Scene file: {scene_path}")
    print()
    
    # Validate scene
    results = validate_scene_xml(
        scene_path,
        compare_keyframes=args.compare_keyframes
    )
    
    # Print validation results
    print("Validation Results:")
    print(f"  Scene loaded: {'✓' if results['scene_loaded'] else '✗'}")
    print(f"  Model created: {'✓' if results['model_created'] else '✗'}")
    print(f"  Keyframe found: {'✓' if results['keyframe_found'] else '✗'}")
    print(f"  Timestep match (0.002s): {'✓' if results['timestep_match'] else '✗'}")
    print(f"  Friction match (0.6): {'✓' if results['friction_match'] else '✗'}")
    
    if results["keyframe_match"] is not None:
        print(f"  Keyframe match: {'✓' if results['keyframe_match'] else '✗'}")
    
    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  ✗ {error}")
    
    # Run walking simulation if ONNX model provided
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            print(f"\nError: ONNX model not found: {onnx_path}")
            return 1
        
        print(f"\nRunning walking simulation with ONNX model...")
        print(f"  ONNX model: {onnx_path}")
        print(f"  Duration: {args.walk_duration} seconds")
        print(f"  Viewer: {'Enabled' if args.viewer else 'Disabled'}")
        
        walk_results = run_walking_simulation(
            scene_path,
            onnx_path,
            duration_seconds=args.walk_duration,
            use_viewer=args.viewer
        )
        
        print(f"\nWalking Results:")
        print(f"  Simulation ran: {'✓' if walk_results['simulation_ran'] else '✗'}")
        if walk_results['simulation_ran']:
            print(f"  Forward distance: {walk_results['forward_distance']:.4f}m")
            print(f"  Episode length: {walk_results['episode_length']} steps")
            print(f"  Final height: {walk_results['final_height']:.4f}m")
        
        if walk_results["errors"]:
            print("\n  Errors:")
            for error in walk_results["errors"]:
                print(f"    ✗ {error}")
    
    # Run basic simulation test if requested
    elif args.run_simulation:
        print("\nRunning simulation test...")
        print(f"  Viewer: {'Enabled' if args.viewer else 'Disabled'}")
        sim_results = run_simulation_test(scene_path, args.num_steps, use_viewer=args.viewer)
        
        print(f"  Simulation ran: {'✓' if sim_results['simulation_ran'] else '✗'}")
        if sim_results['simulation_ran']:
            print(f"  Initial height: {sim_results['initial_height']:.4f}m")
            print(f"  Final height: {sim_results['final_height']:.4f}m")
            print(f"  Height change: {sim_results['height_change']:.4f}m")
        
        if sim_results["errors"]:
            print("\n  Simulation errors:")
            for error in sim_results["errors"]:
                print(f"    ✗ {error}")
    
    # Summary
    all_passed = (
        results["scene_loaded"] and
        results["model_created"] and
        results["keyframe_found"] and
        results["timestep_match"] and
        results["friction_match"] and
        (results["keyframe_match"] is None or results["keyframe_match"]) and
        len(results["errors"]) == 0
    )
    
    if args.onnx:
        all_passed = walk_results["simulation_ran"] and len(walk_results["errors"]) == 0
    elif args.run_simulation:
        all_passed = all_passed and sim_results["simulation_ran"] and len(sim_results["errors"]) == 0
    
    print()
    if all_passed:
        print("✓ All validations passed")
        return 0
    else:
        print("✗ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# uv run python tests/validate_example_18_scene.py --onnx checkpoints/2025_12_26_165635_300482560.onnx --viewer --walk-duration 60