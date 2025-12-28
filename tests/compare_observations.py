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
Compare observations between validate_onnx_simulation.py and ROS2 motion_controller.

This script adds detailed logging to validate_onnx_simulation.py to help debug
differences with the ROS2 implementation.
"""

import argparse
import numpy as np
from validate_onnx_simulation import HeadlessSimulation, ForwardWalkCommand


class ComparisonSimulation(HeadlessSimulation):
    """Simulation with detailed observation logging for comparison."""
    
    def get_obs(self, data, command):
        """Get observation with detailed logging."""
        obs = super().get_obs(data, command)
        
        # Log observation breakdown matching ROS2 format
        if len(obs) >= 101:
            print(f"\n[PYTHON OBS] Full observation (101 dims):")
            print(f"  gyro=[{obs[0]:.4f},{obs[1]:.4f},{obs[2]:.4f}]")
            print(f"  accel=[{obs[3]:.4f},{obs[4]:.4f},{obs[5]:.4f}] (x+1.3={obs[3]:.4f})")
            print(f"  command=[{obs[6]:.4f},{obs[7]:.4f},{obs[8]:.4f},{obs[9]:.4f},{obs[10]:.4f},{obs[11]:.4f},{obs[12]:.4f}]")
            print(f"  joint_pos_rel[0:3]=[{obs[13]:.4f},{obs[14]:.4f},{obs[15]:.4f},{obs[16]:.4f}] (left_hip)")
            print(f"  joint_vel[0:3]=[{obs[27]:.4f},{obs[28]:.4f},{obs[29]:.4f},{obs[30]:.4f}]")
            print(f"  last_action[0:3]=[{obs[41]:.4f},{obs[42]:.4f},{obs[43]:.4f},{obs[44]:.4f}]")
            print(f"  motor_targets[0:3]=[{obs[69]:.4f},{obs[70]:.4f},{obs[71]:.4f},{obs[72]:.4f}]")
            print(f"  contacts=[{obs[83]:.4f},{obs[84]:.4f}]")
            print(f"  phase=[{obs[85]:.4f},{obs[86]:.4f}]")
            print(f"  imitation_i={self.imitation_i:.1f}, phase_period={self.PRM.nb_steps_in_period if self.PRM else 0}")
        
        return obs
    
    def run_headless(self, command_sequence, duration_seconds=10, **kwargs):
        """Run with observation logging every 25 steps."""
        max_steps = int(duration_seconds / self.sim_dt)
        step = 0
        counter = 0
        
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
                
                # Log every 25 policy updates (matching ROS2 logging frequency)
                if (counter // self.decimation) % 25 == 0:
                    print(f"\n[STEP {step}] Policy update #{counter // self.decimation}")
                    print(f"  Command: {command}")
                    print(f"  Action[0:3]: {action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, {action[3]:.4f}")
                    print(f"  Motor targets[0:3]: {self.motor_targets[0]:.4f}, {self.motor_targets[1]:.4f}, {self.motor_targets[2]:.4f}, {self.motor_targets[3]:.4f}")
                    print(f"  Body position: x={self.data.qpos[0]:.4f}, z={self.data.qpos[2]:.4f}")
        
        return {
            "status": "COMPLETE",
            "forward_distance": float(self.data.qpos[0]),
            "final_height": float(self.data.qpos[2]),
        }


def main():
    parser = argparse.ArgumentParser(description="Compare observations with ROS2")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX file")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    
    args = parser.parse_args()
    
    model_path = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    reference_data = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    
    sim = ComparisonSimulation(
        model_path=model_path,
        reference_data=reference_data,
        onnx_model_path=args.onnx,
        standing=False
    )
    
    command_seq = ForwardWalkCommand(linear_vel_x=0.15)
    result = sim.run_headless(command_seq, duration_seconds=args.duration)
    
    print(f"\n[RESULT] Forward distance: {result['forward_distance']:.4f}m")
    print(f"[RESULT] Final height: {result['final_height']:.4f}m")


if __name__ == "__main__":
    main()

#    cd /home/juliajia/dev/Open_Duck_Playground
#    uv run python tests/compare_observations.py --onnx <path_to_onnx> --duration 10


# Next steps:
# Rebuild the ROS2 controller to get the new observation logs.
# Run the Python comparison script:
#    cd /home/juliajia/dev/Open_Duck_Playground   uv run python tests/compare_observations.py --onnx <path_to_onnx> --duration 10
# Compare the [OBS COMPARISON] logs from ROS2 with the [PYTHON OBS] logs from the Python script.
# This will show differences in:
# Gyro/accelerometer values
# Joint positions and velocities
# Action history
# Motor targets
# Contacts and phase
# The phase is updating (theta changes), but the duck isn't moving forward. Possible causes:
# Observation values differ (gyro/accel bias, joint positions, etc.)
# Action processing differs (scaling, offsets)
# Motor targets not being applied correctly
# Run both and compare the observation logs to pinpoint the issue.