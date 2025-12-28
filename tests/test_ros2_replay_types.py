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
Unit test for ROS2Replay type preservation of episode_done and truncation.

This test verifies that episode_done and truncation maintain their input types
(float32) through the step function to avoid JAX scan type mismatch errors.

uv run python -m unittest tests.test_ros2_replay_types
"""

import unittest
import tempfile
import h5py
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from playground.open_duck_mini_v2 import ros2_replay
from mujoco_playground._src import mjx_env


class TestROS2ReplayTypes(unittest.TestCase):
    """Test type preservation in ROS2Replay step function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary HDF5 file with dummy data
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self.temp_file.close()
        
        # Create dummy data (101-element observations)
        n_samples = 100
        obs_size = 101
        action_size = 10
        
        with h5py.File(self.temp_file.name, 'w') as f:
            # Create observations: [gyro(3), accel(3), cmd(7), joint_pos(10), joint_vel(10),
            #                      last_act(10), last_last_act(10), last_last_last_act(10),
            #                      motor_targets(10), contact(2), imitation_phase(2)]
            observations = np.random.randn(n_samples, obs_size).astype(np.float32)
            actions = np.random.randn(n_samples, action_size).astype(np.float32)
            velocity_commands = np.random.randn(n_samples, 3).astype(np.float32)
            
            f.create_dataset('observations', data=observations)
            f.create_dataset('actions', data=actions)
            f.create_dataset('velocity_commands', data=velocity_commands)
        
        # Create environment with test data
        config_overrides = {"data_path": self.temp_file.name}
        self.env = ros2_replay.ROS2Replay(
            task="flat_terrain",
            config_overrides=config_overrides
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import os
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_episode_done_truncation_type_preservation(self):
        """Test that episode_done and truncation maintain float32 dtype."""
        rng = jax.random.PRNGKey(42)
        
        # Reset environment
        state = self.env.reset(rng)
        
        # Simulate what the training wrapper does: add episode_done and truncation
        # with float32 dtype and batched shape [128]
        batch_size = 128
        state.info["episode_done"] = jp.zeros(batch_size, dtype=jp.float32)
        state.info["truncation"] = jp.zeros(batch_size, dtype=jp.float32)
        
        # Get initial types
        initial_episode_done_dtype = state.info["episode_done"].dtype
        initial_truncation_dtype = state.info["truncation"].dtype
        initial_episode_done_shape = state.info["episode_done"].shape
        initial_truncation_shape = state.info["truncation"].shape
        
        self.assertEqual(initial_episode_done_dtype, jp.float32)
        self.assertEqual(initial_truncation_dtype, jp.float32)
        self.assertEqual(initial_episode_done_shape, (batch_size,))
        self.assertEqual(initial_truncation_shape, (batch_size,))
        
        # Take a step
        action = jp.zeros(self.env.action_size)
        next_state = self.env.step(state, action)
        
        # Verify types are preserved
        final_episode_done_dtype = next_state.info["episode_done"].dtype
        final_truncation_dtype = next_state.info["truncation"].dtype
        final_episode_done_shape = next_state.info["episode_done"].shape
        final_truncation_shape = next_state.info["truncation"].shape
        
        # Types must match input types exactly
        self.assertEqual(final_episode_done_dtype, jp.float32, 
                        f"episode_done dtype changed from {initial_episode_done_dtype} to {final_episode_done_dtype}")
        self.assertEqual(final_truncation_dtype, jp.float32,
                        f"truncation dtype changed from {initial_truncation_dtype} to {final_truncation_dtype}")
        self.assertEqual(final_episode_done_shape, initial_episode_done_shape,
                        f"episode_done shape changed from {initial_episode_done_shape} to {final_episode_done_shape}")
        self.assertEqual(final_truncation_shape, initial_truncation_shape,
                        f"truncation shape changed from {initial_truncation_shape} to {final_truncation_shape}")
        
        # Verify done field is boolean (not float32)
        self.assertEqual(next_state.done.dtype, jp.bool_,
                        f"done field should be boolean, got {next_state.done.dtype}")

    def test_multiple_steps_type_consistency(self):
        """Test that types remain consistent across multiple steps."""
        rng = jax.random.PRNGKey(42)
        state = self.env.reset(rng)
        
        # Add wrapper fields
        batch_size = 128
        state.info["episode_done"] = jp.zeros(batch_size, dtype=jp.float32)
        state.info["truncation"] = jp.zeros(batch_size, dtype=jp.float32)
        
        action = jp.zeros(self.env.action_size)
        
        # Take multiple steps
        for _ in range(5):
            state = self.env.step(state, action)
            
            # Verify types are still float32
            self.assertEqual(state.info["episode_done"].dtype, jp.float32,
                           f"episode_done dtype changed to {state.info['episode_done'].dtype} at step {_}")
            self.assertEqual(state.info["truncation"].dtype, jp.float32,
                           f"truncation dtype changed to {state.info['truncation'].dtype} at step {_}")
            self.assertEqual(state.done.dtype, jp.bool_,
                           f"done dtype changed to {state.done.dtype} at step {_}")
    
    def test_jax_scan_type_consistency(self):
        """Test that types work correctly with jax.lax.scan (critical for training).
        
        This test actually runs the step function through jax.lax.scan to verify
        that input and output types match exactly, which is what JAX requires.
        """
        rng = jax.random.PRNGKey(42)
        state = self.env.reset(rng)
        
        # Add wrapper fields as training would
        batch_size = 128
        state.info["episode_done"] = jp.zeros(batch_size, dtype=jp.float32)
        state.info["truncation"] = jp.zeros(batch_size, dtype=jp.float32)
        
        # Create a scan body function that mimics training rollouts
        def scan_step(carry_state, _):
            """Step function for JAX scan."""
            action = jp.zeros(self.env.action_size)
            next_state = self.env.step(carry_state, action)
            return next_state, None  # Return state as carry, no accumulated output
        
        # Run through jax.lax.scan - this will fail if types don't match
        # JAX will raise TypeError if carry input/output types differ
        num_steps = 5
        try:
            final_state, _ = jax.lax.scan(
                scan_step, 
                state, 
                None,  # No input sequence needed
                length=num_steps
            )
            
            # If we get here, types were preserved correctly
            self.assertEqual(final_state.info["episode_done"].dtype, jp.float32)
            self.assertEqual(final_state.info["truncation"].dtype, jp.float32)
            self.assertEqual(final_state.done.dtype, jp.bool_)
            
        except TypeError as e:
            # This is what we're testing for - if types don't match, scan fails
            self.fail(f"JAX scan failed due to type mismatch: {e}")


if __name__ == '__main__':
    unittest.main()

