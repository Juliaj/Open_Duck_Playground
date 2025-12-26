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
Unit tests for BaseRunner class.

uv run python -m unittest tests.test_runner
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from argparse import Namespace

from playground.common.runner import BaseRunner


class TestBaseRunner(unittest.TestCase):
    """Test cases for BaseRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.args = Namespace(
            output_dir=self.temp_dir.name,
            num_timesteps=1000,
            use_jax_to_onnx=False,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            self.temp_dir.cleanup()

    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_init_defaults(self, mock_environ, mock_makedirs, mock_jax_config, mock_writer):
        """Test BaseRunner initialization with default values."""
        runner = BaseRunner(self.args)
        
        self.assertEqual(runner.output_dir.name, os.path.basename(self.temp_dir.name))
        self.assertEqual(runner.num_timesteps, 1000)
        self.assertFalse(runner.use_jax_to_onnx)
        self.assertIsNone(runner.action_size)
        self.assertIsNone(runner.obs_size)

    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_init_with_jax_to_onnx_flag(self, mock_environ, mock_makedirs, mock_jax_config, mock_writer):
        """Test BaseRunner initialization with use_jax_to_onnx flag."""
        self.args.use_jax_to_onnx = True
        runner = BaseRunner(self.args)
        
        self.assertTrue(runner.use_jax_to_onnx)

    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_progress_callback(self, mock_environ, mock_makedirs, mock_jax_config, mock_writer):
        """Test progress_callback logs metrics."""
        runner = BaseRunner(self.args)
        mock_writer_instance = Mock()
        runner.writer = mock_writer_instance
        
        metrics = {
            "eval/episode_reward": 10.5,
            "eval/episode_reward_std": 2.0,
            "train/loss": 0.1,
        }
        
        runner.progress_callback(100, metrics)
        
        self.assertEqual(mock_writer_instance.add_scalar.call_count, 3)

    @patch("playground.common.runner.export_onnx")
    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_export_onnx_model_tensorflow(self, mock_environ, mock_makedirs, mock_jax_config, 
                                          mock_writer, mock_export_onnx):
        """Test _export_onnx_model uses TensorFlow export by default."""
        runner = BaseRunner(self.args)
        runner.action_size = 12
        runner.obs_size = 48
        runner.ppo_params = Mock()
        
        params = Mock()
        output_path = os.path.join(self.temp_dir.name, "test.onnx")
        
        runner._export_onnx_model(params, output_path)
        
        mock_export_onnx.assert_called_once_with(
            params, 12, runner.ppo_params, 48, output_path=output_path
        )

    @patch("playground.common.runner.export_onnx_jax")
    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_export_onnx_model_jax(self, mock_environ, mock_makedirs, mock_jax_config, 
                                   mock_writer, mock_export_onnx_jax):
        """Test _export_onnx_model uses JAX export when flag is set."""
        self.args.use_jax_to_onnx = True
        runner = BaseRunner(self.args)
        runner.action_size = 12
        runner.obs_size = 48
        runner.ppo_params = Mock()
        
        params = Mock()
        output_path = os.path.join(self.temp_dir.name, "test.onnx")
        
        runner._export_onnx_model(params, output_path)
        
        mock_export_onnx_jax.assert_called_once_with(
            params, 12, runner.ppo_params, 48, output_path=output_path
        )

    @patch("playground.common.runner.export_onnx")
    @patch("playground.common.runner.ocp.PyTreeCheckpointer")
    @patch("playground.common.runner.orbax_utils")
    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_policy_params_fn_success(self, mock_environ, mock_makedirs, mock_jax_config, 
                                      mock_writer, mock_orbax_utils, mock_checkpointer_class,
                                      mock_export_onnx):
        """Test policy_params_fn saves checkpoint and exports ONNX successfully."""
        runner = BaseRunner(self.args)
        runner.action_size = 12
        runner.obs_size = 48
        runner.ppo_params = Mock()
        
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        mock_save_args = Mock()
        mock_orbax_utils.save_args_from_target.return_value = mock_save_args
        
        params = Mock()
        make_policy = Mock()
        
        runner.policy_params_fn(100, make_policy, params)
        
        mock_checkpointer.save.assert_called_once()
        mock_export_onnx.assert_called_once()

    @patch("playground.common.runner.export_onnx")
    @patch("playground.common.runner.ocp.PyTreeCheckpointer")
    @patch("playground.common.runner.orbax_utils")
    @patch("playground.common.runner.SummaryWriter")
    @patch("playground.common.runner.jax.config")
    @patch("os.makedirs")
    @patch("os.environ")
    def test_policy_params_fn_export_error(self, mock_environ, mock_makedirs, mock_jax_config, 
                                         mock_writer, mock_orbax_utils, mock_checkpointer_class,
                                         mock_export_onnx):
        """Test policy_params_fn handles ONNX export errors gracefully."""
        runner = BaseRunner(self.args)
        runner.action_size = 12
        runner.obs_size = 48
        runner.ppo_params = Mock()
        
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        mock_orbax_utils.save_args_from_target.return_value = Mock()
        mock_export_onnx.side_effect = Exception("Export failed")
        
        params = Mock()
        make_policy = Mock()
        
        # Should not raise exception
        runner.policy_params_fn(100, make_policy, params)
        
        # Checkpoint should still be saved
        mock_checkpointer.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()

