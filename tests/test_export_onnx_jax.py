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
Unit tests for JAX to ONNX export functionality.

uv run python -m unittest tests.test_export_onnx_jax
"""

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from playground.common.export_jax_to_onnx import export_onnx_jax


class TestExportOnnxJax(unittest.TestCase):
    """Test cases for export_onnx_jax function."""

    def setUp(self):
        """Set up test fixtures."""
        self.act_size = 12
        self.obs_size = 48
        self.output_path = None
        self.temp_dir = None

    def tearDown(self):
        """Clean up test fixtures."""
        if self.output_path and os.path.exists(self.output_path):
            os.remove(self.output_path)
        if self.temp_dir:
            if hasattr(self.temp_dir, 'cleanup'):
                self.temp_dir.cleanup()
            else:
                # If it's a string path, try to remove it
                import shutil
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)

    def create_mock_params(self):
        """Create mock policy parameters for testing."""
        # Mock normalization parameters
        mean = np.random.randn(self.obs_size).astype(np.float32)
        std = np.abs(np.random.randn(self.obs_size).astype(np.float32)) + 0.1
        
        norm_params = Mock()
        norm_params.mean = {"state": mean}
        norm_params.std = {"state": std}
        
        # Mock policy parameters (simplified structure)
        policy_params = {
            "MLP_0": {
                "hidden_0": {
                    "kernel": np.random.randn(self.obs_size, 256).astype(np.float32),
                    "bias": np.random.randn(256).astype(np.float32),
                },
                "hidden_1": {
                    "kernel": np.random.randn(256, 256).astype(np.float32),
                    "bias": np.random.randn(256).astype(np.float32),
                },
                "hidden_2": {
                    "kernel": np.random.randn(256, self.act_size * 2).astype(np.float32),
                    "bias": np.random.randn(self.act_size * 2).astype(np.float32),
                },
            }
        }
        
        model_params = Mock()
        model_params.policy = {"params": policy_params}
        
        return (norm_params, model_params)

    def create_mock_ppo_params(self, include_network_factory=True):
        """Create mock PPO parameters for testing.
        
        Parameters:
        -----------
        include_network_factory : bool
            If True, include network_factory attribute (default: True)
        """
        # Create a dict-like object that supports 'in' operator
        # The implementation uses "network_factory" in ppo_params
        class PPOParams:
            def __init__(self):
                if include_network_factory:
                    network_factory = Mock()
                    network_factory.policy_hidden_layer_sizes = [256, 256]
                    self.network_factory = network_factory
                    self._has_network_factory = True
                else:
                    self._has_network_factory = False
            
            def __contains__(self, key):
                return key == "network_factory" and self._has_network_factory
        
        return PPOParams()

    def test_export_onnx_jax_invalid_obs_size(self):
        """Test that export_onnx_jax raises ValueError for invalid obs_size."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        with self.assertRaises(ValueError) as context:
            export_onnx_jax(
                params, self.act_size, ppo_params, obs_size=0
            )
        self.assertIn("obs_size and act_size must be positive", str(context.exception))

    def test_export_onnx_jax_invalid_act_size(self):
        """Test that export_onnx_jax raises ValueError for invalid act_size."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        with self.assertRaises(ValueError) as context:
            export_onnx_jax(
                params, act_size=-1, ppo_params=ppo_params, obs_size=self.obs_size
            )
        self.assertIn("obs_size and act_size must be positive", str(context.exception))

    def test_export_onnx_jax_signature(self):
        """Test that export_onnx_jax has correct function signature."""
        import inspect
        
        sig = inspect.signature(export_onnx_jax)
        params_list = list(sig.parameters.keys())
        
        self.assertIn("params", params_list)
        self.assertIn("act_size", params_list)
        self.assertIn("ppo_params", params_list)
        self.assertIn("obs_size", params_list)
        self.assertIn("output_path", params_list)
        
        # Check default value for output_path
        self.assertEqual(sig.parameters["output_path"].default, "open_duck_mini_v2_jax.onnx")

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_should_use_opset_11(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax uses opset 11 for Isaac Lab compatibility."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary file for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_model_proto.SerializeToString.return_value = b"mock_onnx_data"
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function
        result = export_onnx_jax(
            params, self.act_size, ppo_params, self.obs_size, self.output_path
        )
        
        # Verify opset 11 was used
        mock_to_onnx.assert_called_once()
        call_kwargs = mock_to_onnx.call_args[1]
        self.assertEqual(call_kwargs["opset"], 11)
        self.assertEqual(result, self.output_path)

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_input_output_names(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax uses correct input/output names."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary file for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_model_proto.SerializeToString.return_value = b"mock_onnx_data"
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function
        export_onnx_jax(
            params, self.act_size, ppo_params, self.obs_size, self.output_path
        )
        
        # Verify input/output names
        mock_to_onnx.assert_called_once()
        call_kwargs = mock_to_onnx.call_args[1]
        self.assertIn("obs", call_kwargs["input_shapes"])
        self.assertEqual(call_kwargs["output_names"], ["continuous_actions"])

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_batch_size(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax uses correct batch size (1, obs_size)."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary file for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_model_proto.SerializeToString.return_value = b"mock_onnx_data"
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function
        export_onnx_jax(
            params, self.act_size, ppo_params, self.obs_size, self.output_path
        )
        
        # Verify batch size
        mock_to_onnx.assert_called_once()
        call_kwargs = mock_to_onnx.call_args[1]
        self.assertEqual(call_kwargs["input_shapes"]["obs"], (1, self.obs_size))

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_calls_make_inference_fn(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax calls make_jax_inference_fn with correct parameters."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary file for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_model_proto.SerializeToString.return_value = b"mock_onnx_data"
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function
        export_onnx_jax(
            params, self.act_size, ppo_params, self.obs_size, self.output_path
        )
        
        # Verify make_jax_inference_fn was called with correct parameters
        mock_make_fn.assert_called_once_with(
            params, self.act_size, ppo_params, self.obs_size
        )
        
        # Verify to_onnx was called with the inference function
        mock_to_onnx.assert_called_once()
        self.assertEqual(mock_to_onnx.call_args[0][0], mock_inference_fn)

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_saves_file(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax saves ONNX model to specified path."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary file for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_data = b"mock_onnx_model_data"
        mock_model_proto.SerializeToString.return_value = mock_data
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function
        result = export_onnx_jax(
            params, self.act_size, ppo_params, self.obs_size, self.output_path
        )
        
        # Verify file was created and contains expected data
        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path, "rb") as f:
            file_data = f.read()
        self.assertEqual(file_data, mock_data)
        self.assertEqual(result, self.output_path)

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_creates_directory(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax creates directory if it doesn't exist."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary directory and nested path
        self.temp_dir = tempfile.TemporaryDirectory()
        nested_dir = os.path.join(self.temp_dir.name, "nested", "path")
        self.output_path = os.path.join(nested_dir, "test_model.onnx")
        
        # Verify directory doesn't exist yet
        self.assertFalse(os.path.exists(nested_dir))
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_model_proto.SerializeToString.return_value = b"mock_onnx_data"
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function
        export_onnx_jax(
            params, self.act_size, ppo_params, self.obs_size, self.output_path
        )
        
        # Verify directory was created and file exists
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(self.output_path))

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    def test_export_onnx_jax_handles_conversion_error(self, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax handles ONNX conversion errors."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Mock inference function
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        # Mock to_onnx to raise an error
        mock_to_onnx.side_effect = Exception("Conversion failed")
        
        # Call function and verify error handling
        with self.assertRaises(ValueError) as context:
            export_onnx_jax(
                params, self.act_size, ppo_params, self.obs_size
            )
        self.assertIn("Failed to convert to ONNX", str(context.exception))
        self.assertIn("Conversion failed", str(context.exception))

    @patch("playground.common.export_jax_to_onnx.make_jax_inference_fn")
    @patch("playground.common.export_jax_to_onnx.to_onnx")
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_export_onnx_jax_handles_file_write_error(self, mock_open, mock_to_onnx, mock_make_fn):
        """Test that export_onnx_jax handles file write errors."""
        params = self.create_mock_params()
        ppo_params = self.create_mock_ppo_params()
        
        # Create temporary file for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        
        # Mock inference function and ONNX model
        mock_inference_fn = Mock()
        mock_make_fn.return_value = mock_inference_fn
        
        mock_model_proto = Mock()
        mock_model_proto.SerializeToString.return_value = b"mock_onnx_data"
        mock_to_onnx.return_value = mock_model_proto
        
        # Call function and verify error handling
        with self.assertRaises(ValueError) as context:
            export_onnx_jax(
                params, self.act_size, ppo_params, self.obs_size, self.output_path
            )
        self.assertIn("Failed to save ONNX model", str(context.exception))

    def test_export_onnx_jax_output_validation(self):
        """Test that export_onnx_jax output matches TensorFlow version."""
        # This test will compare outputs once both implementations are available
        pass


if __name__ == "__main__":
    unittest.main()

