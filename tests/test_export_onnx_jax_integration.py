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
Integration tests for JAX to ONNX export with real checkpoint.

These tests use a real checkpoint file to verify end-to-end functionality.
The checkpoint is stored as a test artifact in tests/test_data/checkpoints/.

uv run python -m unittest tests.test_export_onnx_jax_integration
"""

import unittest
import tempfile
import os
import numpy as np
from pathlib import Path

from playground.common.export_jax_to_onnx import export_onnx_jax, make_jax_inference_fn
from orbax import checkpoint as ocp
from mujoco_playground.config import locomotion_params
import jax
import jax.numpy as jnp


class TestExportOnnxJaxIntegration(unittest.TestCase):
    """Integration tests for export_onnx_jax with real checkpoint."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with checkpoint path."""
        # Checkpoint path - adjust if your checkpoint is in a different location
        # For CI/CD, this should point to a test artifact
        cls.checkpoint_path = os.getenv(
            "TEST_CHECKPOINT_PATH",
            "/home/juliajia/dev/Open_Duck_Playground/checkpoints/2025_12_26_160751_0"
        )
        cls.checkpoint_available = os.path.exists(cls.checkpoint_path)
        
        if not cls.checkpoint_available:
            print(f"Warning: Checkpoint not found at {cls.checkpoint_path}")
            print("Set TEST_CHECKPOINT_PATH environment variable to run integration tests")
    
    def setUp(self):
        """Set up test fixtures."""
        if not self.checkpoint_available:
            self.skipTest("Checkpoint not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_model.onnx")
        
        # Load checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        self.params = checkpointer.restore(self.checkpoint_path)
        
        # Convert list to tuple if needed
        if isinstance(self.params, list):
            self.params = tuple(self.params)
        
        # Get config
        self.ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
        
        # Detect sizes from checkpoint
        self.obs_size = self._detect_obs_size()
        self.act_size = self._detect_act_size()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _detect_obs_size(self):
        """Detect observation size from checkpoint."""
        if len(self.params) >= 1:
            norm_params = self.params[0]
            if isinstance(norm_params, dict) and 'mean' in norm_params and 'state' in norm_params['mean']:
                mean = norm_params['mean']['state']
                return mean.shape[0] if len(mean.shape) == 1 else mean.shape[-1]
        return 46
    
    def _detect_act_size(self):
        """Detect action size from checkpoint."""
        if len(self.params) >= 2:
            policy_params_raw = self.params[1]
            if isinstance(policy_params_raw, dict) and 'params' in policy_params_raw:
                nested = policy_params_raw['params']
                if isinstance(nested, dict) and len(nested) > 0:
                    layer_names = sorted(nested.keys())
                    last_layer_name = layer_names[-1]
                    last_layer = nested[last_layer_name]
                    if isinstance(last_layer, dict) and 'kernel' in last_layer:
                        kernel = last_layer['kernel']
                        output_size = kernel.shape[-1]
                        return output_size // 2  # act_size * 2 = output_size
        return 10
    
    def test_checkpoint_structure(self):
        """Test that checkpoint structure is as expected."""
        self.assertIsInstance(self.params, (list, tuple))
        self.assertGreaterEqual(len(self.params), 2, "Checkpoint should have at least 2 elements")
        
        # Check normalization params
        norm_params = self.params[0]
        self.assertIsInstance(norm_params, dict)
        self.assertIn('mean', norm_params)
        self.assertIn('std', norm_params)
        self.assertIn('state', norm_params['mean'])
        self.assertIn('state', norm_params['std'])
        
        # Check policy params
        policy_params = self.params[1]
        self.assertIsInstance(policy_params, dict)
        self.assertIn('params', policy_params)
    
    def test_inference_function_execution(self):
        """Test that inference function executes successfully."""
        jax_inference_fn = make_jax_inference_fn(
            self.params, self.act_size, self.ppo_params, self.obs_size
        )
        
        # Test with concrete input
        test_input = jnp.zeros((1, self.obs_size), dtype=jnp.float32)
        test_output = jax_inference_fn(test_input)
        
        self.assertEqual(test_output.shape, (1, self.act_size))
        self.assertTrue(np.all(np.isfinite(test_output)))
        # Output should be in [-1, 1] range due to tanh
        self.assertTrue(np.all(test_output >= -1.0))
        self.assertTrue(np.all(test_output <= 1.0))
    
    def test_inference_function_jit_compilation(self):
        """Test that inference function can be JIT compiled."""
        jax_inference_fn = make_jax_inference_fn(
            self.params, self.act_size, self.ppo_params, self.obs_size
        )
        
        # Test JIT compilation
        jax_inference_fn_jit = jax.jit(jax_inference_fn)
        test_input = jnp.zeros((1, self.obs_size), dtype=jnp.float32)
        test_output = jax_inference_fn_jit(test_input)
        
        self.assertEqual(test_output.shape, (1, self.act_size))
        self.assertTrue(np.all(np.isfinite(test_output)))
    
    def test_onnx_export_creates_file(self):
        """Test that ONNX export creates a valid file."""
        export_onnx_jax(
            self.params,
            self.act_size,
            self.ppo_params,
            self.obs_size,
            output_path=self.output_path
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(self.output_path), "ONNX file should be created")
        
        # Check file size
        file_size = os.path.getsize(self.output_path)
        self.assertGreater(file_size, 0, "ONNX file should not be empty")
        self.assertGreater(file_size, 1000, "ONNX file should be substantial")
    
    def test_onnx_model_structure(self):
        """Test that exported ONNX model has correct structure."""
        export_onnx_jax(
            self.params,
            self.act_size,
            self.ppo_params,
            self.obs_size,
            output_path=self.output_path
        )
        
        # Load and inspect ONNX model
        import onnx
        model = onnx.load(self.output_path)
        
        # Check opset version
        self.assertEqual(model.opset_import[0].version, 11, "Should use opset 11")
        
        # Check inputs
        self.assertEqual(len(model.graph.input), 1, "Should have one input")
        input_name = model.graph.input[0].name
        self.assertEqual(input_name, "obs", "Input should be named 'obs'")
        input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        self.assertEqual(input_shape, [1, self.obs_size], f"Input shape should be [1, {self.obs_size}]")
        
        # Check outputs
        self.assertEqual(len(model.graph.output), 1, "Should have one output")
        output_name = model.graph.output[0].name
        self.assertEqual(output_name, "continuous_actions", "Output should be named 'continuous_actions'")
        output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
        self.assertEqual(output_shape, [1, self.act_size], f"Output shape should be [1, {self.act_size}]")
    
    def test_onnx_runtime_inference(self):
        """Test that exported ONNX model works with ONNX Runtime."""
        export_onnx_jax(
            self.params,
            self.act_size,
            self.ppo_params,
            self.obs_size,
            output_path=self.output_path
        )
        
        # Test with ONNX Runtime
        import onnxruntime as ort
        session = ort.InferenceSession(self.output_path, providers=["CPUExecutionProvider"])
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        self.assertEqual(input_name, "obs")
        self.assertEqual(output_name, "continuous_actions")
        
        # Test inference with random input
        test_input = np.random.randn(1, self.obs_size).astype(np.float32)
        outputs = session.run([output_name], {input_name: test_input})
        output = outputs[0]
        
        self.assertEqual(output.shape, (1, self.act_size))
        self.assertTrue(np.all(np.isfinite(output)))
        # Output should be in [-1, 1] range due to tanh
        self.assertTrue(np.all(output >= -1.0))
        self.assertTrue(np.all(output <= 1.0))
    
    def test_onnx_runtime_consistency(self):
        """Test that ONNX Runtime produces consistent outputs."""
        export_onnx_jax(
            self.params,
            self.act_size,
            self.ppo_params,
            self.obs_size,
            output_path=self.output_path
        )
        
        import onnxruntime as ort
        session = ort.InferenceSession(self.output_path, providers=["CPUExecutionProvider"])
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test with same input multiple times
        test_input = np.random.randn(1, self.obs_size).astype(np.float32)
        outputs = []
        for _ in range(3):
            result = session.run([output_name], {input_name: test_input})[0]
            outputs.append(result)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            np.testing.assert_array_almost_equal(
                outputs[0], outputs[i], decimal=5,
                err_msg="ONNX Runtime should produce consistent outputs"
            )
    
    def test_onnx_vs_jax_output_comparison(self):
        """Test that ONNX output matches JAX inference output (approximately)."""
        export_onnx_jax(
            self.params,
            self.act_size,
            self.ppo_params,
            self.obs_size,
            output_path=self.output_path
        )
        
        # Get JAX inference output
        jax_inference_fn = make_jax_inference_fn(
            self.params, self.act_size, self.ppo_params, self.obs_size
        )
        test_input = np.random.randn(1, self.obs_size).astype(np.float32)
        jax_output = np.array(jax_inference_fn(jnp.array(test_input)))
        
        # Get ONNX inference output
        import onnxruntime as ort
        session = ort.InferenceSession(self.output_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_output = session.run([output_name], {input_name: test_input})[0]
        
        # Compare outputs (allow some numerical differences)
        np.testing.assert_array_almost_equal(
            jax_output, onnx_output, decimal=4,
            err_msg="ONNX output should match JAX output approximately"
        )


if __name__ == "__main__":
    unittest.main()

