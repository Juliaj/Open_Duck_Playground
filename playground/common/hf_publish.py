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

"""Hugging Face publishing utilities."""


def generate_readme(metadata):
    """Generate README.md content for Hugging Face.
    
    Parameters:
    -----------
    metadata : dict
        Metadata dictionary
    
    Returns:
    --------
    str
        README.md content
    """
    readme = f"""# PPO Policy Model

## Model Details

- **Model Type**: PPO Policy (converted from JAX to ONNX)
- **Original Framework**: JAX/Brax
- **Conversion**: Direct JAX-to-ONNX export (jax2onnx) for RTX 5090 compatibility
- **Training**: Trained using [Open Duck Playground]({metadata['training_repo']})
- **License**: Apache 2.0

## Model Architecture

- **Observation Size**: {metadata['obs_size']}
- **Action Size**: {metadata['act_size']}
- **ONNX Opset Version**: {metadata['opset_version']}
- **Original Architecture**: Brax PPO networks

## Attribution

- Model architecture based on Brax PPO networks
- Training code: {metadata['training_repo']}
- Conversion code: {metadata['training_repo']} (JAX-to-ONNX export)

## Usage

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# Prepare input (normalized observation)
obs = np.array([[0.0] * {metadata['obs_size']}], dtype=np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
action = session.run([output_name], {{input_name: obs}})[0]

print(f"Action: {{action}}")
```

## Training Information
"""
    
    if "training_step" in metadata:
        readme += f"- **Training Step**: {metadata['training_step']}\n"
    if "reward" in metadata:
        readme += f"- **Reward**: {metadata['reward']:.2f}\n"
    if "reward_std" in metadata:
        readme += f"- **Reward Std**: {metadata['reward_std']:.2f}\n"
    
    readme += f"""
## Input/Output

- **Input**: `obs` - shape `[1, {metadata['obs_size']}]`, normalized observation
- **Output**: `continuous_actions` - shape `[1, {metadata['act_size']}]`, action values in range [-1, 1]

## Normalization

The model expects normalized observations. Normalization parameters are included in `config.json`.
"""
    
    return readme

