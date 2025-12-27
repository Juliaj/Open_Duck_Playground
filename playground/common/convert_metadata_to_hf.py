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
Convert local metadata to Hugging Face format.

This script reads .metadata.json files and converts them to Hugging Face format
(config.json + README.md) for publishing to Hugging Face Model Hub.

Usage:
    uv run python playground/common/convert_metadata_to_hf.py --onnx checkpoints/model.onnx
    uv run python playground/common/convert_metadata_to_hf.py --checkpoints-dir checkpoints
"""

import argparse
import json
import os
from pathlib import Path

from playground.common.hf_publish import generate_readme


def convert_metadata_to_hf(onnx_path):
    """Convert metadata from local format to Hugging Face format.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX file (or directory containing ONNX files)
    
    Returns:
    --------
    bool
        True if conversion successful, False otherwise
    """
    if os.path.isdir(onnx_path):
        # Process all ONNX files in directory
        onnx_files = [f for f in os.listdir(onnx_path) if f.endswith(".onnx")]
        success_count = 0
        for filename in onnx_files:
            file_path = os.path.join(onnx_path, filename)
            if convert_single_metadata(file_path):
                success_count += 1
        print(f"\nCompleted: {success_count}/{len(onnx_files)} files converted")
        return success_count > 0
    else:
        # Process single file
        return convert_single_metadata(onnx_path)


def convert_single_metadata(onnx_path):
    """Convert metadata for a single ONNX file.
    
    Parameters:
    -----------
    onnx_path : str
        Path to ONNX file
    
    Returns:
    --------
    bool
        True if conversion successful, False otherwise
    """
    metadata_path = f"{onnx_path}.metadata.json"
    
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found: {metadata_path}")
        return False
    
    try:
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Generate Hugging Face format files
        base_dir = os.path.dirname(onnx_path)
        config_path = os.path.join(base_dir, "config.json")
        readme_path = os.path.join(base_dir, "README.md")
        
        # Save config.json (same as metadata, but named config.json)
        with open(config_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Generate README.md
        readme_content = generate_readme(metadata)
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        print(f"✓ Converted {os.path.basename(onnx_path)} to Hugging Face format")
        print(f"  Created: {config_path}")
        print(f"  Created: {readme_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {os.path.basename(onnx_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata to Hugging Face format for publishing"
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
    
    args = parser.parse_args()
    
    if args.onnx:
        target_path = args.onnx
    else:
        target_path = args.checkpoints_dir
    
    if not os.path.exists(target_path):
        print(f"Error: Path does not exist: {target_path}")
        return
    
    print(f"Converting metadata to Hugging Face format...")
    print(f"Target: {target_path}")
    print()
    
    convert_metadata_to_hf(target_path)


if __name__ == "__main__":
    main()

