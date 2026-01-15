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
#
"""
Inspect a .npz file (NumPy zip archive) and print a readable summary.

Example:
  uv run python tests/inspect_npz.py /tmp/contacts_expected.npz
  uv run python tests/inspect_npz.py /tmp/contacts_expected.npz --head 20
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def _summarize_array(name: str, arr: np.ndarray, head: int) -> str:
    shape = tuple(arr.shape)
    dtype = str(arr.dtype)
    parts = [f"{name}: shape={shape}, dtype={dtype}"]

    if arr.size == 0:
        parts.append("  (empty)")
        return "\n".join(parts)

    # Special-case expected contact signals saved as uint8 (0/1)
    if name.endswith("_contact_raw_expected") or name.endswith("_contact_expected"):
        a = arr.astype(np.uint8).reshape(-1)
        ones = int(a.sum())
        parts.append(f"  ones={ones}/{a.size} ({ones / a.size:.2%})")
        parts.append(f"  head={a[:head].tolist()}")
        return "\n".join(parts)

    flat = arr.reshape(-1)
    preview = flat[:head]
    # Avoid printing huge floats with too much precision.
    if np.issubdtype(arr.dtype, np.floating):
        parts.append(f"  min={float(np.min(arr)):.6g}, max={float(np.max(arr)):.6g}")
    parts.append(f"  head={preview.tolist()}")
    return "\n".join(parts)


def _npz_get(obj: Any, key: str) -> np.ndarray | None:
    try:
        return obj[key]
    except KeyError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .npz file and print its contents summary.")
    parser.add_argument("npz_path", type=str, help="Path to .npz file")
    parser.add_argument("--head", type=int, default=10, help="Number of elements to show from the start (default: 10)")
    args = parser.parse_args()

    head = max(1, int(args.head))

    with np.load(args.npz_path) as z:
        keys = list(z.files)
        print(f"file: {args.npz_path}")
        print(f"keys ({len(keys)}): {keys}")
        print()

        # Print core arrays first if present.
        preferred = [
            "step",
            "time",
            "ncon",
            "left_contact_raw_expected",
            "right_contact_raw_expected",
            "left_contact_expected",
            "right_contact_expected",
            "debounce_on_steps",
            "debounce_off_steps",
        ]
        printed = set()
        for k in preferred:
            arr = _npz_get(z, k)
            if arr is None:
                continue
            print(_summarize_array(k, arr, head))
            print()
            printed.add(k)

        # Print remaining keys.
        for k in keys:
            if k in printed:
                continue
            arr = z[k]
            print(_summarize_array(k, arr, head))
            print()


if __name__ == "__main__":
    main()

