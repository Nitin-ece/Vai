#!/usr/bin/env python
"""
GPU / CUDA diagnostic script.

Prints CUDA availability, driver version, device names, VRAM, and
whether PyTorch can allocate a tensor on the GPU.  Exits 0 regardless
of GPU presence (CPU fallback is always OK).
"""
from __future__ import annotations

import platform
import sys


def main() -> None:
    print("=" * 60)
    print("MangaVrse — GPU / CUDA Diagnostics")
    print("=" * 60)
    print(f"Platform : {platform.system()} {platform.release()}")
    print(f"Python   : {sys.version}")

    try:
        import torch

        print(f"PyTorch  : {torch.__version__}")
        print(f"CUDA compiled : {torch.version.cuda or 'N/A'}")
        print(f"cuDNN         : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print()

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"CUDA devices found: {count}")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_mem / (1024 ** 3)
                print(f"  [{i}] {props.name}")
                print(f"      Compute capability : {props.major}.{props.minor}")
                print(f"      VRAM               : {vram_gb:.1f} GB")
                print(f"      Multi-processors   : {props.multi_processor_count}")

            # Quick allocation test
            device = torch.device("cuda:0")
            t = torch.zeros(1, device=device)
            print(f"\n✓ Successfully allocated tensor on {device}")
            del t
            torch.cuda.empty_cache()
        else:
            print("CUDA is NOT available.  Running in CPU-only mode.")
            print("  To enable CUDA, install the matching PyTorch build:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")

    except ImportError:
        print("PyTorch is not installed.")

    try:
        import cv2
        build_info = cv2.getBuildInformation()
        cuda_line = [l for l in build_info.splitlines() if "CUDA" in l]
        print(f"\nOpenCV   : {cv2.__version__}")
        if cuda_line:
            for line in cuda_line[:3]:
                print(f"  {line.strip()}")
    except ImportError:
        print("OpenCV is not installed.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
