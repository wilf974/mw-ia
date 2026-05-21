"""Diagnostic CUDA pour MW_IA — affiche device, VRAM, version CUDA."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"CUDA dispo  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[!] CUDA indisponible — fallback CPU force.")
        return 1

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / (1024**3)
    print(f"GPU         : {name}")
    print(f"Compute Cap : {props.major}.{props.minor}")
    print(f"VRAM total  : {vram_gb:.2f} Go")

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    print(f"Sanity matmul OK : shape={tuple(y.shape)} dtype={y.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
