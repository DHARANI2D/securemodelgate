"""Environment setup and checks."""
import sys, platform

def print_banner():
    print("=" * 60)
    print("  SecureModelGate — Model Supply Chain Attestation")
    print("  Parasparam 2026 Evaluation Pipeline")
    print("=" * 60)

def check_environment():
    """Check all required dependencies are present."""
    missing = []
    required = ["torch", "torchvision", "numpy", "scipy",
                "sklearn", "jwt", "matplotlib"]
    for pkg in required:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt\n")
        sys.exit(1)

    import torch
    device_info = []
    if torch.backends.mps.is_available():
        device_info.append("MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device_info.append(f"CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device_info.append("CPU")

    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : {', '.join(device_info)}")
    print(f"  Platform: {platform.machine()}")
