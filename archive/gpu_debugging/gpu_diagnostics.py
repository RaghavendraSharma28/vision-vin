"""
GPU diagnostics for SmolVLM2 training.

Run this before training to verify your CUDA installation, GPU, and
package stack are compatible. Reports the most likely root cause when
something is wrong.

    python scripts/gpu_diagnostics.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys


def _pkg_version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except ImportError:
        return "NOT INSTALLED"


def _nvidia_smi_driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.decode().strip().split("\n")[0].strip()
    except Exception:
        return "N/A (nvidia-smi not found)"


def _torch_info(torch) -> dict:
    info: dict = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_compiled_version": torch.version.cuda or "N/A",
        "cudnn_version": (
            str(torch.backends.cudnn.version())
            if torch.cuda.is_available()
            else "N/A"
        ),
        "nvidia_driver_version": _nvidia_smi_driver(),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "vram_total_gb": round(p.total_memory / 1024**3, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "fp16_supported": True,
                "tf32_supported": p.major >= 8,
            }
        )
    return info


def _cuda_smoke_tests(torch) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []

    def run(name: str, fn) -> bool:
        try:
            fn()
            torch.cuda.synchronize()
            results.append((name, "PASS"))
            return True
        except Exception as exc:
            results.append((name, f"FAIL — {type(exc).__name__}: {exc}"))
            return False

    ok = run(
        "tensor_allocation",
        lambda: torch.randn(128, 128, device="cuda"),
    )
    if not ok:
        return results  # no point continuing if basic allocation fails

    run(
        "float32_matmul",
        lambda: torch.mm(
            torch.randn(128, 128, device="cuda"),
            torch.randn(128, 128, device="cuda"),
        ),
    )
    run(
        "backward_pass",
        lambda: (lambda x: (x**2).sum().backward())(
            torch.randn(128, 128, device="cuda", requires_grad=True)
        ),
    )
    run(
        "fp16_matmul",
        lambda: torch.mm(
            torch.randn(64, 64, device="cuda", dtype=torch.float16),
            torch.randn(64, 64, device="cuda", dtype=torch.float16),
        ),
    )
    run(
        "bf16_matmul",
        lambda: torch.mm(
            torch.randn(64, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 64, device="cuda", dtype=torch.bfloat16),
        ),
    )
    run(
        "sdpa_fp16",
        lambda: torch.nn.functional.scaled_dot_product_attention(
            *[torch.randn(1, 4, 16, 32, device="cuda", dtype=torch.float16)] * 3
        ),
    )
    return results


def _check_attention_backends(torch) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []

    try:
        torch.nn.functional.scaled_dot_product_attention(
            *[torch.randn(1, 4, 16, 32, device="cuda", dtype=torch.float16)] * 3
        )
        torch.cuda.synchronize()
        results.append(("sdpa (PyTorch built-in)", "AVAILABLE"))
    except Exception as exc:
        results.append(("sdpa (PyTorch built-in)", f"FAILED — {exc}"))

    try:
        fa = importlib.import_module("flash_attn")
        results.append(("flash_attention_2", f"INSTALLED v{fa.__version__}"))
    except ImportError:
        results.append(("flash_attention_2", "NOT INSTALLED (expected on Windows)"))
    except Exception as exc:
        results.append(("flash_attention_2", f"IMPORT ERROR — {exc}"))

    try:
        xf = importlib.import_module("xformers")
        results.append(("xformers", f"INSTALLED v{xf.__version__}"))
    except ImportError:
        results.append(("xformers", "NOT INSTALLED"))
    except Exception as exc:
        results.append(("xformers", f"IMPORT ERROR — {exc}"))

    return results


def _diagnose(
    torch_info: dict,
    smoke: list[tuple[str, str]],
    attn: list[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []

    smoke_failures = [n for n, r in smoke if not r.startswith("PASS")]
    attn_dict = dict(attn)

    if smoke_failures:
        lines.append(f"ISSUE: CUDA smoke test failures: {smoke_failures}")
        lines.append(
            "  Likely cause: PyTorch was compiled for a different CUDA version or compute"
            " capability than what is installed on this machine."
        )
        lines.append(
            "  Fix: reinstall PyTorch matching your CUDA driver from https://pytorch.org/get-started/locally/"
        )
        return lines

    if "INSTALLED" in attn_dict.get("flash_attention_2", ""):
        lines.append(
            "WARNING: flash_attention_2 is installed. On Windows this package is often"
            " compiled for a specific compute capability and can produce 'illegal instruction'"
            " crashes on other GPUs."
        )
        lines.append(
            "  Fix: pass --attention-implementation eager (or sdpa) when training."
        )

    cc_str = torch_info.get("compute_capability", "")
    if cc_str:
        major = int(cc_str.split(".")[0])
        if major < 7:
            lines.append(
                f"WARNING: Compute capability {cc_str} is below 7.0."
                " FP16 tensor cores are unavailable — use --no-fp16 --no-bf16 for fp32 training."
            )

    if not lines:
        lines.append("All smoke tests passed — CUDA and basic ops look healthy.")
        lines.append(
            "If training crashes, the most common culprits are:"
        )
        lines.append(
            "  1. Attention kernel (run: python scripts/smoke_test_vlm_gpu.py --attn-impl eager)"
        )
        lines.append(
            "  2. bitsandbytes incompatibility — try uninstalling bitsandbytes and training without quantization"
        )
        lines.append(
            "  3. Accelerate device placement — verified by the --device cpu smoke test working"
        )

    return lines


def main() -> None:
    sep = "=" * 64

    print(sep)
    print("  GPU DIAGNOSTICS")
    print(sep)

    print("\n[Package Versions]")
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("Pillow", "PIL"),
        ("flash-attn", "flash_attn"),
        ("xformers", "xformers"),
    ]
    for label, mod in packages:
        print(f"  {label:<22} {_pkg_version(mod)}")

    try:
        import torch
    except ImportError:
        print("\ntorch is not installed — cannot continue.")
        sys.exit(1)

    print("\n[PyTorch / CUDA]")
    info = _torch_info(torch)
    for k, v in info.items():
        print(f"  {k:<30} {v}")

    if not info.get("cuda_available"):
        print("\nCUDA is not available. Skipping GPU smoke tests.")
        print("If you expected CUDA: check your PyTorch install and GPU drivers.")
        sys.exit(0)

    print("\n[CUDA Smoke Tests]")
    smoke = _cuda_smoke_tests(torch)
    for name, result in smoke:
        marker = "OK" if result == "PASS" else "!!"
        print(f"  [{marker}] {name:<32} {result}")

    print("\n[Attention Backends]")
    attn = _check_attention_backends(torch)
    for name, result in attn:
        print(f"  {name:<36} {result}")

    print("\n[Diagnosis]")
    for line in _diagnose(info, smoke, attn):
        print(f"  {line}")

    print(f"\n{sep}")
    print("  Next step: python scripts/smoke_test_vlm_gpu.py")
    print(sep)


if __name__ == "__main__":
    main()
