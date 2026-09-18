# Phase 0 Validation Report

Date: 2026-09-18

## Detected machine

| Item | Result |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.12.8 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU memory | 6,141 MiB (approximately 6 GiB) |
| NVIDIA driver | 566.26 |
| Pre-existing PyTorch | 2.7.1 CPU-only; CUDA unavailable |
| Sample input | `images/dog_running_in_park.jpg` |

## Readiness result

The initial environment inspection correctly reports **not ready** because the pre-existing Python installation has CPU-only PyTorch and does not contain Transformers, Falcon Perception, or pycocotools.

`scripts/bootstrap.ps1` creates an isolated `.venv` and installs CUDA PyTorch from the cu126 wheel index. cu126 is its default because the installed NVIDIA driver is below the cu128 reference project’s stated driver target. After the installation finishes, run the validation and individual model smoke tests documented in the README.

## Capacity risk

The referenced Gemma 4 E4B CUDA checkpoint is not expected to fit reliably on this 6 GiB GPU at the reference implementation’s native automatic dtype. Falcon should be evaluated independently first. Gemma must either prove it can run in the isolated smoke test or be replaced with an explicitly supported, CUDA-compatible quantized configuration before Phase 1 starts.

The smoke-test scripts never load Falcon and Gemma together, preventing an avoidable combined-memory failure during validation.

## Next commands

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe .\scripts\validate_environment.py
.\.venv\Scripts\python.exe .\scripts\smoke_falcon.py --image .\images\dog_running_in_park.jpg --query dog
.\.venv\Scripts\python.exe .\scripts\smoke_gemma.py --image .\images\dog_running_in_park.jpg
```
