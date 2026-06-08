# Integration Guide for upstream maintainer

This document is intended for **allanmeng**, maintainer of
[ComfyUI-XPUSYS-Monitor](https://github.com/allanmeng/ComfyUI-XPUSYS-Monitor).

The changes below are scoped to making `AMDProvider` work on Windows without
`rocm_smi_lib`. Nothing outside `providers/amd.py` and `providers/__init__.py`
needs to change in your codebase. The `_utils.py` and `_TypeperfGpuQuery` are
new standalone files you can take or leave.

---

## 1. Detection fix — `providers/__init__.py`

**Location:** `_is_amd_rocme()` function

**Problem:** `torch.version.roc` does not exist as an attribute on some
Windows ROCm builds (tested with PyTorch 2.9.1+rocm7.2.1). Bare attribute
access raises `AttributeError`, caught by the outer `except`, and the
detector falls through to `NvidiaProvider`.

**Fix (3 lines changed):** Replace `torch.version.roc` with
`getattr(torch.version, 'roc', None)`. Add `getattr(torch.version, 'hip',
None)` as a secondary signal. GPU name fallback (`"amd"`, `"radeon"`,
`"advanced micro devices"`) for builds where neither `roc` nor `hip`
attribute exists.

```
if getattr(torch.version, 'roc', None) is not None:
    return True
```

---

## 2. VRAM — `providers/amd.py` → `_read_vram()`

**Replaces:** `rocm_smi.getMemFreeVdev(0)`, `.getMemSizeVdev(0)`,
`.getMemUsedVdev(0)`

**Substitute:** `torch.cuda.mem_get_info(device_index)` returns `(free_bytes,
total_bytes)`. This is the same function used by NVIDIA CUDA — ROCm's HIP
runtime implements the same API surface. Works on ROCm 6+ for Windows.

```python
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
free_gb = free_bytes / (1024**3)
total_gb = total_bytes / (1024**3)
used_gb = max(0.0, total_gb - free_gb)
```

**Caveat:** Call `torch.cuda.synchronize(0)` before `mem_get_info()` on
initialisation — some ROCm builds defer HIP context creation until the first
GPU operation and `mem_get_info` returns `(0, 0)` without an active context.

---

## 3. GPU load — `providers/amd.py` → `_read_gpu_load()`

**Replaces:** `rocm_smi.getGpuBusyVdev(0)`

**No direct torch equivalent.** Two options:

### Option A (recommended): `typeperf` (Windows built-in)

Add the `_TypeperfGpuQuery` class from `providers/_utils.py` in our repo.
It calls:

```
typeperf "\GPU Engine(*)\Utilization Percentage" -sc 1
```

Parses the CSV output (one column per engine instance), takes `max()` across
all engines. Available on every Windows system since Vista — zero
dependencies. The `_utils.py` module is self-contained.

### Option B: `amdsmi` (official AMD SMI library)

`pip install amdsmi`. Talks directly to the AMD driver (not through WDDM).
Currently Linux-only — the PyPI wrapper searches for `libamd_smi.so`. If AMD
releases a Windows wheel in the future, this will work without code changes.
The `_AmdSmiGpuQuery` class is in `providers/_utils.py`.

---

## 4. GPU frequency / temperature / power — `providers/amd.py`

**Replaces:** `rocm_smi.getSingleClockSpeed(0)`, `.getTempVdev(0)`,
`.getPowerVdev(0)`, `.getPowerCapVdev(0)`

**No substitute available.** The AMD Windows WDDM driver on tested hardware
(RX 9070 XT, ROCm 7.2) does not expose these through any Python-accessible
API. Return sentinel values matching the `GPUSnapshot` contract defaults:

| Metric | Sentinel | Effect |
|---|---|---|
| Core clock | `0.0` | Capsule shows `0MHz` |
| Temperature | `-1.0` | Frontend greys out display |
| Power draw | `(-1.0, 0.0, False)` | `power_available=False` greys out PWR capsule |

---

## 5. Shared utility functions — `providers/_utils.py` (optional)

The CPU/RAM utility functions (`_get_cpu_info`, `_read_cpu_ram_stats`,
`_read_commit_charge`, `_is_admin`) were extracted from `providers/intel.py`
into a shared module. If you prefer to keep them in `intel.py`, just update
the import in `amd.py` (and `nvidia.py`) accordingly:

```python
# For _utils.py:
from ._utils import _get_cpu_info, _read_cpu_ram_stats, ...

# For intel.py (original):
from .intel import _get_cpu_info, _read_cpu_ram_stats, ...
```

---

## Files to touch (minimal set)

| File | Action |
|---|---|
| `providers/__init__.py` | Fix `_is_amd_rocme()` — 3 lines |
| `providers/amd.py` | Replace `_read_vram`, `_read_gpu_load`, freq/temp/power sentinels, add `_TypeperfGpuQuery` import |
| `providers/_utils.py` | **New file** — contains `_TypeperfGpuQuery` and optionally shared CPU/RAM utils |

Everything else (`base.py`, `nvidia.py`, `xpu_server.py`, `web/`, `__init__.py`)
is unchanged functionally from the upstream baseline.
