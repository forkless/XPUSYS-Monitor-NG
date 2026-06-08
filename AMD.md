# AMD ROCm Windows Port — Summary

## Intent

The upstream `ComfyUI-XPUSYS-Monitor` was designed primarily for Intel Arc (XPU)
and NVIDIA (CUDA). Its AMD GPU support depended on **`rocm_smi_lib`** — a
Linux-only Python package that wraps `librocm_smi64.so`. On Windows with AMD
ROCm, `rocm_smi_lib` cannot be installed via pip because the native library
(`.so`) does not exist on Windows.

This fork (`ComfyUI-XPUSYS-Monitor-AMD`) replaces the AMD provider's GPU
monitoring with Windows-native APIs that work on a standard Windows ROCm
stack — **no `rocm_smi_lib` required**.

## Changes Made

### `providers/amd.py` — Rewritten AMD provider

| Before | After |
|---|---|
| Used `rocm_smi.getMemFreeVdev()` / `getMemSizeVdev()` for VRAM | Uses `torch.cuda.mem_get_info(0)` — returns `(free_bytes, total_bytes)` from the AMD driver directly |
| Used `rocm_smi.getGpuBusyVdev()` for GPU load | Uses `typeperf` (Windows built-in) to read WDDM GPU engine counters averaged via `max()` |
| Used `rocm_smi.getSingleClockSpeed()` for core clock | Returns `0` — driver does not expose via Windows API |
| Used `rocm_smi.getTempVdev()` for temperature | Returns `-1` — driver does not expose via Windows API |
| Used `rocm_smi.getPowerVdev()` for power draw | Returns `(-1, 0, False)` — driver does not expose via Windows API |
| Imported shared utils from `providers/intel.py` | Imports from `providers/_utils.py` (extracted shared module) |

### `providers/_utils.py` — New shared utilities module

Extracted from `providers/intel.py` so all providers can share CPU/RAM polling
without depending on the Intel Level Zero provider:

- `_get_cpu_info()` — CPU model name and thread count
- `_read_cpu_ram_stats()` — CPU load, frequency, RAM total/used/free
- `_read_commit_charge()` — Windows commit charge (virtual memory)
- `_PdhQuery` — PDH API ctypes wrapper (not used on AMD — kept for reference)
- `_TypeperfGpuQuery` — GPU engine utilisation via `typeperf.exe` (CSV parse, uses `max` across all engines)
- `_AmdSmiGpuQuery` — Official AMD SMI library probe (gracefully skipped on Windows where `libamd_smi.so` is unavailable)

### `providers/__init__.py` — Detection fix

`_is_amd_rocme()` now uses `getattr(torch.version, 'roc', None)` instead of
`torch.version.roc` directly. Windows ROCm builds may lack the `roc` attribute
entirely — `getattr` avoids the `AttributeError` and falls through to GPU name
matching (`"amd"`, `"radeon"`, `"advanced micro devices"`).

### `providers/nvidia.py` — Updated imports

NvidiaProvider now imports shared utilities from `_utils.py` instead of `intel.py`
(no functional change).

### `requirements.txt` — Removed `rocm_smi_lib`

`rocm_smi_lib` is commented out with a note explaining it is Linux-only. No
replacement dependency needed — all monitoring uses `torch` (already installed
with ROCm) and `typeperf` (Windows built-in).

### `pyproject.toml`

Project name updated to `ComfyUI-XPUSYS-Monitor-AMD`, publisher set to `forkless`.

### `web/` directory — Frontend

Added `web/xpu_monitor.js` from upstream (56 KB). This is the JavaScript
toolbar extension that renders the seven-capsule status bar in ComfyUI's UI.

## End Result

| Metric | Source | Status |
|---|---|---|
| VRAM free / total | `torch.cuda.mem_get_info(0)` | ✅ |
| VRAM allocated / reserved | `torch.cuda.memory_allocated()` / `memory_reserved()` | ✅ |
| GPU utilisation | `typeperf` WDDM engine counters (max across engines) | ✅ |
| Device name | `torch.cuda.get_device_name(0)` | ✅ |
| GPU core temperature | Driver does not expose via Windows API | ❌ |
| Power draw | Driver does not expose via Windows API | ❌ |
| Core clock | Driver does not expose via Windows API | ❌ |
| GPU load on idle | `max()` across all WDDM engines → ~0% | ✅ |
| GPU load under workflow | `max()` reflects busy 3D/compute engine | ✅ |
| `rocm_smi_lib` dependency | Removed; no replacement needed | ✅ |
| Windows ROCm detection | `getattr(torch.version, 'roc', None)` + GPU name fallback | ✅ |
