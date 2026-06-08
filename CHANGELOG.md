# Changelog — XPUSYS-Monitor-NG

All notable changes for the Windows-native AMD ROCm port are documented here.

This fork is based on [ComfyUI-XPUSYS-Monitor](https://github.com/allanmeng/ComfyUI-XPUSYS-Monitor)
v1.0.3 by allanmeng. The version below tracks deviations from that baseline.

---

## v0.1.0 — 2026-06-08

### Intent

The upstream AMD provider (`providers/amd.py`) relied on `rocm_smi_lib` — a
Linux-only Python package that wraps `librocm_smi64.so`. On Windows with AMD
ROCm, `pip install rocm_smi_lib` fails because the native `.so` library does
not exist. This port replaces every `rocm_smi` call with a Windows-native
alternative.

### Detection: `providers/__init__.py`

**Problem:** The upstream detection function `_is_amd_rocme()` accessed
`torch.version.roc` directly. On Windows AMD ROCm builds (tested with PyTorch
2.9.1+rocm7.2.1), the `roc` attribute **does not exist** on the
`torch.version` module — raising `AttributeError`. The outer `try/except`
caught it and returned `False`, causing the auto-detector to load
`NvidiaProvider` instead of `AMDProvider`. The GPU name fallback code after
the `roc` check was unreachable.

**Fix:** Replaced bare attribute access with `getattr(torch.version, 'roc',
None)`. Added a secondary signal `getattr(torch.version, 'hip', None)` for
HIP-based detection. Added a tertiary fallback scanning the GPU device name
(via `torch.cuda.get_device_name(0)`) for the markers `"amd"`, `"radeon"`,
or `"advanced micro devices"`.

### VRAM: `providers/amd.py` → `_read_vram()`

**Problem:** The upstream AMD provider used `rocm_smi.getMemFreeVdev(0)`,
`rocm_smi.getMemSizeVdev(0)`, and `rocm_smi.getMemUsedVdev(0)` for
driver-level VRAM reads. Without `rocm_smi_lib`, the fallback returned only
total VRAM from `torch.cuda.get_device_properties(0).total_memory`, leaving
`free` and `driver_used` as `0.0` — making the PRED predictor and VRAM
capsule unusable.

**Fix:** Replaced all three `rocm_smi` VRAM calls with
`torch.cuda.mem_get_info(device_index)`, which returns `(free_bytes,
total_bytes)` from the AMD driver on ROCm 6+ for Windows. Moved the call
inside a `try/except` with a fallback to `get_device_properties().total_memory`
if `mem_get_info` is unavailable. Added `torch.cuda.synchronize()` before
reads to force CUDA/HIP context creation (some ROCm builds defer context
init until the first GPU operation, returning zeros otherwise).

### GPU Load: `providers/amd.py` → `_read_gpu_load()`

**Problem:** The upstream used `rocm_smi.getGpuBusyVdev(0)`. No standard
Python-accessible equivalent exists on Windows AMD.

**Solution attempted — PDH (ctypes):** Added `_PdhQuery` to
`providers/_utils.py` using `ctypes` wrappers around `pdh.dll` to query
`\GPU Engine(*)\Utilization Percentage`. The wildcard counter path does not
aggregate correctly with `PdhGetFormattedCounterValue` (returns only the
first matching instance). This approach was disabled for AMD in favour of
typeperf.

**Solution adopted — typeperf:** Added `_TypeperfGpuQuery` to
`providers/_utils.py`. Uses Windows built-in `typeperf.exe` (available since
Vista) with the same counter path `\GPU Engine(*)\Utilization Percentage`.
Output is CSV; we parse columns after the timestamp and take `max()` across
all engine instances. Averaging would dilute the signal (hundreds of engine
columns including idle video/copy/timer). `max()` correctly reflects the
busiest engine (typically 3D or Compute during a ComfyUI workflow). At idle
all engines report ~0%, so the capsule drops cleanly.

**Attempted — amdsmi:** Added `_AmdSmiGpuQuery` to `providers/_utils.py`
using the official AMD SMI Python library (`pip install amdsmi`). On Windows
the library searches for `libamd_smi.so` (a Linux shared object) at
`D:\opt\rocm\lib\`, which does not exist on the tested configuration. The
class logs a single info line and gracefully skips if `amdsmi` is not
installed or fails to load.

### GPU Frequency / Temperature / Power: `providers/amd.py`

**Problem:** The upstream used `rocm_smi.getSingleClockSpeed(0)`,
`rocm_smi.getTempVdev(0)`, `rocm_smi.getPowerVdev(0)`, and
`rocm_smi.getPowerCapVdev(0)` for clock speed, temperature, and power draw.
The AMD Windows WDDM driver on the RX 9070 XT does not register these
performance counters through any standard Python-accessible interface.

**Resolution:** All three return sentinel values matching the `GPUSnapshot`
contract defaults — `0.0` for frequency, `-1.0` for temperature,
`(-1.0, 0.0, False)` for power. The frontend displays these as unavailable
(`--` / greyed out), identical behaviour to when the Intel provider cannot
open Level Zero handles or the NVIDIA provider cannot reach pynvml.

### Shared Utilities: `providers/_utils.py` (new file)

**Problem:** The upstream AMD provider imported system-level CPU and RAM
utility functions (`_get_cpu_info`, `_read_cpu_ram_stats`,
`_read_commit_charge`) from `providers/intel.py`. This created a spurious
dependency on the Intel Level Zero provider code for non-Intel users.

**Fix:** Extracted these three functions plus `_is_admin()` into a new shared
module `providers/_utils.py`. Also relocated `_PdhQuery`, `_TypeperfGpuQuery`,
and `_AmdSmiGpuQuery` into the same module. Both `amd.py` and `nvidia.py`
now import from `_utils.py` instead of `intel.py`. The `intel.py` module is
no longer needed unless the Intel provider is loaded (auto-detection
fallback path).

### NVIDIA Provider: `providers/nvidia.py`

**Change:** Updated import path from `from .intel import ...` to
`from ._utils import ...`. No functional change — identical utility
functions.

### Frontend: `web/xpu_monitor.js`

**Problem:** The `__init__.py` declares `WEB_DIRECTORY = "./web"` which
tells ComfyUI to serve the JavaScript toolbar extension from a `web/`
subdirectory. This directory was not included in the initial workspace,
causing the toolbar capsules to not render.

**Fix:** Added `web/xpu_monitor.js` (56 KB, identical to upstream v1.0.3).
The file is the full JavaScript frontend that renders the seven-capsule
status bar, handles WebSocket updates from the backend, and provides the
VRAM predictor UI. No modifications were made.

### Dependencies: `requirements.txt`

**Change:** `rocm_smi_lib` commented out with an explanatory note. No
replacement dependency added — VRAM reads use `torch.cuda` (bundled with
the ROCm PyTorch installation), GPU load reads use `typeperf` (Windows
built-in), and CPU/RAM reads use `psutil` (already required by the upstream).

### Project Metadata: `pyproject.toml`

**Changes:**
- Repository URL updated to `https://github.com/forkless/XPUSYS-Monitor-NG`
- Display name set to `XPUSYS-Monitor-NG`
- Description updated to reflect the POC nature
- Publisher ID set to `forkless`

### Documentation

- `README.md` — rewritten for the fork with POC context, status table,
  relationship to upstream, tested hardware, support disclaimer, MIT license
- `AMD.md` — detailed technical summary of every change, intent, and end
  result table
- `LICENSE.md` — MIT license (matches upstream)
