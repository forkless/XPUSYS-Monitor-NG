# XPUSYS-Monitor-NG

> **Proof of Concept** — Native Windows ROCm support for ComfyUI-XPUSYS-Monitor

This is a fork of [ComfyUI-XPUSYS-Monitor](https://github.com/allanmeng/ComfyUI-XPUSYS-Monitor)
by allanmeng, created as a proof of concept to add **native Windows AMD ROCm
support** without requiring `rocm_smi_lib` (a Linux-only Python package).

## Intent

The upstream project is an excellent hardware monitor for ComfyUI, but its AMD
GPU support depended on `rocm_smi_lib`, which cannot be installed on Windows.
This POC replaces the AMD provider's GPU monitoring with Windows-native APIs:

- **VRAM**: `torch.cuda.mem_get_info()` — works on ROCm 6+ for Windows
- **GPU load**: `typeperf` (Windows built-in) — reads WDDM engine counters
- **Temperature / Power / Clock**: Unavailable — the AMD Windows driver does
  not expose these through standard Python-accessible APIs on this hardware

## Relationship to Upstream

This project has **no intention to fork** the original. The changes here are
narrowly scoped to making the AMD provider work on Windows ROCm. AI agent-
assisted modifications were used during development, and the author does not
want to push AI-generated changes into allanmeng's original repository.

If the upstream maintainer wishes to incorporate any of these changes, they
are welcome to take whatever is useful.

## Status

| Metric | Source | Status |
|---|---|---|
| VRAM free / total | `torch.cuda.mem_get_info(0)` | ✅ |
| VRAM allocated / reserved | `torch.cuda.memory_allocated()` / `memory_reserved()` | ✅ |
| GPU utilisation | `typeperf` WDDM engine counters (max) | ✅ |
| Device name | `torch.cuda.get_device_name(0)` | ✅ |
| GPU core temperature | Driver does not expose via Windows API | ❌ |
| Power draw | Driver does not expose via Windows API | ❌ |
| Core clock | Driver does not expose via Windows API | ❌ |
| `rocm_smi_lib` dependency | Removed; no replacement needed | ✅ |
| Windows ROCm detection | `getattr(torch.version, 'roc', None)` + GPU name fallback | ✅ |

## Requirements

- ComfyUI (any recent version)
- [ROCm for Windows](https://rocm.docs.amd.com/) 6+ (tested on ROCm 7.2)
- PyTorch ROCm build (e.g. `pytorch 2.9.1+rocm7.2.1`)
- `psutil` (`pip install psutil`)

## Installation

Clone into ComfyUI `custom_nodes/`:

```cmd
cd ComfyUI/custom_nodes
git clone https://github.com/forkless/XPUSYS-Monitor-NG
cd XPUSYS-Monitor-NG
pip install -r requirements.txt
```

Restart ComfyUI. The plugin auto-detects AMD ROCm and loads the AMDProvider.

## License

Same as upstream — MIT.
