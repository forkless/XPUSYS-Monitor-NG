"""
providers/amd.py — AMD GPU hardware provider for Windows ROCm.

VRAM free/total   : torch.cuda.mem_get_info(device)  (driver-level, ROCm 6+)
PyTorch stats     : torch.cuda.memory_allocated / memory_reserved
GPU load          : Windows PDH API (\\GPU Engine(*)\\Utilization Percentage)
GPU frequency     : Unavailable without vendor API -> 0
GPU temperature   : Unavailable without vendor API -> -1
Power             : Unavailable without vendor API -> -1 / False

No dependency on rocm_smi_lib — works with native Windows ROCm PyTorch.
GPU utilisation via PDH (pdh.dll, zero pip deps). Temperature/freq/power
return unavailable sentinels on Windows where no vendor driver API exists.
"""

import logging
import sys
from typing import Tuple

from .base import BaseGPUProvider, GPUSnapshot
from ._utils import _get_cpu_info, _read_cpu_ram_stats, _TypeperfGpuQuery, _AmdSmiGpuQuery, _is_admin

logger = logging.getLogger("XPUSYSMonitor")


# ---------------------------------------------------------------------------
# AMDProvider
# ---------------------------------------------------------------------------

class AMDProvider(BaseGPUProvider):
    """
    Hardware provider for AMD GPUs on Windows ROCm.

    Uses torch.cuda for VRAM and PyTorch allocator stats.
    Uses Windows PDH API for GPU engine utilisation.
    Does NOT require rocm_smi_lib.

    Temperature, core clock, and power return unavailable sentinels
    since no standard Python-accessible driver API exists on Windows.
    """

    GPU_VENDOR = "amd"

    def __init__(self, interval_ms: int = 1000):
        self._torch_ok    = False
        self._psutil_ok   = False
        self._device_index = 0
        self._is_admin     = _is_admin()
        self._cpu_model    = ""
        self._cpu_threads  = 0

        self._check_torch()
        self._check_psutil()

        # Windows GPU utilisation — amdsmi (official) -> typeperf (fallback)
        # PDH has wildcard-counter issues with AMD drivers, so skip it.
        self._pdh_ok = False
        self._as_gpu = _AmdSmiGpuQuery()
        self._as_gpu_ok = self._as_gpu.init()
        self._tp_gpu = _TypeperfGpuQuery()
        self._tp_gpu_ok = self._tp_gpu.init() if not self._as_gpu_ok else False

        # BaseGPUProvider.__init__ starts the polling thread — call last
        super().__init__(interval_ms=interval_ms)

        logger.info(
            f"XPUSYSMonitor: AMDProvider started "
            f"(torch={self._torch_ok}, pdh={self._pdh_ok})"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _check_torch(self) -> None:
        """Check if torch.cuda is available (ROCm PyTorch on Windows)."""
        try:
            import torch
            if torch.cuda.is_available():
                self._torch_ok = True
                logger.info(
                    f"XPUSYSMonitor: torch.cuda OK (AMD ROCm), "
                    f"device count={torch.cuda.device_count()}, "
                    f"device={torch.cuda.get_device_name(self._device_index)!r}"
                )
                # Force CUDA context init — some ROCm builds
                # defer context creation until first GPU operation.
                torch.cuda.synchronize(self._device_index)
            else:
                logger.warning("XPUSYSMonitor: torch.cuda not available.")
        except Exception as exc:
            logger.warning(f"XPUSYSMonitor: torch import error — {exc}")

    def _check_psutil(self) -> None:
        try:
            import psutil
            psutil.cpu_percent(interval=None)
            self._psutil_ok = True
            self._cpu_model, self._cpu_threads = _get_cpu_info()
            logger.info(
                f"XPUSYSMonitor: psutil OK — CPU={self._cpu_model!r}, "
                f"threads={self._cpu_threads}"
            )
        except Exception as exc:
            logger.warning(f"XPUSYSMonitor: psutil not available — {exc}")

    # ------------------------------------------------------------------
    # Hardware reads
    # ------------------------------------------------------------------

    def _read_device_name(self) -> str:
        """Return the GPU model name via torch.cuda."""
        if self._torch_ok:
            try:
                import torch
                return torch.cuda.get_device_name(self._device_index)
            except Exception:
                pass
        return "AMD GPU (ROCm)"

    def _read_vram(self) -> Tuple[float, float, float]:
        """
        Return (free_gb, total_gb, driver_used_gb) via torch.cuda.mem_get_info.

        REPLACES: upstream rocm_smi.getMemFreeVdev(0),
                  rocm_smi.getMemSizeVdev(0),
                  rocm_smi.getMemUsedVdev(0)

        torch.cuda.mem_get_info() returns (free_bytes, total_bytes) from
        the AMD HIP driver on ROCm 6+ for Windows. This is the same function
        used by NVIDIA CUDA — AMD ROCm's HIP runtime implements the same
        CUDA API surface, so it works without any AMD-specific library.

        Falls back to get_device_properties if mem_get_info is unavailable.
        Forces CUDA context init (torch.cuda.synchronize) before reading;
        some ROCm builds defer HIP context creation until the first GPU
        operation, and mem_get_info() returns (0, 0) without an active
        context.
        """
        if not self._torch_ok:
            return 0.0, 0.0, 0.0

        import torch

        # Force CUDA context init — required by some ROCm builds
        # before device queries return valid data.
        try:
            torch.cuda.synchronize(self._device_index)
        except Exception:
            pass

        gb = 1024 ** 3

        # Primary: mem_get_info — driver-level free/total
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._device_index)
            free_gb   = free_bytes  / gb
            total_gb  = total_bytes / gb
            used_gb   = max(0.0, total_gb - free_gb)
            logger.debug(
                f"XPUSYSMonitor: mem_get_info OK "
                f"(free={free_gb:.1f}G, total={total_gb:.1f}G)"
            )
            return free_gb, total_gb, used_gb
        except Exception as exc:
            logger.debug(f"XPUSYSMonitor: mem_get_info failed — {exc}")

        # Fallback 1: total from device properties
        try:
            total_gb = torch.cuda.get_device_properties(
                self._device_index
            ).total_memory / gb
            logger.debug(
                f"XPUSYSMonitor: get_device_properties OK "
                f"(total={total_gb:.1f}G)"
            )
            return 0.0, total_gb, 0.0
        except Exception as exc:
            logger.debug(f"XPUSYSMonitor: get_device_properties failed — {exc}")

        return 0.0, 0.0, 0.0

    def _read_torch_stats(self) -> Tuple[float, float]:
        """Return (allocated_gb, reserved_gb) from torch.cuda allocator."""
        if not self._torch_ok:
            return 0.0, 0.0
        try:
            import torch
            idx = self._device_index
            gb = 1024 ** 3
            return (
                torch.cuda.memory_allocated(idx) / gb,
                torch.cuda.memory_reserved(idx) / gb,
            )
        except Exception:
            return 0.0, 0.0

    def _read_gpu_load(self) -> float:
        """
        Return GPU utilisation %.

        REPLACES: upstream rocm_smi.getGpuBusyVdev(0)

        Two-layer fallback chain:
          1. amdsmi       — official AMD SMI library (pip install amdsmi).
                            Bypasses WDDM, talks directly to the AMD driver.
                            Gracefully skipped on Windows because the PyPI
                            package searches for libamd_smi.so (Linux-only).
          2. typeperf     — Windows built-in (available since Vista).
                            Reads \\GPU Engine(*)\\Utilization Percentage
                            via WDDM performance counters. Returns CSV with
                            one column per engine instance (3D, Compute,
                            Copy, Video, Timer, etc.). We use max() across
                            all engines — averaging dilutes the signal
                            across hundreds of idle engine types.
        """
        if self._as_gpu_ok:
            return self._as_gpu.read_gpu_utilization()
        if self._tp_gpu_ok:
            return self._tp_gpu.read_gpu_utilization()
        return 0.0

    def _read_gpu_freq_mhz(self) -> float:
        """
        GPU core frequency in MHz.

        REPLACES: upstream rocm_smi.getSingleClockSpeed(0)

        The AMD WDDM driver on Windows does not expose GPU core clock
        through any standard Python-accessible interface (no PDH counter,
        no WMI class, no torch.cuda equivalent). Returns 0 (unavailable
        sentinel matching the GPUSnapshot contract default).
        """
        return 0.0

    def _read_gpu_temp_c(self) -> float:
        """
        GPU core temperature in C.

        REPLACES: upstream rocm_smi.getTempVdev(0)

        The AMD WDDM driver on the tested configuration (RX 9070 XT,
        ROCm 7.2, Windows) does not register a GPU temperature performance
        counter. Tested: typeperf -q "GPU Adapter" returned "object not
        found". Returns -1 (unavailable sentinel). Some AMD cards on newer
        driver versions or different Windows builds may expose this through
        WMI or PDH — this is hardware/driver-dependent.
        """
        return -1.0

    def _read_power(self) -> Tuple[float, float, bool]:
        """
        Return (power_w, tgp_w, power_available).

        REPLACES: upstream rocm_smi.getPowerVdev(0) and
                  rocm_smi.getPowerCapVdev(0)

        GPU power monitoring is not exposed through Windows standard APIs
        on the tested AMD driver. Returns (-1.0, 0.0, False) — the
        GPUSnapshot power_available=False tells the frontend to grey out
        the PWR capsule.
        """
        return -1.0, 0.0, False

    # ------------------------------------------------------------------
    # Poll — called by BaseGPUProvider._loop() every interval
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Collect all hardware metrics and push a fresh GPUSnapshot."""
        snap = GPUSnapshot(gpu_vendor=self.GPU_VENDOR)
        snap.is_admin = self._is_admin

        if not self._torch_ok:
            snap.error = "AMD ROCm (torch.cuda) unavailable"
        else:
            try:
                snap.device_name = self._read_device_name()

                # VRAM — driver level via torch.cuda.mem_get_info
                free_gb, total_gb, driver_used_gb = self._read_vram()
                snap.vram_total_gb       = total_gb
                snap.vram_free_gb        = free_gb
                snap.vram_driver_used_gb = driver_used_gb

                # PyTorch allocator stats
                snap.vram_allocated_gb, snap.vram_reserved_gb = self._read_torch_stats()

                # GPU metrics
                snap.gpu_load_pct = self._read_gpu_load()
                snap.gpu_freq_mhz = self._read_gpu_freq_mhz()
                snap.gpu_temp_c   = self._read_gpu_temp_c()

                # Power
                snap.power_w, snap.tgp_w, snap.power_available = self._read_power()

            except Exception as exc:
                logger.debug(f"XPUSYSMonitor: AMDProvider poll error — {exc}")
                snap.error = str(exc)

        # CPU / RAM — always collected regardless of GPU state
        sys_stats = _read_cpu_ram_stats(self._psutil_ok)
        snap.cpu_pct         = sys_stats.get("cpu_pct",         0.0)
        snap.cpu_freq_ghz    = sys_stats.get("cpu_freq_ghz",    0.0)
        snap.cpu_model       = self._cpu_model
        snap.cpu_threads     = self._cpu_threads
        snap.ram_pct         = sys_stats.get("ram_pct",         0.0)
        snap.ram_total_gb    = sys_stats.get("ram_total_gb",    0.0)
        snap.ram_used_gb     = sys_stats.get("ram_used_gb",     0.0)
        snap.ram_free_gb     = sys_stats.get("ram_free_gb",     0.0)
        snap.commit_used_gb  = sys_stats.get("commit_used_gb",  0.0)
        snap.commit_limit_gb = sys_stats.get("commit_limit_gb", 0.0)

        self._update_snapshot(snap)


__all__ = ["AMDProvider"]
