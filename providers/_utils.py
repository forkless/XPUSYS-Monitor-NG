"""
providers/_utils.py — Shared CPU/RAM utility functions.

Used by all providers to avoid duplicating Windows system-level reads.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Any, Dict, Tuple

logger = logging.getLogger("XPUSYSMonitor")


# ---------------------------------------------------------------------------
# Admin detection
# ---------------------------------------------------------------------------

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU info (model name + thread count)
# ---------------------------------------------------------------------------

def _get_cpu_info() -> Tuple[str, int]:
    """Return (model_name, logical_thread_count). Called once at startup."""
    model = ""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        winreg.CloseKey(key)
        model = " ".join(model.strip().split())   # collapse extra spaces
    except Exception:
        try:
            import platform
            model = platform.processor()
        except Exception:
            model = "Unknown CPU"
    threads = 0
    try:
        import psutil
        threads = psutil.cpu_count(logical=True) or 0
    except Exception:
        import os
        threads = os.cpu_count() or 0
    return model, threads


# ---------------------------------------------------------------------------
# CPU / RAM polling
# ---------------------------------------------------------------------------

def _read_cpu_ram_stats(psutil_ok: bool) -> Dict[str, Any]:
    """
    Poll CPU utilisation, frequency, and RAM stats.
    Returns a dict consumed by every provider's _poll().
    """
    out: Dict[str, Any] = {
        "cpu_pct": 0.0,
        "cpu_freq_ghz": 0.0,
        "ram_pct": 0.0,
        "ram_total_gb": 0.0,
        "ram_used_gb": 0.0,
        "ram_free_gb": 0.0,
        "commit_used_gb": 0.0,
        "commit_limit_gb": 0.0,
    }

    if psutil_ok:
        try:
            import psutil
            out["cpu_pct"] = psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq()
            if freq is not None:
                out["cpu_freq_ghz"] = round(freq.current / 1000.0, 2)
            mem = psutil.virtual_memory()
            out["ram_pct"] = mem.percent
            gb = 1024 ** 3
            out["ram_total_gb"] = round(mem.total / gb, 1)
            out["ram_used_gb"] = round(mem.used / gb, 1)
            out["ram_free_gb"] = round(mem.available / gb, 1)
        except Exception:
            logger.debug("XPUSYSMonitor: psutil read failed.", exc_info=True)
    else:
        # Fallback: GlobalMemoryStatusEx
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
            gb = 1024 ** 3
            out["ram_pct"] = float(state.dwMemoryLoad)
            out["ram_total_gb"] = round(state.ullTotalPhys / gb, 1)
            out["ram_free_gb"] = round(state.ullAvailPhys / gb, 1)
            out["ram_used_gb"] = round((state.ullTotalPhys - state.ullAvailPhys) / gb, 1)
            out["commit_used_gb"] = round(
                (state.ullTotalPageFile - state.ullAvailPageFile) / gb, 1
            )
            out["commit_limit_gb"] = round(state.ullTotalPageFile / gb, 1)
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# Windows Commit Charge (for RAM capsule)
# ---------------------------------------------------------------------------

def _read_commit_charge() -> Tuple[float, float]:
    """Return (commit_used_gb, commit_limit_gb) via GlobalMemoryStatusEx."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
        gb = 1024 ** 3
        used = (state.ullTotalPageFile - state.ullAvailPageFile) / gb
        limit = state.ullTotalPageFile / gb
        return round(used, 1), round(limit, 1)
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Windows Performance Data Helper (PDH) — GPU engine utilisation
#
# Uses pdh.dll via ctypes to query:
#   \GPU Engine(*)\Utilization Percentage
#
# This is the same source Task Manager uses — zero pip dependencies.
# ---------------------------------------------------------------------------

class _PdhQuery:
    """Thin ctypes wrapper around PDH API for GPU engine utilisation."""

    def __init__(self):
        self._pdh = None
        self._query = None
        self._counters: list = []
        self._ok = False

    def init(self) -> bool:
        if self._ok:
            return True
        try:
            self._pdh = ctypes.windll.pdh  # pdh.dll
            # PdhOpenQueryW
            self._pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
            self._pdh.PdhOpenQueryW.restype = wintypes.LONG

            # PdhAddEnglishCounterW
            self._pdh.PdhAddEnglishCounterW.argtypes = [
                ctypes.c_void_p, wintypes.LPCWSTR, wintypes.DWORD,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._pdh.PdhAddEnglishCounterW.restype = wintypes.LONG

            # PdhCollectQueryData
            self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            self._pdh.PdhCollectQueryData.restype = wintypes.LONG

            # PdhGetFormattedCounterValue
            self._pdh.PdhGetFormattedCounterValue.argtypes = [
                ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            self._pdh.PdhGetFormattedCounterValue.restype = wintypes.LONG

            # PdhRemoveCounter / PdhCloseQuery
            self._pdh.PdhRemoveCounter.argtypes = [ctypes.c_void_p]
            self._pdh.PdhRemoveCounter.restype = wintypes.LONG
            self._pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            self._pdh.PdhCloseQuery.restype = wintypes.LONG

            # Open a query
            self._query = ctypes.c_void_p()
            ret = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
            if ret != 0:  # ERROR_SUCCESS
                logger.warning(f"XPUSYSMonitor: PdhOpenQueryW failed — 0x{ret:08x}")
                return False

            # Enumerate GPU engine counters
            # The \GPU Engine(*)\Utilization Percentage counter path covers all GPU engines
            # (3D, Compute, Copy) on both NVIDIA and AMD GPUs.
            counter_path = "\\GPU Engine(*)\\Utilization Percentage"
            counter_buf = ctypes.c_void_p()
            ret = self._pdh.PdhAddEnglishCounterW(
                self._query, counter_path, 0, ctypes.byref(counter_buf)
            )
            if ret != 0:
                logger.warning(f"XPUSYSMonitor: PDH GPU counter not available — 0x{ret:08x}")
                self._pdh.PdhCloseQuery(self._query)
                return False

            self._counters = [counter_buf]
            self._ok = True
            logger.info("XPUSYSMonitor: PDH GPU utilisation counters OK.")
            return True

        except Exception as exc:
            logger.debug(f"XPUSYSMonitor: PDH init error — {exc}")
            return False

    def read_gpu_utilization(self) -> float:
        """Query total GPU utilisation % across all engines."""
        if not self._ok:
            return 0.0
        try:
            # Collect
            self._pdh.PdhCollectQueryData(self._query)

            # Read the wildcard counter — it aggregates across all GPU engines
            # We use PDH_FMT_DOUBLE (0x00000200) | PDH_FMT_NOCAP100 (0x00008000)
            fmt = 0x00008200  # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100

            class PDH_FMT_COUNTERVALUE_DOUBLE(ctypes.Structure):
                _fields_ = [
                    ("CStatus", wintypes.DWORD),
                    ("doubleValue", ctypes.c_double),
                ]

            for counter in self._counters:
                dwType = wintypes.DWORD(0)
                val = PDH_FMT_COUNTERVALUE_DOUBLE()
                ret = self._pdh.PdhGetFormattedCounterValue(
                    counter, fmt, ctypes.byref(dwType), ctypes.byref(val)
                )
                if ret == 0 and val.CStatus == 0:  # PDH_CSTATUS_VALID_DATA
                    return min(val.doubleValue, 100.0)

            return 0.0
        except Exception:
            return 0.0

    def close(self) -> None:
        if self._pdh and self._query:
            for c in self._counters:
                self._pdh.PdhRemoveCounter(c)
            self._pdh.PdhCloseQuery(self._query)
        self._ok = False


# ---------------------------------------------------------------------------
# typeperf-based GPU utilisation fallback
#
# Uses Windows built-in typeperf.exe (available since Vista) to query
# the \GPU Engine(*)\Utilization Percentage performance counter.
#
# typeperf avoids the quoting/escaping headaches of PowerShell -Command
# and is available on every Windows system with WDDM drivers.
#
# Output format (CSV):
#   "(PDH-CSV 4.0) (...)", "\\COMPUTER\GPU Engine(*)\Utilization Percentage"
#   "date time", "val1,val2,val3,..."
#
# We parse the second line, split the comma-separated values, and
# average them to get total GPU utilisation.
# ---------------------------------------------------------------------------

import csv as _csv
import subprocess as _subprocess


class _TypeperfGpuQuery:
    """GPU utilisation reader via typeperf (Windows built-in)."""

    def __init__(self):
        self._ok = False
        self._counter_path = "\\GPU Engine(*)\\Utilization Percentage"

    def init(self) -> bool:
        try:
            val = self._run_query()
            self._ok = val is not None
            if self._ok:
                logger.info(
                    f"XPUSYSMonitor: typeperf GPU counters OK "
                    f"(test={val:.1f}%)."
                )
            else:
                logger.warning(
                    "XPUSYSMonitor: typeperf GPU counters unavailable."
                )
            return self._ok
        except Exception as exc:
            logger.debug(f"XPUSYSMonitor: typeperf GPU init error — {exc}")
            return False

    def read_gpu_utilization(self) -> float:
        """Query total GPU utilisation % via typeperf."""
        if not self._ok:
            return 0.0
        try:
            val = self._run_query()
            return min(val, 100.0) if val is not None else 0.0
        except Exception:
            return 0.0

    def _run_query(self) -> float | None:
        """Run typeperf and parse the output. Returns average % or None."""
        try:
            r = _subprocess.run(
                ["typeperf", self._counter_path, "-sc", "1"],
                capture_output=True, text=True, timeout=10,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
            if r.returncode != 0:
                return None

            # Parse CSV output
            # Line 1: header with all counter paths (one per column after timestamp)
            # Line 2: data   e.g. "date time","0.000000","1.299634","0.000000",...
            lines = r.stdout.strip().splitlines()
            if len(lines) < 2:
                return None

            # Second line: each column is a separate counter value
            row = list(_csv.reader([lines[1]]))[0]
            if len(row) < 2:
                return None

            # Columns 1..N are individual GPU engine utilisation percentages
            values = []
            for v in row[1:]:
                v = v.strip()
                if v:
                    values.append(float(v))

            if not values:
                return None

            # Use the maximum value across all engines.
            # Averaging would dilute the signal (hundreds of engines
            # including idle video/copy/timer, only a few doing real work).
            # Under load the busy 3D/compute engine dominates; at idle all are ~0.
            peak = max(values)
            logger.debug(
                f"XPUSYSMonitor: typeperf read {len(values)} engines, "
                f"max={peak:.4f}%"
            )
            return peak

        except Exception:
            return None


__all__ = [
    "_is_admin",
    "_get_cpu_info",
    "_read_cpu_ram_stats",
    "_read_commit_charge",
    "_PdhQuery",
    "_TypeperfGpuQuery",
]
