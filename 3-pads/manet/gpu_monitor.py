"""Background GPU-utilisation sampling.

This is deliberately separate from profile_model()'s torch.profiler usage in
train.py: torch.profiler answers "how long did each operator take", which
says nothing about what fraction of wall-clock time the accelerator was
actually busy. GPU utilisation (%) -- the number nvidia-smi reports as
"GPU-Util" -- requires polling NVML directly, which is what this module does.

Needs nvidia-ml-py (`pip install nvidia-ml-py`, importable as `pynvml`) inside
the training container. If it isn't installed, GPUUtilMonitor degrades to a
no-op that returns None for mean_utilization() rather than failing the run --
a missing utilisation number should not crash a training job that otherwise
doesn't need it.
"""
import threading

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class GPUUtilMonitor:
    """Samples GPU utilisation (%) on a background thread for the duration of
    a `with` block.

    Usage:
        with GPUUtilMonitor(device_index=0) as mon:
            trainer.fit(model, ...)
        print(mon.summary())
    """

    def __init__(self, device_index=0, interval_s=0.5):
        self.device_index = device_index
        self.interval_s = interval_s
        self.available = _PYNVML_AVAILABLE
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._handle = None

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(util.gpu)
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def start(self):
        if not self.available:
            print(
                "[GPUUtilMonitor] pynvml not installed -- GPU utilisation will "
                "not be recorded for this run. Install with "
                "'pip install nvidia-ml-py' inside the training container."
            )
            return self
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            print(f"[GPUUtilMonitor] nvmlInit failed ({exc}); GPU utilisation "
                  "will not be recorded for this run.")
            self.available = False
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5)
            self._thread = None
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def mean_utilization(self):
        """Mean GPU utilisation (%) over the monitored window, or None if no
        samples were collected (pynvml unavailable, nvmlInit failed, or the
        monitored window was shorter than interval_s)."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def summary(self):
        return {
            "gpu_util_mean_pct": self.mean_utilization(),
            "gpu_util_n_samples": len(self._samples),
            "gpu_util_interval_s": self.interval_s,
            "pynvml_available": self.available,
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
