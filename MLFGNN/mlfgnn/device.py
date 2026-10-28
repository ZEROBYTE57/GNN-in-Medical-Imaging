"""Device resolution -- one place that decides CPU vs CUDA.

``"auto"`` picks CUDA when a GPU is visible and falls back to CPU otherwise,
so the same command works on the laptop, a workstation and a CPU-only CI run.
The resolved device is logged once at the start of a run and written into the
run's provenance, because "which hardware" is part of reproducibility whenever
non-deterministic CUDA kernels are involved.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(spec: str = "auto"):
    """Return a ``torch.device`` for ``spec`` (``auto`` | ``cpu`` | ``cuda`` | ``cuda:N``)."""
    import torch

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA not available -- falling back to CPU")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {spec!r} requested but torch reports no CUDA. "
            f"Installed torch is {torch.__version__}; a '+cpu' build cannot see the GPU."
        )
    return dev


def describe_device(dev) -> dict:
    """Provenance record for the results directory."""
    import torch

    info = {"device": str(dev), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if dev.type == "cuda":
        idx = dev.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 1e9, 2),
                "cuda_version": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return info
