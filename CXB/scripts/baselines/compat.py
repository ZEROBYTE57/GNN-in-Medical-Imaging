"""Compatibility shims for running older baseline code on PyTorch 2.x."""
import torch


def symeig(a, eigenvectors=False, upper=True):
    """Stand-in for torch.symeig (removed after PyTorch 1.9), used by BrainIB V1 and V2.

    BrainIB only uses the eigenvalues ([0]) of a trace-normalised Gram matrix, so this
    computes eigenvalues only, in float64, after exact symmetrisation. The GPU solver fails
    on the near-repeated eigenvalues these 32x32 matrices can have, so it falls back to the
    CPU. Eigenvectors are never used by BrainIB and are returned empty."""
    a64 = a.double()
    if not torch.isfinite(a64).all():
        raise FloatingPointError(
            "BrainIB's kernel matrix contains NaN/inf: the batch embeddings are NaN or "
            "(near-)identical, so its bandwidth sigma is about 0.")
    a64 = (a64 + a64.transpose(-1, -2)) / 2
    try:
        vals = torch.linalg.eigvalsh(a64)
    except RuntimeError:
        vals = torch.linalg.eigvalsh(a64.cpu()).to(a.device)
    return vals.to(a.dtype), torch.empty(0, device=a.device)


def install():
    torch.symeig = symeig
