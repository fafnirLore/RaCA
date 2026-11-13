"""Fallback for the three torch_scatter functions MRA calls, written with plain PyTorch.

Only used when molab/install.sh cannot install (or cannot run on the GPU) the real torch-scatter wheel; install.sh
then adds this folder to the environment's path. Results match torch_scatter for the calls MRA makes
(floating-point `src`, 1-D `index`, groups without elements give 0). One difference: when several elements of a
group tie for the maximum, torch_scatter sends the gradient to one of them and this version splits it evenly.
"""
import torch

__version__ = "0.0-mra-shim"


def _prep(src, index, dim, dim_size):
    dim = dim % src.dim()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape); shape[dim] = dim_size
    if index.dim() == 1 and src.dim() > 1:                     # broadcast a 1-D index along `dim`
        view = [1] * src.dim(); view[dim] = -1
        index = index.view(view).expand_as(src)
    return dim, shape, index


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    assert out is None, "out= is not supported by this fallback"
    dim, shape, index = _prep(src, index, dim, dim_size)
    return src.new_zeros(shape).scatter_add_(dim, index, src)


scatter_sum = scatter_add


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    assert out is None, "out= is not supported by this fallback"
    dim, shape, index = _prep(src, index, dim, dim_size)
    total = src.new_zeros(shape).scatter_add_(dim, index, src)
    count = src.new_zeros(shape).scatter_add_(dim, index, torch.ones_like(src)).clamp_(min=1)
    return total / count if src.is_floating_point() else torch.div(total, count, rounding_mode="floor")


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    """Returns (values, None): MRA only uses the values, so the argmax is not computed."""
    assert out is None, "out= is not supported by this fallback"
    dim, shape, index = _prep(src, index, dim, dim_size)
    return src.new_zeros(shape).scatter_reduce(dim, index, src, reduce="amax", include_self=False), None
