"""Check that the installed PyTorch can actually compute on this GPU, before spending time on data and training.
Usage: /tmp/mra_env/bin/python kaggle/check_gpu.py
MRA's stack is pinned to torch 1.9.1 + CUDA 11.1 (mmcv-full 1.4.0 and mmdet3d 0.17.1 need it). That build contains kernels
for GPU architectures up to sm_86 (RTX 30 / A-series). Newer GPUs report a higher compute capability and fail with
"no kernel image is available for execution on the device". This script makes that failure explicit.
"""
import sys, torch
assert torch.cuda.is_available(), "CUDA not available in this environment"
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"GPU {name} | compute capability sm_{cap[0]}{cap[1]} | {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB | "
      f"torch {torch.__version__} built for {archs}")
try:
    x = torch.randn(2, 8, 32, 32, device="cuda")
    y = torch.nn.Conv2d(8, 8, 3, padding=1).cuda()(x); y.sum().backward(); torch.cuda.synchronize()
    import spconv.pytorch as spconv
    idx = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32, device="cuda")
    s = spconv.SparseConvTensor(torch.randn(2, 4, device="cuda"), idx, [8, 8, 8], 1)
    spconv.SubMConv3d(4, 4, 3, padding=1).cuda()(s).features.sum().item()
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    print(f"This PyTorch build ({torch.__version__}, {archs}) cannot run on {name} (sm_{cap[0]}{cap[1]}). The MRA software stack "
          "would have to be ported to a PyTorch release that supports this GPU; a config change cannot fix it.")
    sys.exit(1)
print("OK: dense and sparse convolutions run on this GPU")
