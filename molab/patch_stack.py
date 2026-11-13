# Make the OpenMMLab 1.x stack (mmcv 1.7.2 lite, mmdet 2.14.0, mmseg 0.14.1, mmdet3d 0.17.1) importable on PyTorch 2.7.
# No MRA file is touched; only the installed third-party packages. Idempotent.
# Usage: python molab/patch_stack.py /path/to/mmdetection3d
#
# 1. mmcv/_ext.py placeholder. 'mmcv' (lite) ships the Python side of mmcv.ops but not the compiled library mmcv._ext,
#    so `import mmcv.ops` (done by mmdet at import time) fails. MRA calls none of these operators (its sparse layers
#    come from spconv, its scatter ops from torch-scatter), so a placeholder that raises when an operator is actually
#    called keeps the imports working - the same idea as kaggle/patch_mmdet3d.py.
# 2. Version asserts. mmdet 2.14 / mmseg 0.14 / mmdet3d 0.17 refuse mmcv > 1.4.0; mmcv 1.7.2 is API-compatible for
#    what MRA uses and is the 1.x release that supports PyTorch 2.
import os, re, sys, importlib.util


def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    assert spec and spec.origin, f"{name} is not installed"
    return os.path.dirname(spec.origin)


STUB = '''# Placeholder for mmcv's compiled operators (written by molab/patch_stack.py): the lite 'mmcv' package has none
# and MRA uses none. Any operator that is actually called raises instead of silently doing something else.
def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    def _not_built(*args, **kwargs):
        raise NotImplementedError("mmcv._ext." + name + " is not compiled in this environment (mmcv lite)")
    _not_built.__name__ = name
    return _not_built
'''

mmcv_dir = pkg_dir("mmcv")
ext = os.path.join(mmcv_dir, "_ext.py")
compiled = [f for f in os.listdir(mmcv_dir) if f.startswith("_ext.") and f != "_ext.py"]
if compiled:
    print("mmcv._ext is compiled (mmcv-full) - no placeholder written")
else:
    open(ext, "w").write(STUB)
    print("placeholder:", ext)


def relax(path, names):
    """Raise the upper bound of the version asserts in an OpenMMLab __init__.py."""
    src = open(path).read(); new = src
    for n in names:
        new = re.sub(rf"^({n}\s*=\s*)['\"][^'\"]+['\"]", r"\g<1>'9.9.9'", new, flags=re.M)
    if new != src:
        open(path, "w").write(new)
    print("version bound relaxed:" if new != src else "version bound already relaxed:", path)


relax(os.path.join(pkg_dir("mmdet"), "__init__.py"), ["mmcv_maximum_version"])
relax(os.path.join(pkg_dir("mmseg"), "__init__.py"), ["MMCV_MAX"])
m3d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(pkg_dir("mmdet3d"))
relax(os.path.join(m3d, "mmdet3d", "__init__.py"),
      ["mmcv_maximum_version", "mmdet_maximum_version", "mmseg_maximum_version"])
