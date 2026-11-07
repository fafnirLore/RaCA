# Make mmdet3d v0.17.1 installable without compiling its CUDA operators.
# MRA does not use these operators; placeholder modules keep `import mmdet3d` working.
# Usage: python kaggle/patch_mmdet3d.py /path/to/mmdetection3d
import os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else "."
setup_path = os.path.join(root, "setup.py")
src = open(setup_path).read()
pairs = re.findall(r"name='(\w+)',\s*module='([\w.]+)'", src)
if "ext_modules=[] and [" not in src:
    src = src.replace("ext_modules=[", "ext_modules=[] and [", 1)
    open(setup_path, "w").write(src)
STUB = '''# Placeholder: this compiled operator was not built (MRA does not use it).
def __getattr__(name):
    def _not_built(*args, **kwargs):
        raise NotImplementedError(__name__ + "." + name + " was not compiled")
    return _not_built
'''
for name, module in pairs:
    path = os.path.join(root, *module.split("."), name + ".py")
    open(path, "w").write(STUB)
    print("placeholder:", module + "." + name)
print("patched setup.py; extensions skipped:", len(pairs))
