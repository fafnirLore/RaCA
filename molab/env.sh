# Shared settings for the molab scripts (sourced, not run).
export MRA_ENV=${MRA_ENV:-/tmp/mra_env}
export PY=${PY:-$MRA_ENV/bin/python}
export GDOWN=${GDOWN:-$MRA_ENV/bin/gdown}
export OUT_ROOT=${OUT_ROOT:-/tmp/mra_runs}
# PyTorch >= 2.6 loads checkpoints with weights_only=True by default; mmcv 1.x checkpoints (and SGN-T's) also hold
# plain Python metadata, so restore the old behaviour for these trusted files.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Quieter logs: the deprecation warnings below come from the old OpenMMLab code on new PyTorch / NumPy and are harmless.
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning}
