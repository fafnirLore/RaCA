_base_ = ['../../kaggle/configs/raca_c2c3c4.py']
# molab run: RaCA, training-only components C2 + C3 + C4 on one RTX PRO 6000 Blackwell (96 GB).
# Inherits kaggle/configs/raca_c2c3c4.py unchanged (data, schedule, learning rate, SGN-T warm start, validation, early
# stopping); only the output folder differs. Compare with molab/configs/mra_before.py run on the same molab stack.
# Overrides: bash molab/train.sh molab/configs/raca_c2c3c4.py raca_c2c3c4 --cfg-options runner.max_epochs=3   (load_from=None for
# MRA's own initialisation; use it for the baseline too if you do).
work_dir = '/tmp/mra_runs/raca_c2c3c4'
