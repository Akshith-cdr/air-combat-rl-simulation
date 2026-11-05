"""Historical launch rules for reproducing Stages 1-6.

Stages 1-6 were trained (26-30 Oct 2026) before the 500-1000 m launch envelope
was added to ``weapon_config.py``. In those stages a missile could launch at
any range below the stage's launch ceiling (100 m in ``configs/turning_A.json``),
so close-range launches and hits shaped the reward.

The Stage 1-6 training scripts call ``apply_training_launch_rules()`` at
startup. Restoring these rules is what makes retraining reproduce the original
checkpoints bit-for-bit.

Stage 7 (``train_fire_ppo.py``), evaluation and visualization do NOT call it.
They use the current envelope, 500 m <= range <= 1000 m, and the policy's own
learned fire output.
"""

import weapon_config

TRAINING_MIN_LAUNCH_DISTANCE_M = 0.0
# Launch ceiling used in Stages 1-6: the historical 100 m fire-range gate
# (Stage 5's configs/turning_A.json sets the same value itself).
TRAINING_LAUNCH_CEILING_M = 100.0


def apply_training_launch_rules():
    """Remove the 500 m minimum for this process only (Stages 1-6)."""
    weapon_config.MIN_MISSILE_LAUNCH_DISTANCE_M = TRAINING_MIN_LAUNCH_DISTANCE_M
    print(f'[training rules] minimum missile launch distance = {TRAINING_MIN_LAUNCH_DISTANCE_M:g} m '
          '(rules in force when Stages 1-6 were trained)')
