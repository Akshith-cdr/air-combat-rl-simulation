"""Launch rules used while training (all stages, 1-6).

The final policy was trained (26-30 Oct 2026) before the 500 m minimum missile
launch distance was added to ``weapon_config.py`` (2 Nov 2026). In training, a
missile could launch at any range below the stage's launch ceiling (100 m in
``configs/turning_A.json``), so close-range launches and hits shaped the reward.

Every training script calls ``apply_training_launch_rules()`` at startup.
Restoring these rules is what makes retraining reproduce the original
checkpoints bit-for-bit. The evaluation and visualization scripts do not
call it. They keep the current 500 m minimum and scripted 500-600 m firing,
which are the rules behind results/final_eval_1000/final_eval_1000.csv.
"""

import weapon_config

TRAINING_MIN_LAUNCH_DISTANCE_M = 0.0
# Launch ceiling used in training: the historical 100 m fire-range gate
# (Stage 5's configs/turning_A.json sets the same value itself).
TRAINING_LAUNCH_CEILING_M = 100.0


def apply_training_launch_rules():
    """Remove the post-training 500 m minimum for this process only."""
    weapon_config.MIN_MISSILE_LAUNCH_DISTANCE_M = TRAINING_MIN_LAUNCH_DISTANCE_M
    print(f'[training rules] minimum missile launch distance = {TRAINING_MIN_LAUNCH_DISTANCE_M:g} m '
          '(rules in force when the final model was trained)')
