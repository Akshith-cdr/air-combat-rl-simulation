# UCAV Reinforcement Learning Project

An autonomous fighter aircraft (UCAV) learns, with PPO, to pursue and engage a manoeuvring
enemy aircraft in a 3D Panda3D air-combat simulation. The simulation includes radar and
guided missiles. The policy is trained through a seven-stage curriculum, and each stage builds
on the weights of the one before it. In the final model (Stage 7) **both flying and the
decision to fire are learned by PPO**.

## Results

1,000 seeded test episodes, Stage 7 model (deterministic flight, learned fire decision sampled
with a fixed seed per episode)
(`results/final_eval_1000_stage7/final_eval_1000.csv`):

| Metric | Result |
| --- | --- |
| Missile launches (by the learned fire network, all inside 500–1000 m) | **1,000 / 1,000** |
| Missile hits | **979 / 1,000 (97.9 %)** |
| Misses | 21 (17 missile expired, 4 hit the ground) |
| Boundary failures / timeouts | 0 / 0 |
| Launch range (min / median / max) | 500 / 857 / 1000 m |
| Median time to kill | 15.6 s |
| Rejected fire requests per episode (median) | 20 (blocked by the launch interlock) |

Rows are reproducible: rerunning the evaluation gives identical CSV rows. The rejected
requests are fire requests the interlock refused (wrong range or alignment, or the missile
already in flight). They never become launches, but they show that the learned fire decision
is not yet sharp at the edges of the envelope.

## System architecture

| Component | Role |
| --- | --- |
| **PPO policy** (learned) | One policy, two parts, both reading the same 19 observations. A Gaussian flight network flies the aircraft (pitch, roll, yaw, throttle; Stages 1–6). A Bernoulli fire network decides when to fire (Stage 7). |
| **Launch interlock** (`weapon_config.py`) | Rejects a fire request unless 500 m ≤ range ≤ 1000 m (both inclusive), cos(ATA) > 0.95, the missile is still mounted and the target is alive. It never fires by itself. |
| **Missile guidance** | Proportional navigation (gain 3) with line-of-sight acquisition |
| **Radar** | Tracks the enemy aircraft and missiles on the cockpit display |
| **Enemy aircraft** | Rule-based patrol AI that manoeuvres in 3D |

## Setup

Python **3.12.10** (64-bit), CPU.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Keep every model `.zip` together with its `_vecnormalize.pkl` file.

## 1. Test the final model

```bash
python evaluate_visualize_hybrid_combat.py --output results/final_eval_1000_stage7
```

This runs 1,000 seeded episodes (base seed 987654321) on the Stage 7 model,
`experiments/fire_policy/stage7_fire/final/candidate.zip`, and writes `final_eval_1000.csv`,
`summary.json`, `manifest.json` and `seed_list.json`. It takes about 35–40 minutes. The
policy's fire action goes to the environment unchanged; the launch interlock alone decides
whether a request becomes a launch. An episode ends on a hit, a chase-boundary exit, the
missile being wasted, or 2,000 steps (100 s).

The aircraft flies at the Gaussian mean (deterministic). The fire bit is **sampled** from the
learned fire probability, as in training, with the torch seed fixed per episode, so a rerun
gives identical rows. `--fire-mode deterministic` instead fires only when p(fire) ≥ 0.5. That
mode is weaker (33/50 hits vs 50/50 sampled in a 50-episode check): during training a launch
follows within a few steps whenever p(fire) is moderately high inside the envelope, so PPO
has no pressure to push p(fire) above 0.5.

For a quick check, run the first N episodes of the same seed list:

```bash
python evaluate_visualize_hybrid_combat.py --output results/quick_eval --episodes 20
```

To evaluate another model, add `--model-path <run>/final/candidate.zip` (Stage 7:
`experiments/fire_policy/<run-id>/...`, Stage 6: `experiments/flight_policy/<run-id>/...`).

## 2. Live 3D visualization

```bash
python visualize_hybrid_live.py
```

This loads the Stage 7 model by default. To replay a specific evaluation episode, pass its
`scenario_seed` from the CSV, for example the first row:

```bash
python visualize_hybrid_live.py --seed 4016774293
```

Other options: `--model-path <candidate.zip>`, `--scenario REC-001..REC-006`,
`--playback-rate 1.5`, `--no-vfx` (radar only) and `--visible-missiles` (simple missile
glyphs), `--deterministic` (fire only when p(fire) ≥ 0.5; the default samples the learned
fire decision, as in evaluation). Run `python visualize_hybrid_live.py --help` for the full list.

## 3. Train stage by stage

Run the stages in order. Each `--run-id` must be a new name. Times were measured on one CPU.

### Stages 1–4: basic flight, pursuit and the hybrid fire action

The original pipeline. Stage 4 output is supplied as
`models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip`.

```bash
# Stage 1 – fixed-enemy training (from random weights)
python train_ppo.py --timesteps 100000 --model-path models/stage1_fixed

# Stage 2 – randomized-enemy training
python train_ppo.py --randomized-enemy --timesteps 100000 --model-path models/stage2_randomized --init-model-path models/stage1_fixed.zip --init-vecnormalize-path models/stage1_fixed_vecnormalize.pkl

# Stage 3 – add the binary fire action (weight transfer)
python initialize_hybrid_ppo.py --source-model models/stage2_randomized.zip --source-vecnormalize models/stage2_randomized_vecnormalize.pkl --target-model models/stage3_hybrid_init.zip

# Stage 4 – hybrid flight-and-fire training
python train_hybrid_ppo.py --timesteps 100000 --init-model models/stage3_hybrid_init.zip --init-vecnormalize models/stage3_hybrid_init_vecnormalize.pkl --output-model models/stage4_final.zip
```

Test a Stage 4 model:

```bash
python test_final.py --model-path models/stage4_final.zip --vecnormalize-path models/stage4_final_vecnormalize.pkl --scenarios 100 --seed 41003 --policy-seed 41003
```

### Stage 5: guided-missile turning fine-tune (500k steps, about 45 min)

This continues PPO from Stage 4 with guided missiles enabled. Each episode starts from a
random heading and geometry. Configuration: `configs/turning_A.json` (seed 62001).

```bash
# From the supplied Stage 4 model (reproduces the final model exactly):
python train_turning_ppo.py --config configs/turning_A.json --run-id stage5_turning

# Or from your own Stage 4 model:
python train_turning_ppo.py --config configs/turning_A.json --run-id stage5_turning --init-model models/stage4_final.zip
```

The output is written to `experiments/turning_improvement/stage5_turning/final/`. The
supplied result of this stage is `experiments/turning_improvement/A_seed62001_run01/final/`.

### Stage 6: flight-policy refinement with the fire-control classifier (2,097,152 steps, about 3 h 10 min)

This trains the flight policy (four continuous controls) with a fresh optimizer. Weapon
release is handled by the fire-control classifier, so PPO concentrates on flight. The
reward favours closing on the target while aligned. A curriculum widens the start geometries: first ahead only, then ahead, left
and right, then all five. Configuration: `configs/flight_ppo_v2_seed63011.json` (seed 63011).

```bash
python train_flight_ppo.py --config configs/flight_ppo_v2_seed63011.json --enable-conjunction-v1 --run-id stage6_flight --source experiments/turning_improvement/stage5_turning/final
```

If you leave out `--source`, training starts from the supplied Stage 5 result. The final
model is written to `experiments/flight_policy/stage6_flight/final/candidate.zip`. During
training, development checkpoints are evaluated every 131,072 steps to track progress.

Then test it as in section 1, adding
`--model-path experiments/flight_policy/stage6_flight/final/candidate.zip`.

### Stage 7: learned firing (262,144 steps, about 17 min on one CPU)

Stage 6 flies well but fires through a classifier that was fitted when missiles launched at
about 100 m, so it never requests a launch inside the 500–1000 m envelope. Stage 7 replaces
that classifier with the policy's own Bernoulli fire head and trains it with PPO under the
current launch rules:

- **Start:** the Stage 6 final weights for flight and value. The fire decision gets its own
  small network (19 observations → 64 → 64 → 1, tanh), initialised to a 5 % fire probability
  everywhere, so firing is learned from scratch. Fresh optimizer, learning rate 1e-3, γ = 0.95,
  GAE λ = 0.8, no entropy bonus.
- **Flight frozen** (`"train_flight": false`): the Stage 6 flight tensors are not trained
  (checked bit-for-bit at every checkpoint) and the aircraft is flown at the Gaussian mean,
  exactly as in evaluation. Only the fire network and the value function learn.
- **Rewards (fire decision only):** +5 for a launch, +100 for a hit, −1 for every request with
  the wrong range or alignment, −0.01 per step and −25 for leaving the chase boundary. A
  request with valid geometry made after the missile has left is ignored, because the policy
  cannot observe that its missile is gone. The Stage 6 flight-shaping terms are left out: the
  flight is frozen, and that shaping (about +1 per step while chasing) would reward delaying
  the kill.
- **Episodes** end on a hit, a chase-boundary exit, a wasted missile or 2,000 steps. Start
  geometries cover all five families. Configuration: `configs/fire_ppo_stage7.json` (seed 71001).
- **Why these settings.** With γ = 0.99, λ = 0.98 and a 1M-step budget, the fire probability
  drifted instead of converging (kept as `experiments/fire_policy/stage7_fire_1M_unconverged/`).
  The shorter horizon (γ = 0.95, λ = 0.8) gives each fire request near-immediate credit
  (+5 valid launch, −1 rejected) instead of mixing it with the later, unrelated hit timing.

```bash
python train_fire_ppo.py --run-id stage7_fire
```

The final model is written to `experiments/fire_policy/stage7_fire/final/candidate.zip`
(the viewer and evaluator default). Checkpoints are saved every 65,536 steps in
`checkpoints/`, the per-episode training log is `episodes.csv`, and the PPO log is in `logs/`.
To continue an interrupted run:

```bash
python train_fire_ppo.py --resume experiments/fire_policy/stage7_fire/checkpoints/step_000000131072
```

Stage 7 does **not** use `training_rules.py`; it trains under the 500–1000 m envelope.

### Check that a retrained model matches the supplied one

```bash
python compare_checkpoints.py experiments/flight_policy/stage6_flight/final/candidate.zip experiments/flight_policy/flight_v2_seed63011_run03/final/candidate.zip
```

The script compares the weight tensors. The `.zip` files themselves always differ because
they contain save timestamps.

## Reproducibility

Retraining **Stages 5 and 6 from the supplied Stage 4 model** reproduces the final model
**bit-for-bit** on CPU with the package versions in `requirements.txt`. I checked this by
retraining and comparing weights with `compare_checkpoints.py`. The Stage 5 checkpoint at
120,832 steps and the Stage 6 checkpoint at 65,536 steps were identical to the original
runs, and so were all logged reward values up to those points.

- **Launch envelopes.** Stages 1–6 were trained with a close-range envelope (ceiling 100 m,
  no minimum), which `training_rules.py` restores in those training scripts. Stage 7,
  testing and the viewer use the current envelope in `weapon_config.py`:
  500 m ≤ range ≤ 1000 m and cos(ATA) > 0.95. Adding that envelope did not change Stage 5/6
  training: seeded Stage 6 training rollouts with the old and the new code are identical.
- **Stages 1–4** were trained earlier with an older version of the simulator. Retraining
  them gives a working Stage 4 model, but not the supplied one bit-for-bit, so Stages 5–6
  built on it will also differ. To reproduce the final model exactly, start Stage 5 from the
  supplied Stage 4 model.
- **The fire classifier** used only in Stage 6
  (`experiments/turning_improvement/fire_decision_conjunction_v1_run01/qualified_model.json`)
  works on two normalized observations: range (index 10) and alignment (index 11). It was
  fitted deterministically on 12,508 labelled engagement states from Stage 5 rollouts and
  is supplied with the project.
- The Stage 6 scripts check SHA-256 hashes of their inputs and will refuse to run if a pinned
  file has been edited. The pins are in
  `experiments/turning_improvement/fire_decision_integration_v1_20261027_01/runtime_artifacts_stage7.json`
  (`runtime_artifacts_500m.json` holds the pins from before the launch-envelope change).

## Folder layout

```
README.md, requirements.txt
training_rules.py                  historical launch rules for Stages 1-6
compare_checkpoints.py             weight comparison of two checkpoints

Simulation
  fighter_jet_env.py               Gymnasium env: 19 observations, 5 actions, dt 0.05 s
  updated_chase_jet_sim_v3.py      Panda3D aircraft physics, enemy AI, missiles
  projectile_motion.py             PN missile guidance
  radar.py, live_navigation_panel.py   radar display
  weapon_config.py                 launch envelope (500 m <= range <= 1000 m)
  missile_vfx.py                   missile and explosion effects
  *.glb                            aircraft and missile 3D models

Stages 1–4   train_ppo.py, initialize_hybrid_ppo.py, train_hybrid_ppo.py, hybrid_ppo.py,
             test_final.py, evaluate_ppo.py, evaluate_hybrid_ppo.py, visualize_hybrid_trajectory.py
Stage 5      train_turning_ppo.py, turning_experiment.py, turning_checkpoints.py, configs/turning_A.json
Stage 6      train_flight_ppo.py, experimental_flight_ppo_v1.py, evaluate_turning_ppo.py,
             turning_telemetry.py, experimental_fire_turn/, configs/flight_ppo_v*_seed63011.json
Stage 7      train_fire_ppo.py, learned_fire_policy.py, configs/fire_ppo_stage7.json
Evaluation   evaluate_visualize_hybrid_combat.py
Viewer       visualize_hybrid_live.py, live_visualization.py, live_diagnostics.py,
             visualization_telemetry.py, camera_observation/

models/                                       Stage 4 model (supplied)
experiments/turning_improvement/
  A_seed62001_run01/                          Stage 5 result (supplied)
  fire_decision_conjunction_v1_run01/         fitted fire classifier
  fire_decision_conjunction_v1_20261027_01/   classifier code
  fire_decision_integration_v1_20261027_01/   classifier loader + input hash pins
  fire_head_manifests_20261026_01/            pinned validation manifest
experiments/flight_policy/
  flight_v2_seed63011_run03/                  Stage 6 result (Stage 7 starts from it)
experiments/fire_policy/
  stage7_fire/                                Stage 7 result = FINAL MODEL (after training)
results/final_eval_1000_stage7/               the 1,000-episode evaluation (after training)
```
