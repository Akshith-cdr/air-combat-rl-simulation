# UCAV Reinforcement Learning Project

This project trains a fighter aircraft using PPO through progressive training stages: fixed-enemy training, randomized-enemy training, and hybrid flight-and-fire training.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training Sequence

Run these commands one by one in the given order.

### 1. Fixed-Enemy Training

Learns basic flight control, pursuit and alignment from random weights.

```bash
python train_ppo.py --timesteps 100000 --model-path models/stage1_fixed
```

### 2. Randomized-Enemy Training

Continues training using the weights learned in Stage 1.

```bash
python train_ppo.py --randomized-enemy --timesteps 100000 --model-path models/stage2_randomized --init-model-path models/stage1_fixed.zip --init-vecnormalize-path models/stage1_fixed_vecnormalize.pkl
```

### 3. Initialize the Hybrid Policy

Transfers the learned flight-control weights and adds a binary firing action.

```bash
python initialize_hybrid_ppo.py --source-model models/stage2_randomized.zip --source-vecnormalize models/stage2_randomized_vecnormalize.pkl --target-model models/stage3_hybrid_init.zip
```

### 4. Final Hybrid Training

```bash
python train_hybrid_ppo.py --timesteps 100000 --init-model models/stage3_hybrid_init.zip --init-vecnormalize models/stage3_hybrid_init_vecnormalize.pkl --output-model models/stage4_final.zip
```

## Test the Trained Model

```bash
python test_final.py --model-path models/stage4_final.zip --vecnormalize-path models/stage4_final_vecnormalize.pkl --scenarios 100 --seed 41003 --policy-seed 41003
```

The results are saved in the `tests` folder.

## Live 3D Visualization

```bash
python visualize_hybrid_live.py --model-path models/stage4_final.zip
```

## Generate Trajectory Graphs

```bash
python visualize_hybrid_trajectory.py --model-path models/stage4_final.zip --output-dir trajectory_results
```

## Included Final Model

To test the supplied pretrained final model directly:

```bash
python evaluate_hybrid_ppo.py
```

To visualize it:

```bash
python visualize_hybrid_live.py
```

> Keep every trained `.zip` model together with its matching `_vecnormalize.pkl` file.