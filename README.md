# Air Combat RL Simulation

A 3D fighter-jet air combat simulation built as the foundation for a reinforcement learning project.

## Current Features

* F-15 fighter jet simulation
* Enemy fighter aircraft
* Aircraft movement and maneuvering
* Missile launching and projectile simulation
* Missile collision detection
* Enemy destruction and mission success
* 10 km combat boundary
* Basic 3D environment with clouds

## Controls

| Key   | Action                     |
| ----- | -------------------------- |
| ↑ / ↓ | Pitch                      |
| ← / → | Roll                       |
| Q / E | Yaw                        |
| W / S | Increase / decrease speed  |
| M     | Fire missile               |
| F9    | Run missile collision test |
| Esc   | Exit                       |

## Tech Stack

* Python
* Panda3D

## Project Goal

The simulation will be extended into an **MDP-based reinforcement learning environment** for training an agent to maneuver the fighter jet and engage the enemy using missiles.

Future work will include:

* MDP implementation
* RL environment
* Reward design and validation
* RL algorithm selection
* Agent training and evaluation

## Running the Simulation

```bash
python updated_chase_jet_sim_v3.py
```

## Status

**Current:** Fighter-jet simulation and missile system implemented.

**Next:** MDP implementation and RL environment development.
