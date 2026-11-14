"""Single-run pursuit-control pipeline used for the revised paper.

Only the pursuit flight is learned. Launch requests come from a fixed rule,
the launch gate and the PN missile are fixed, and the target never reacts.

Modules
-------
config      frozen constants: launch band, collision radius, reward weights
scenarios   initial-condition sampler (training) and evaluation suites
env         PursuitEnv: 4 flight actions, scripted fire rule, collision check
baselines   pure-pursuit and lead-pursuit controllers
provenance  git hash / clean-tree check / package versions (read-only git)
train       PPO from scratch, one run per seed, with learning curves
evaluate    any controller on any suite; per-sector tables, failure trajectories
"""
