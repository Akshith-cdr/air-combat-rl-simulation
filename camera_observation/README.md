# Camera observation scheduling

This non-weapon task learns when to record a geometric camera sighting. The
original PPO flight controller stays frozen; camera actions never enter the
aircraft or firing action lanes.

See [the task specification and measured results](../docs/CAMERA_OBSERVATION_TASK.md).

```powershell
.venv\Scripts\python.exe visualize_hybrid_live.py --camera-observation --seed 41003
```

The trained camera checkpoint and evaluation are separate under
`camera_observation_runs/run_20261102_01/`.
