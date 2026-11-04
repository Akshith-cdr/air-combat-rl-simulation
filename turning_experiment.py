"""Opt-in turning experiment contracts. Importing this module never trains."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from panda3d.core import Quat, Vec3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fighter_jet_env import FighterJetEnv
from hybrid_ppo import HybridActorCriticPolicy

BASE_MODEL = Path('models/ppo_ucav_100k_19d_hybrid_fire_randomized_training.zip')
BASE_STATS = BASE_MODEL.with_name(BASE_MODEL.stem+'_vecnormalize.pkl')
BASE_HASHES = {BASE_MODEL.as_posix(): 'fd2dc4bfd404f0f65bd362043c96d5e595af416ba263f628f7dc66f0b24c884c',
               BASE_STATS.as_posix(): '793d0862782bd6c126da53926dcc43159268f904a16a6779144296d0ca05905c'}
RUN_ROOT = Path('experiments/turning_improvement')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_baseline():
    actual = {p: sha256(p) for p in BASE_HASHES}
    if actual != BASE_HASHES:
        raise ValueError('Baseline model/statistics hashes differ from the protected original')
    return actual


@dataclass(frozen=True)
class ExperimentConfig:
    variant: str = 'A'
    seed: int = 62001
    total_timesteps: int = 500000
    device: str = 'cpu'
    n_steps: int = 2048
    batch_size: int = 64
    learning_rate: float = 0.0001
    gamma: float = 0.99
    checkpoint_every: int = 20000
    max_episode_steps: int = 2000
    bearing_progress_weight: float = 0.0
    update_normalization: bool = False
    player_headings: tuple = (0, 60, 120, 180, 240, 300)
    guided_missiles: bool = True
    launch_range_m: float = 100.0

    def __post_init__(self):
        if self.variant not in ('A', 'B'):
            raise ValueError('variant must be A or B; no unvalidated maneuver penalty is supported')
        for name in ('seed', 'total_timesteps', 'n_steps', 'batch_size', 'checkpoint_every', 'max_episode_steps'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name=='seed' else 1):
                raise ValueError(f'{name} must be an integer in its positive range')
        if self.n_steps < 2 or self.batch_size < 2 or self.n_steps % self.batch_size:
            raise ValueError('n_steps must be >=2 and divisible by batch_size >=2')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .001:
            raise ValueError('learning_rate must be in (0, .001]')
        if not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError('gamma must be in (0,1)')
        if not math.isfinite(self.bearing_progress_weight) or not 0 <= self.bearing_progress_weight <= .5:
            raise ValueError('bearing_progress_weight must be in [0,.5]')
        if (self.variant=='A' and self.bearing_progress_weight!=0) or (self.variant=='B' and self.bearing_progress_weight<=0):
            raise ValueError('A requires zero bearing weight; B requires a positive weight')
        if type(self.update_normalization) is not bool or type(self.guided_missiles) is not bool:
            raise ValueError('normalization/guidance flags must be booleans')
        if not self.player_headings or any(not math.isfinite(h) or not 0<=h<360 for h in self.player_headings):
            raise ValueError('player_headings must contain finite headings in [0,360)')
        if not math.isfinite(self.launch_range_m) or self.launch_range_m != 100:
            raise ValueError('This turning experiment preserves the 100 m launch gate')
        if not isinstance(self.device,str) or (self.device not in ('cpu','cuda') and
                not (self.device.startswith('cuda:') and self.device[5:].isdigit())):
            raise ValueError('device must be cpu, cuda or cuda:N')


def load_config(path):
    data = json.loads(Path(path).read_text())
    if 'player_headings' in data:
        data['player_headings'] = tuple(data['player_headings'])
    return ExperimentConfig(**data)


def validate_device(device):
    target = torch.device(device)
    if target.type=='cuda' and (not torch.cuda.is_available() or
                               (target.index is not None and target.index>=torch.cuda.device_count())):
        raise ValueError(f'Requested device {device} is unavailable; select --device cpu')
    return target


@dataclass(frozen=True)
class TurningScenario:
    family: str
    player_heading: float
    distance: float
    bearing: float  # Clockwise, right-positive in the initial horizontal body frame.
    player_speed: float
    enemy_speed: float
    relative_enemy_heading: float  # Clockwise relative to player's compass heading.
    altitude: float
    delta_altitude: float

    def mirror(self):
        families = {'left':'right','right':'left','behind_left':'behind_right',
                    'behind_right':'behind_left','ahead':'ahead'}
        return replace(self, family=families[self.family], bearing=-self.bearing,
                       relative_enemy_heading=-self.relative_enemy_heading)


def sample_scenario(rng, headings=(0,60,120,180,240,300), family=None):
    family = family or str(rng.choice(['ahead','left','right','behind_left','behind_right']))
    bounds = {'ahead':(-15,15), 'right':(30,80), 'left':(-80,-30),
              'behind_right':(100,160), 'behind_left':(-160,-100)}
    return TurningScenario(family, float(rng.choice(headings)), float(rng.uniform(1500,5000)),
                           float(rng.uniform(*bounds[family])), float(rng.uniform(160,260)),
                           float(rng.uniform(140,240)), float(rng.uniform(-150,150)),
                           float(rng.uniform(1000,2200)), float(rng.uniform(-500,500)))


def apply_scenario(env, scenario):
    """Diagnostic/training initial conditions only; legacy enemy reset is retained."""
    q = Quat()
    q.setHpr(Vec3(scenario.player_heading,0,0))
    angle = math.radians(scenario.bearing)
    offset = q.xform(Vec3(math.sin(angle)*scenario.distance,math.cos(angle)*scenario.distance,
                         scenario.delta_altitude))
    position = Vec3(0,0,scenario.altitude)+offset
    # Enemy reset historically treats heading as clockwise for velocity and
    # assigns it directly to Panda yaw. Preserve that inconsistency explicitly.
    enemy_heading = (-scenario.player_heading+scenario.relative_enemy_heading)%360
    env._reset_simulation_state({'scenario_id':'TURNING-'+scenario.family, 'seed':0,
        'enemy_initial_position':tuple(position), 'enemy_initial_speed':scenario.enemy_speed,
        'enemy_initial_heading':enemy_heading,'enemy_initial_altitude':float(position.z)})
    p = env.sim.player
    p.position = Vec3(0,0,scenario.altitude)
    p.roll, p.pitch, p.yaw = 0,0,scenario.player_heading
    p.velocity = q.xform(Vec3(0,scenario.player_speed,0))
    p.current_speed=p.target_speed=p.airspeed=scenario.player_speed
    p.altitude=scenario.altitude
    p.sync_graphics()
    env._prev_range = env._distance_to_target()
    env._prev_cos_ata = env._current_cos_ata()
    return env._get_observation()


def bearing_potential(error):
    wrapped = (float(error)+180)%360-180
    return 1-abs(wrapped)/180


def bearing_progress_reward(previous, current, gamma, weight, terminated=False):
    """Discount-consistent potential shaping; closed loops cannot farm reward.

    True terminal potential is zero. Time-limit truncations retain the next
    potential because SB3 bootstraps their terminal observation.
    """
    return weight*(gamma*(0 if terminated else bearing_potential(current))-bearing_potential(previous))


class TurningEnv(FighterJetEnv):
    def __init__(self, config):
        self.experiment_config=config
        self.pending_mirror=None
        super().__init__(hybrid_fire=True,missile_guidance=config.guided_missiles,headless=True,
                         launch_range_gate=config.launch_range_m)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed)
        if seed is not None:
            self.pending_mirror=None
        explicit = (options or {}).get('turning_scenario')
        if explicit is not None:
            scenario=TurningScenario(**explicit)
            self.pending_mirror=None
        elif self.pending_mirror is not None:
            scenario,self.pending_mirror=self.pending_mirror,None
        else:
            scenario=sample_scenario(self.np_random,self.experiment_config.player_headings)
            self.pending_mirror=scenario.mirror()
        observation=apply_scenario(self,scenario)
        self.current_scenario=scenario
        self.previous_bearing=float(observation[15])
        info['turning_scenario']=asdict(scenario)
        return observation,info

    def step(self, action):
        obs,reward,terminated,truncated,info=super().step(action)
        if not terminated and self._episode_step>=self.experiment_config.max_episode_steps:
            truncated=True
            info['terminal_reason']='turning_time_limit'
        extra=bearing_progress_reward(self.previous_bearing,obs[15],self.experiment_config.gamma,
                                      self.experiment_config.bearing_progress_weight,terminated)
        self.previous_bearing=float(obs[15])
        info['reward_breakdown']['base_total']=reward
        info['reward_breakdown']['bearing_progress']=extra
        reward+=extra
        self._episode_reward+=extra
        info['reward_breakdown']['total']=reward
        info['episode']['truncated']=truncated
        info['episode']['r']=self._episode_reward
        if not terminated and not truncated:
            info.pop('episode')  # SB3 episode statistics should contain completed episodes only.
        return obs,reward,terminated,truncated,info


def reserve_run(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run-id must use letters, digits, underscores and hyphens')
    path=RUN_ROOT/run_id
    path.mkdir(parents=True,exist_ok=False)
    return path


def read_bundle(directory):
    directory=Path(directory)
    data=json.loads((directory/'bundle.json').read_text())
    if data.get('schema')!=1:
        raise ValueError('Unsupported checkpoint bundle schema')
    for name in ('candidate.zip','candidate_vecnormalize.pkl'):
        if sha256(directory/name)!=data['hashes'].get(name):
            raise ValueError(f'Checkpoint pair integrity mismatch: {name}')
    return data


def load_pair(model_path,stats_path,config,training=False,device='cpu'):
    """Read weights, optimizer and paired normalization; no learning occurs."""
    vector=DummyVecEnv([lambda:TurningEnv(config)])
    norm=None
    try:
        norm=VecNormalize.load(str(stats_path),vector)
        if norm.obs_rms.mean.shape!=(19,) or norm.obs_rms.var.shape!=(19,) or norm.obs_rms.count<=0:
            raise ValueError('Incompatible normalization dimensions/count')
        if not np.isfinite(norm.obs_rms.mean).all() or not np.isfinite(norm.obs_rms.var).all() or (norm.obs_rms.var<0).any():
            raise ValueError('Invalid normalization statistics')
        if not norm.norm_obs or norm.epsilon<=0 or not math.isfinite(norm.epsilon) or not math.isfinite(norm.clip_obs) or norm.clip_obs<=0:
            raise ValueError('Expected finite observation normalization with positive epsilon/clip')
        norm.action_space=vector.action_space  # Historical VecNormalize action metadata.
        norm.training=bool(training and config.update_normalization)
        norm.norm_reward=False
        kwargs=({'n_steps':config.n_steps,'batch_size':config.batch_size,
                 'learning_rate':config.learning_rate,'gamma':config.gamma} if training else {})
        model=PPO.load(str(model_path),env=norm,device=device,**kwargs)
        if training:
            for group in model.policy.optimizer.param_groups:
                group['lr']=config.learning_rate
        if model.observation_space.shape!=(19,) or model.action_space.shape!=(5,):
            raise ValueError('Expected 19 observations and 5 hybrid actions')
        if not isinstance(model.policy,HybridActorCriticPolicy) or tuple(model.policy.action_net.weight.shape)!=(4,64) or tuple(model.policy.fire_net.weight.shape)!=(1,64):
            raise ValueError('Incompatible hybrid action heads')
        np.testing.assert_array_equal(model.action_space.low,[-1,-1,-1,-1,0])
        np.testing.assert_array_equal(model.action_space.high,[1,1,1,1,1])
        return model,norm
    except BaseException:
        (norm or vector).close()
        raise
