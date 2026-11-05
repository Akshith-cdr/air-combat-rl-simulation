"""Stage 7 policy: Stage 6 flight network plus a PPO-trained fire network.

* Flight: the Stage 6 Gaussian head (19 observations -> 64 -> 64 -> 4), loaded
  unchanged. With ``train_flight=False`` (Stage 7 default) its tensors are frozen
  and the aircraft is flown at the Gaussian mean, exactly as in evaluation, so
  training returns vary only with the fire decisions being learned.
* Fire: a Bernoulli head with its own network (19 normalized observations ->
  64 -> 64 -> 1 logit, tanh). It reads the observations directly, including
  range (index 10) and cos(ATA) (index 11). Stage 6's fire output was a linear
  layer on the flight network's features, which do not separate the
  500-1000 m launch band well.

There is no rule or fixed classifier in the fire lane: during training the fire
action is sampled from the Bernoulli; in deterministic evaluation the policy
fires when p(fire) >= 0.5. The environment's launch interlock then accepts or
rejects the request.
"""

import torch as th
from torch import nn

from hybrid_ppo import HybridActorCriticPolicy


FLIGHT_PREFIXES = ('mlp_extractor.policy_net.', 'action_net.', 'log_std')
FIRE_PREFIXES = ('fire_mlp.', 'fire_net.')


class LearnedFirePolicy(HybridActorCriticPolicy):
    """Hybrid policy with a separate fire network; flight can be frozen."""

    def __init__(self, *args, train_flight=False, fire_hidden=64, **kwargs):
        if type(train_flight) is not bool:
            raise ValueError('train_flight must be a boolean')
        self.train_flight = train_flight
        self.fire_hidden = int(fire_hidden)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        inputs = self.features_extractor.features_dim
        self.fire_mlp = nn.Sequential(nn.Linear(inputs, self.fire_hidden), nn.Tanh(),
                                      nn.Linear(self.fire_hidden, self.fire_hidden), nn.Tanh())
        if self.ortho_init:
            self.fire_mlp.apply(lambda module: self.init_weights(module, gain=2 ** 0.5))
        if self.fire_net.in_features != self.fire_hidden:
            self.fire_net = nn.Linear(self.fire_hidden, 1)
        if not self.train_flight:
            for name, parameter in self.named_parameters():
                if name.startswith(FLIGHT_PREFIXES):
                    parameter.requires_grad_(False)
        # Optimizer over the trainable parameters only; rebuilt identically on load.
        self.optimizer = self.optimizer_class(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def _distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        fire_logits = self.fire_net(self.fire_mlp(features))
        distribution = self.action_dist.proba_distribution(
            self.action_net(latent_pi), self.log_std, fire_logits)
        return distribution, latent_vf

    def _flight_mean(self, distribution, actions):
        if self.train_flight:
            return actions
        return th.cat((distribution.gaussian.mean, actions[..., 4:5]), dim=-1)

    def forward(self, obs, deterministic=False):
        distribution, latent_vf = self._distribution(obs)
        actions = self._flight_mean(distribution, distribution.get_actions(deterministic=deterministic))
        # With frozen flight the Gaussian factor of the PPO ratio is identically 1.
        return actions, self.value_net(latent_vf), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution, latent_vf = self._distribution(obs)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        return self._distribution(obs)[0]

    def _predict(self, observation, deterministic=False):
        distribution = self.get_distribution(observation)
        return self._flight_mean(distribution, distribution.get_actions(deterministic=deterministic))

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'train_flight': self.train_flight,
                'fire_hidden': self.fire_hidden}

    def frozen_parameter_names(self):
        return [name for name, p in self.named_parameters() if not p.requires_grad]


__all__ = ['LearnedFirePolicy', 'FLIGHT_PREFIXES', 'FIRE_PREFIXES']
