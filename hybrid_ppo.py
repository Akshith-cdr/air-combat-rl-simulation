"""Project-local Gaussian-flight/Bernoulli-fire policy for SB3 PPO 2.9.0."""

from functools import partial

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Bernoulli, Normal

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class HybridGaussianBernoulliDistribution(Distribution):
    """Product of a four-dimensional diagonal Gaussian and one Bernoulli."""

    def __init__(self, continuous_dim: int = 4):
        super().__init__()
        if continuous_dim != 4:
            raise ValueError("The fighter policy requires exactly four continuous actions")
        self.continuous_dim = continuous_dim
        self.gaussian: Normal | None = None
        self.bernoulli: Bernoulli | None = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        mean_net = nn.Linear(latent_dim, self.continuous_dim)
        log_std = nn.Parameter(
            th.full((self.continuous_dim,), float(log_std_init)),
            requires_grad=True,
        )
        fire_net = nn.Linear(latent_dim, 1)
        return mean_net, log_std, fire_net

    def proba_distribution(self, mean_actions, log_std, fire_logits):
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.gaussian = Normal(mean_actions, action_std)
        self.bernoulli = Bernoulli(logits=fire_logits)
        self.distribution = [self.gaussian, self.bernoulli]
        return self

    def _require_parameters(self):
        if self.gaussian is None or self.bernoulli is None:
            raise RuntimeError("Distribution parameters have not been supplied")

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        self._require_parameters()
        continuous_actions = actions[..., : self.continuous_dim]
        fire_actions = actions[..., self.continuous_dim : self.continuous_dim + 1]
        gaussian_log_prob = self.gaussian.log_prob(continuous_actions).sum(dim=-1)
        fire_log_prob = self.bernoulli.log_prob(fire_actions).sum(dim=-1)
        return gaussian_log_prob + fire_log_prob

    def entropy(self) -> th.Tensor:
        self._require_parameters()
        gaussian_entropy = self.gaussian.entropy().sum(dim=-1)
        fire_entropy = self.bernoulli.entropy().sum(dim=-1)
        return gaussian_entropy + fire_entropy

    def sample(self) -> th.Tensor:
        self._require_parameters()
        return th.cat((self.gaussian.rsample(), self.bernoulli.sample()), dim=-1)

    def mode(self) -> th.Tensor:
        self._require_parameters()
        fire_mode = (self.bernoulli.probs >= 0.5).to(self.gaussian.mean.dtype)
        return th.cat((self.gaussian.mean, fire_mode), dim=-1)

    def actions_from_params(
        self, mean_actions, log_std, fire_logits, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(mean_actions, log_std, fire_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions, log_std, fire_logits):
        actions = self.actions_from_params(mean_actions, log_std, fire_logits)
        return actions, self.log_prob(actions)


class HybridActorCriticPolicy(ActorCriticPolicy):
    """SB3 actor-critic policy with Gaussian flight and Bernoulli fire heads."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("net_arch", {"pi": [64, 64], "vf": [64, 64]})
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        self.action_dist = HybridGaussianBernoulliDistribution(continuous_dim=4)
        self.action_net, self.log_std, self.fire_net = (
            self.action_dist.proba_distribution_net(
                latent_dim=self.mlp_extractor.latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.fire_net: 0.01,
                self.value_net: 1.0,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        continuous_means = self.action_net(latent_pi)
        fire_logits = self.fire_net(latent_pi)
        return self.action_dist.proba_distribution(
            continuous_means, self.log_std, fire_logits
        )


__all__ = ["HybridActorCriticPolicy", "HybridGaussianBernoulliDistribution"]
