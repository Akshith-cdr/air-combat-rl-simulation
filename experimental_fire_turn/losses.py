"""Fire-head graph construction, never an optimizer or training loop."""

import torch
from torch.nn import functional as F

ALLOWED = ('fire_net.weight', 'fire_net.bias')


def freeze_policy(policy):
    names = dict(policy.named_parameters())
    if any(name not in names for name in ALLOWED):
        raise ValueError('Expected existing two-parameter fire head')
    for name, parameter in names.items():
        parameter.requires_grad_(name in ALLOWED)
        parameter.grad = None
    policy.set_training_mode(False)
    # The source PPO optimizer contains stale/shared parameter state. It must
    # never be reused for this supervised probe; no optimizer is constructed.
    policy.optimizer = None
    return {name: parameter.requires_grad for name, parameter in names.items()}


def fire_logits(policy, observations):
    if {name for name, p in policy.named_parameters() if p.requires_grad} != set(ALLOWED):
        raise ValueError('Freeze contract violated')
    with torch.no_grad():
        features = policy.extract_features(observations)
        if policy.share_features_extractor:
            latent, _ = policy.mlp_extractor(features)
        else:
            latent = policy.mlp_extractor.forward_actor(features[0])
    return policy.fire_net(latent.detach()).squeeze(-1)


def sampling_weights(labels, population_counts, class_draw_probabilities=(.5, .5)):
    """p(i)/q(i) for uniform population, class-stratified uniform draws."""
    if len(population_counts) != 2 or any(type(n) is not int or n <= 0 for n in population_counts):
        raise ValueError('Both population classes must be present')
    probs = torch.as_tensor(class_draw_probabilities, dtype=torch.float64)
    if probs.shape != (2,) or not torch.isfinite(probs).all() or (probs <= 0).any() or not torch.isclose(probs.sum(), torch.tensor(1., dtype=probs.dtype)):
        raise ValueError('Class draw probabilities must be positive and sum to one')
    if labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Binary one-dimensional labels required')
    prevalence = torch.tensor(population_counts, dtype=probs.dtype)/sum(population_counts)
    return (prevalence/probs)[labels.long().cpu()].to(device=labels.device, dtype=torch.float32)


def weighted_bce(logits, labels, weights, fit_mask=None):
    if logits.ndim != 1 or labels.shape != logits.shape or weights.shape != logits.shape or not logits.numel():
        raise ValueError('Nonempty matching vectors required')
    if not torch.isfinite(logits).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Finite logits and binary labels required')
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Weights must be finite and positive')
    if fit_mask is not None:
        if fit_mask.shape != logits.shape or fit_mask.dtype != torch.bool or not fit_mask.any():
            raise ValueError('A nonempty boolean actionable-state mask is required')
        logits, labels, weights = logits[fit_mask], labels[fit_mask], weights[fit_mask]
    # Mean of importance-weighted losses, NOT division by sum(weights).
    return (weights*F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')).mean()
