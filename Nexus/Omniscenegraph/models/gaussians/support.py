"""Physical ellipsoid support constraints for object-local Gaussian assets."""
import torch


def confine_gaussian_support(means, scales, rotations, sizes, sigma=3., margin=.08):
    if sigma <= 0 or margin < 0:
        raise ValueError('Invalid Gaussian support limits')
    remaining = sizes/2 + margin - means.abs()
    valid = (remaining > 0).all(dim=-1)
    # Axis-aligned standard deviations of R diag(s**2) R.T, including
    # ellipsoid rotation. A scalar shrink factor preserves anisotropy.
    axis_std = torch.sqrt((rotations.square()*scales[:,None,:].square()).sum(dim=-1).clamp_min(1e-16))
    factor = (remaining.clamp_min(0)/(sigma*axis_std)).amin(dim=-1).clamp(max=1.)
    return (scales*factor[:,None]).clamp_min(1e-6), valid
