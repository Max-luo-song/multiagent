"""Keep observed vehicle RGB supervision on the foreground rendering branch."""
import torch


def foreground_owned_rgb(composite_rgb, foreground_rgb, observed_vehicle_mask):
    if composite_rgb.shape!=foreground_rgb.shape or observed_vehicle_mask.shape!=composite_rgb.shape[:-1]:
        raise ValueError('Foreground supervision dimensions disagree')
    return torch.where(observed_vehicle_mask.bool()[...,None],foreground_rgb,composite_rgb)


def refinement_gradients(main_info, auxiliary_infos=(), absgrad=True):
    """Include supervised auxiliary rasterizations in Gaussian refinement.

    All passes use the same unpacked Gaussian ordering. Absolute gradients
    accumulate without cancellation, as in gsplat's single-pass absgrad path.
    """
    key = 'absgrad' if absgrad else 'grad'
    result = torch.zeros_like(main_info['means2d'])
    for info in (main_info, *auxiliary_infos):
        value = getattr(info['means2d'], key, None)
        if value is not None:
            if value.shape != result.shape:
                raise ValueError('Refinement passes use different Gaussian layouts')
            result = result + value.detach()
    return result
