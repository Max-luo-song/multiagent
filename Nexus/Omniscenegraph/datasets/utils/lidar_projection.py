"""Order-independent LiDAR depth rasterization and pointwise visibility."""
import torch


def nearest_depth_and_visibility(pixels, depth, height, width, tolerance=.2):
    """Rasterize nearest positive depth, leaving unobserved pixels at zero.

    Inputs have already been projected into the image and filtered to its
    bounds. Multiple measurements may share a pixel. A hidden measurement
    must not receive the foreground pixel's RGB as its own appearance.
    """
    if pixels.ndim != 2 or pixels.shape != (len(depth), 2) or depth.ndim != 1:
        raise ValueError('Expected [N,2] pixel coordinates and [N] depths')
    if height <= 0 or width <= 0 or tolerance < 0:
        raise ValueError('Invalid image dimensions or depth tolerance')
    if not torch.isfinite(pixels).all() or not torch.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('Projected LiDAR samples must be finite with positive depth')
    if ((pixels[:, 0] < 0) | (pixels[:, 0] >= width) | (pixels[:, 1] < 0) | (pixels[:, 1] >= height)).any():
        raise ValueError('Projected LiDAR pixel outside image')
    coordinates = pixels.long()
    indices = coordinates[:, 1]*width+coordinates[:, 0]
    nearest = depth.new_full((height*width,), float('inf'))
    nearest.scatter_reduce_(0, indices, depth, reduce='amin', include_self=True)
    visible = depth <= nearest[indices]+tolerance
    result = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
    return result.reshape(height, width), visible
