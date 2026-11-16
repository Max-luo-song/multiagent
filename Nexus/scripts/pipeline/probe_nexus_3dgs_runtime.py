"""Validate real CUDA operators and Omniscenegraph imports before training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'Omniscenegraph'))
os.environ.setdefault('MAX_JOBS', '4')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/nexus-3dgs-torch-extensions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'artifacts/nexus_3dgs/runtime_probe.json')
    args = parser.parse_args()
    report = {'status': 'started', 'python': sys.executable, 'checks': [], 'packages': {}}
    try:
        for name in ('torch', 'numpy', 'gsplat', 'pytorch3d', 'kornia', 'open3d', 'diffrp-nvdiffrast', 'viser'):
            try:
                report['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report['packages'][name] = None
        print('PROBE torch/CUDA', flush=True)
        import torch
        torch.set_num_threads(8)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        report['gpu'] = torch.cuda.get_device_name(0)
        print('PROBE PyTorch3D CUDA KNN', flush=True)
        from pytorch3d.ops import knn_points
        from pytorch3d.transforms import matrix_to_quaternion
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device='cuda')
        torch.testing.assert_close(knn_points(points, points).dists, torch.zeros(1, 2, 1, device='cuda'))
        torch.testing.assert_close(matrix_to_quaternion(torch.eye(3, device='cuda')), torch.tensor([1., 0., 0., 0.], device='cuda'))
        report['checks'].append('pytorch3d_cuda_knn_and_rotation')
        print('PROBE gsplat CUDA rasterization and backward (first run compiles kernels)', flush=True)
        from gsplat.rendering import rasterization
        means = torch.tensor([[0., 0., 3.]], device='cuda', requires_grad=True)
        colors, alphas, _ = rasterization(means=means,
            quats=torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            scales=torch.full((1, 3), .2, device='cuda'), opacities=torch.tensor([.9], device='cuda'),
            colors=torch.tensor([[1., .25, .1]], device='cuda'),
            viewmats=torch.eye(4, device='cuda')[None],
            Ks=torch.tensor([[[16., 0., 8.], [0., 16., 8.], [0., 0., 1.]]], device='cuda'),
            width=16, height=16, packed=False)
        if colors.shape != (1, 16, 16, 3) or not torch.isfinite(colors).all() or alphas.max() <= .1:
            raise RuntimeError('Invalid Gaussian rasterization output')
        colors.sum().backward()
        if means.grad is None or not torch.isfinite(means.grad).all():
            raise RuntimeError('Gaussian rasterizer backward failed')
        report['checks'].append('gsplat_cuda_forward_backward')
        print('PROBE nvdiffrast sky texture forward/backward', flush=True)
        import nvdiffrast.torch as dr
        texture = torch.ones(1, 6, 8, 8, 3, device='cuda', requires_grad=True)
        directions = torch.tensor([[[[1., 0., 0.]]]], device='cuda')
        result = dr.texture(texture, directions, filter_mode='linear', boundary_mode='cube')
        torch.testing.assert_close(result, torch.ones_like(result))
        result.sum().backward()
        if texture.grad is None or not torch.isfinite(texture.grad).all():
            raise RuntimeError('Sky texture backward failed')
        report['checks'].append('nvdiffrast_cuda_forward_backward')
        from models.modules import EnvLight
        sky = EnvLight('Sky', resolution=8, bounded=True).cuda()
        inputs = {'viewdirs': directions[0]}
        torch.testing.assert_close(sky(inputs), torch.full_like(directions[0], .5))
        with torch.no_grad():
            sky.base.copy_(torch.linspace(-20, 20, sky.base.numel(), device='cuda').reshape_as(sky.base))
        bounded_rgb = sky(inputs)
        if not ((bounded_rgb >= 0) & (bounded_rgb <= 1)).all():
            raise RuntimeError('Bounded sky escaped RGB range')
        bounded_rgb.sum().backward()
        if sky.base.grad is None or not torch.isfinite(sky.base.grad).all() or sky.base.grad.abs().sum() == 0:
            raise RuntimeError('Bounded sky lost its training gradient')
        legacy = EnvLight('Sky', resolution=8).cuda()
        with torch.no_grad():
            legacy.base.fill_(2.)
        torch.testing.assert_close(legacy(inputs), torch.full_like(directions[0], 2.))
        report['checks'].append('envlight_bounded_cuda_gradient_and_legacy_compatibility')
        print('PROBE actual dataset/trainer/train entry imports', flush=True)
        from datasets.driving_dataset import DrivingDataset
        from datasets.nuplan.nuplan_sourceloader import NuPlanPixelSource
        import inspect
        if inspect.isabstract(NuPlanPixelSource):
            raise RuntimeError('NuPlan source does not implement the current dataset interface')
        from models.trainers.scene_graph import MultiTrainer
        from tools.train import main as train
        report['checks'].append('omniscenegraph_dataset_trainer_training_entry_imports')
        torch.cuda.synchronize()
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('NEXUS_3DGS_RUNTIME_OK '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
