"""Refine small RGB camera residuals after independent LiDAR registration.

Validation RGB is never used in optimization. Its camera residuals are filled
from neighboring training times, independently for each physical camera.
"""
import argparse, json, random, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Omniscenegraph')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'checkpoint', 'output-root']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--camera-only-steps', type=int, default=2000)
    a = p.parse_args()
    import numpy as np, torch, yaml
    from models.trainers.base import BasicTrainer
    from datasets.base.pixel_source import ScenePixelSource
    cfg = yaml.safe_load(a.config.read_text())
    cameras = len(cfg['data']['pixel_source']['cameras'])
    original_init = BasicTrainer.initialize_optimizer
    original_pre = BasicTrainer.preprocess_per_train_step
    original_step = BasicTrainer.optimizer_step
    original_loss = BasicTrainer.compute_losses
    original_sample = ScenePixelSource.propose_training_image

    def initialize(self):
        original_init(self)
        self.step = 0
        self.gaussian_classes.pop('RigidNodes')
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad_(name == 'CamPose')
        weights = self.models['CamPose'].embeds.weight
        assert torch.count_nonzero(weights) == 0
        nframes = len(weights) // cameras
        held = torch.zeros(nframes, dtype=torch.bool, device=weights.device)
        held[10::10] = True
        self._rgb_train_frames = torch.where(~held)[0]
        mask = (~held).repeat_interleave(cameras)[:, None].to(weights.dtype)
        weights.register_hook(lambda grad: grad * mask)

    def preprocess(self, step):
        original_pre(self, step)
        if step == a.camera_only_steps:
            for name in ['Background', 'Sky']:
                for param in self.models[name].parameters():
                    param.requires_grad_(True)
            print('REGISTERED_RGB_REFINE joint_background_camera_phase', flush=True)

    def loss(self, outputs, image_infos, cam_infos, gs):
        result = original_loss(self, outputs, image_infos, cam_infos, gs)
        weights = self.models['CamPose'].embeds.weight.reshape(-1, cameras, 9)
        train = weights[self._rgb_train_frames]
        result['camera_residual_prior'] = .005 * (train ** 2).mean()
        result['camera_residual_smooth'] = .01 * ((train[1:] - train[:-1]) ** 2).mean()
        return result

    def optimizer_step(self):
        original_step(self)
        # The 6D offset is close to the identity; this component bound limits
        # angular changes to well below one degree and translations to 8 cm.
        with torch.no_grad():
            weights = self.models['CamPose'].embeds.weight
            for block, maximum in [(weights[:, :3], .08), (weights[:, 3:], .008)]:
                norm = torch.linalg.vector_norm(block, dim=-1, keepdim=True)
                block.mul_(torch.clamp(maximum / norm.clamp_min(1e-12), max=1))

    def sample(self, candidate_indices=None):
        if not hasattr(self, '_rgb_residual_candidates'):
            self._rgb_residual_candidates = {
                c: [int(i) for i in candidate_indices if int(i) % cameras == c]
                for c in range(cameras)}
        c = 0 if random.random() < .6 else random.choice(list(range(1, cameras)))
        candidates = self._rgb_residual_candidates[c]
        moving = [i for i in candidates if i // cameras >= 60]
        return random.choice(moving if random.random() < .6 else candidates)

    BasicTrainer.initialize_optimizer = initialize
    BasicTrainer.preprocess_per_train_step = preprocess
    BasicTrainer.optimizer_step = optimizer_step
    BasicTrainer.compute_losses = loss
    ScenePixelSource.propose_training_image = sample
    sys.argv = ['train.py', '--config_file', str(a.config), '--resume_from', str(a.checkpoint),
                '--output_root', str(a.output_root), '--project', 'nuplan', '--run_name', a.run_name]
    try:
        runpy.run_path(str(ROOT / 'Omniscenegraph/tools/train.py'), run_name='__main__')
    finally:
        BasicTrainer.initialize_optimizer = original_init
        BasicTrainer.preprocess_per_train_step = original_pre
        BasicTrainer.optimizer_step = original_step
        BasicTrainer.compute_losses = original_loss
        ScenePixelSource.propose_training_image = original_sample
    directory = a.output_root / 'nuplan' / a.run_name
    checkpoint = directory / 'checkpoint_final.pth'
    before = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['models']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    after = state['models']
    for name in ['RigidNodes', 'Affine']:
        for key, value in before[name].items():
            assert torch.equal(value, after[name][key]), (name, key)
    weights = after['CamPose']['embeds.weight'].reshape(-1, cameras, 9)
    nframes = len(weights)
    held = np.arange(10, nframes, 10)
    assert torch.count_nonzero(weights[held]) == 0
    train = np.setdiff1d(np.arange(nframes), held)
    for camera in range(cameras):
        for dimension in range(9):
            weights[held, camera, dimension] = torch.from_numpy(np.interp(held, train, weights[train, camera, dimension].numpy())).to(weights.dtype)
    torch.save(state, checkpoint)
    report = {'status': 'passed', 'heldout_rgb_never_used_for_camera_or_background_training': True,
              'validation_camera_policy': 'linear interpolation of small 6D rotation and translation residuals at neighboring training frames; last frame holds last trained residual',
              'heldout_frames': held.tolist(), 'heldout_gradients_verified_zero_before_interpolation': True,
              'max_translation_residual_m': float(torch.linalg.vector_norm(weights[:, :, :3], dim=-1).max()),
              'max_rotation_6d_offset_norm': float(torch.linalg.vector_norm(weights[:, :, 3:], dim=-1).max()),
              'foreground_and_affine_preserved_exactly': True, 'camera_only_steps': a.camera_only_steps,
              'nexus_parameters_or_predictions_changed': False}
    (directory / 'rgb_refinement_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REGISTERED_RGB_REFINE_COMPLETE ' + str(directory), flush=True)


if __name__ == '__main__':
    main()
