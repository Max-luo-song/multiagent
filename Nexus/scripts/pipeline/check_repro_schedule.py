"""Check that scene generation denoises unknown history and BP is unchanged.

Extract the actual sampler methods to test their behavior without importing the
full TensorFlow/nuPlan stack. Run with the existing Nexus Python environment.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def main():
    source = Path(__file__).resolve().parents[2] / (
        "nuplan_extent/planning/training/modeling/models/diffusion/scenediffusion.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "simpleDiffusion")
    names = {"_generate_scheduling_matrix", "_filling_scene_tensor"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 2
    scope = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    sampler = SimpleNamespace(scheduling_matrix="full_sequence", steps=32, fill_scene_tensor=True)
    scene = torch.zeros(1, 3, 21, 8)
    noise = torch.ones_like(scene)
    schedule = scope["_generate_scheduling_matrix"]
    fill = scope["_filling_scene_tensor"]

    # Both histories of all agents are known for BP. The task mask should
    # produce exactly the same effective schedule and initial state.
    bp = torch.zeros_like(scene)
    bp[:, :, :5] = 1
    old, old_noise = fill(sampler, scene, noise, bp, schedule(sampler, scene))
    new, new_noise = fill(sampler, scene, noise, bp, schedule(sampler, scene, zero_init=False))
    assert np.array_equal(old, new)
    assert torch.equal(old_noise, new_noise)
    print("BP_EFFECTIVE_SCHEDULE_UNCHANGED")

    # SG knows two complete trajectories; the third must be generated at all
    # times, including history. The default incorrectly starts history at t=0.
    sg = torch.zeros_like(scene)
    sg[:, :2] = 1
    old, _ = fill(sampler, scene, noise, sg, schedule(sampler, scene))
    new, _ = fill(sampler, scene, noise, sg, schedule(sampler, scene, zero_init=False))
    assert np.all(old[:, 0, 2, :5] == 0)
    assert np.all(new[0, 0, 2, :] == 1)
    assert np.all(new[-1, 0, 2, :] == 0)
    assert np.all(new[:, 0, :2] == 0)
    print("SG_UNKNOWN_HISTORY_DENOISED_AND_CONTEXT_PRESERVED")


if __name__ == "__main__":
    main()
