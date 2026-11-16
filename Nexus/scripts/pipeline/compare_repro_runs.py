"""Compare saved single-scene runs; these diagnostics are not paper metrics."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def quantiles(values):
    if not values.numel():
        return None
    return dict(zip(("median", "p95", "max"), torch.quantile(values.float(), torch.tensor([0.5, 0.95, 1.0])).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [torch.load(p / "prediction.pt", map_location="cpu", weights_only=False) for p in args.runs]
    reference = loaded[0]
    common = reference["validity"].bool().clone()
    for data in loaded:
        assert torch.equal(data["scene_tensor_gt"], reference["scene_tensor_gt"]), "Scenes differ"
        assert torch.equal(data["validity"], reference["validity"]), "Validity differs"
        common &= ~data["task_mask"][..., 0].bool()
    common[:, 0] = False  # Ego conditioning differs across runs.
    pairs = common[:, :, 1:] & common[:, :, :-1]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    panels = []
    for run, data in zip(args.runs, loaded):
        manifest = json.loads((run / "run.json").read_text())
        pred, gt = data["sampled_tensor"], data["scene_tensor_gt"]
        xy = pred[..., :2] * 104.0  # FEATURE_STD[x,y] * 2, metres.
        jumps = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)
        dimensions = pred[..., 6:8] * torch.tensor([5.0, 1.6]) + torch.tensor([4.5, 2.0])
        known = data["task_mask"].bool() & data["validity"].bool().unsqueeze(-1)
        summary = {
            "run": str(run.resolve()),
            "status": manifest["status"],
            "scenario_token": manifest["scenario_token"],
            "condition_ego": bool(data["task_mask"][0, 0].bool().all()),
            "denoise_history": manifest.get("denoise_history", False),
            "all_finite": bool(torch.isfinite(pred).all()),
            "common_unconditioned_valid_tokens": int(common.sum()),
            "adjacent_displacement_metres": quantiles(jumps[pairs]),
            "history_displacement_metres": quantiles(jumps[:, :, :4][pairs[:, :, :4]]),
            "history_future_boundary_displacement_metres": quantiles(jumps[:, :, 4][pairs[:, :, 4]]),
            "future_displacement_metres": quantiles(jumps[:, :, 5:][pairs[:, :, 5:]]),
            "nonpositive_dimension_fraction": float((dimensions <= 0).any(-1)[common].float().mean()),
            "max_conditioned_error_normalized": float((pred - gt).abs()[known].max()) if known.any() else None,
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
        with Image.open(run / "scripts/pipeline/scene_gen_pred.gif") as gif:
            gif.seek(gif.n_frames - 1)
            frame = gif.convert("RGB")
        panel = Image.new("RGB", (800, 845), "white")
        panel.paste(frame.resize((800, 800)), (0, 45))
        label = f'ego={summary["condition_ego"]}, denoise_history={summary["denoise_history"]}'
        ImageDraw.Draw(panel).text((15, 15), label, fill="black")
        panels.append(panel)
    canvas = Image.new("RGB", (800 * len(panels), 845), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (800 * i, 0))
    canvas.save(args.output / "comparison.png")
    (args.output / "diagnostics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("Diagnostics describe one sample, not ADE/FDE, off-road rate, or paper benchmark results.")


if __name__ == "__main__":
    main()
