# Derived from ../run_nexus_demo.py SHA256=5d43cbbda4cbe63111306b306271d18797c46e7e9b87fffbaca1b4ae66a12517
"""Run one Nexus demo scene using the official pipeline helpers.

Upload this file beside cache_env.sh on the Linux GPU instance. The shared
databases/maps are inputs; map caches, configuration and results stay in workdir.
This is a smoke test and visualization, not a benchmark evaluation.
"""

import argparse
import functools
import hashlib
import json
import os
import random
import runpy
import sys
import time
from datetime import datetime
from pathlib import Path


CHECKPOINT_SHA256 = "679f6ccfbd5563f696218311566b76bb62f73c7fbfb04b39f2714fb80aefacfc"
SHARED_ROOT = Path("/inspire/hdd/global_public/public_datas/nuPlan")
DEFAULT_DB = SHARED_ROOT / (
    "plan_data/train_vegas_3/2021.06.11.12.06.26_veh-35_03726_03971.db"
)
MAP_LOCATIONS = (
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_maps(shared_maps, local_maps):
    """Link only GPKG files, leaving their parent/cache directories local."""
    metadata_bytes = (shared_maps / "nuplan-maps-v1.0.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    links = []
    for location in MAP_LOCATIONS:
        version = str(metadata[location]["version"])
        if version in {"", ".", ".."} or "/" in version or "\\" in version:
            raise ValueError(f"Invalid map version: {version!r}")
        relative = Path(location) / version / "map.gpkg"
        source = shared_maps / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        links.append((source, local_maps / relative))

    local_maps.mkdir(parents=True, exist_ok=True)
    if local_maps.is_symlink():
        raise ValueError(f"Map cache root must be a local directory: {local_maps}")
    local_metadata = local_maps / "nuplan-maps-v1.0.json"
    if local_metadata.exists() or local_metadata.is_symlink():
        if local_metadata.is_symlink() or local_metadata.read_bytes() != metadata_bytes:
            raise ValueError(f"Existing map metadata differs: {local_metadata}")
    else:
        local_metadata.write_bytes(metadata_bytes)

    for source, target in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.parent.resolve().relative_to(local_maps.resolve())
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError(f"Map link conflicts with an existing file: {target}")
        else:
            target.symlink_to(source.resolve())
    return local_maps


def model_state_dict(checkpoint):
    """Remove Lightning's model. prefix; strict loading checks all model keys."""
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no non-empty state_dict")
    prefixed = [key.startswith("model.") for key in state]
    if all(prefixed):
        return {key[len("model."):]: value for key, value in state.items()}
    if any(prefixed):
        raise ValueError("Checkpoint state_dict has mixed model. prefixes")
    return state


def configure_demo(cfg, workdir, db_file, maps, checkpoint):
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.scenario_builder.data_root = str(workdir / "data/nuplan")
    cfg.scenario_builder.map_root = str(maps)
    cfg.scenario_builder.map_version = "nuplan-maps-v1.0"
    cfg.scenario_builder.db_files = [str(db_file)]
    cfg.scenario_builder.include_cameras = False
    cfg.scenario_builder.max_workers = 1
    cfg.scenario_filter.log_names = [db_file.stem]
    cfg.scenario_filter.num_scenarios_per_type = 1
    cfg.scenario_filter.limit_total_scenarios = 1
    cfg.scenario_filter.shuffle = False
    cfg.worker = OmegaConf.create({
        "_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential",
        "_convert_": "all",
    })
    cfg.checkpoint = str(checkpoint)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=os.getenv("NEXUS_WORKDIR"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--shared-maps", type=Path, default=SHARED_ROOT / "maps")
    parser.add_argument("--task", choices=("bp", "scene_gen"), default="bp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition-ego", action="store_true", help="Always condition on the full ego trajectory for scene_gen")
    parser.add_argument("--denoise-history", action="store_true", help="Denoise every unconditioned time step with the full_sequence schedule")
    parser.add_argument("--reference-run", type=Path, help="Verify identical scenario tensors and random mask against a saved run")
    args = parser.parse_args()
    if args.condition_ego and args.task != "scene_gen":
        parser.error("--condition-ego requires --task scene_gen")
    if args.workdir is None:
        parser.error("Source cache_env.sh first, or provide --workdir")

    workdir = args.workdir.resolve()
    repository = workdir / "Nexus"
    checkpoint = workdir / "checkpoints/nuplan.ckpt"
    db_file = args.db.resolve()
    pipeline_path = repository / "scripts/pipeline/pipeline.py"
    config_path = repository / "scripts/pipeline/config.yaml"
    for path in (pipeline_path, config_path, checkpoint, db_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")

    print("[1/5] Preparing private map cache and run directory", flush=True)
    maps = prepare_maps(args.shared_maps, workdir / "data/nuplan/maps")
    run_dir = workdir / "runs" / datetime.now().strftime("demo_%Y%m%d_%H%M%S_%f")
    gif_dir = run_dir / "scripts/pipeline"
    gif_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIR={run_dir}", flush=True)

    os.environ["MPLBACKEND"] = "Agg"
    os.environ["NUPLAN_DATA_STORE"] = "local"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(repository))

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pipeline = runpy.run_path(str(pipeline_path), run_name="nexus_demo_helpers")
    cfg = configure_demo(
        OmegaConf.load(config_path), workdir, db_file, maps, checkpoint
    )
    OmegaConf.save(cfg, run_dir / "config.yaml")
    manifest = {
        "task": args.task,
        "condition_ego": args.condition_ego,
        "denoise_history": args.denoise_history,
        "runner_sha256": file_sha256(Path(__file__)),
        "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
        "seed": args.seed,
        "database": str(db_file),
        "map_root": str(maps),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "pipeline_sha256": file_sha256(pipeline_path),
        "status": "started",
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[2/5] Building model and strictly loading checkpoint", flush=True)
    model = pipeline["build_torch_module_wrapper"](cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state_dict(state), strict=True)
    del state
    model = model.to("cuda").eval()
    if args.denoise_history:
        if model.diffuser.scheduling_matrix != "full_sequence":
            raise ValueError("--denoise-history requires the full_sequence schedule")
        # The default schedule skips the first five steps even for agents whose
        # history is unknown. Let the task mask alone select clean context.
        model.diffuser._generate_scheduling_matrix = functools.partial(
            model.diffuser._generate_scheduling_matrix, zero_init=False
        )
    print("CHECKPOINT_LOAD_OK", flush=True)

    print("[3/5] Selecting one scenario from one database", flush=True)
    worker = pipeline["build_worker"](cfg)
    builder = pipeline["build_scenario_builder"](cfg)
    scenarios = pipeline["build_scenarios_from_config"](cfg, builder, worker)
    if not scenarios:
        raise RuntimeError("No scenarios matched the demo filter")
    scenario = scenarios[0]
    manifest["scenario_token"] = str(scenario.token)
    manifest["log_name"] = str(scenario.log_name)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Scenario: {scenario.log_name} / {scenario.token}", flush=True)

    print("[4/5] Extracting features and running GPU inference", flush=True)
    feature_builder = model.get_list_of_required_feature()[0]
    scene = pipeline["get_features_from_scenario"](feature_builder, scenario)
    scene = pipeline["to_tensor"](scene, torch.device("cuda"))
    scene = pipeline["unsqueeze"](scene)
    noise = torch.randn_like(scene.tensor)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if args.condition_ego:
        task_mask = pipeline["create_sg_mask"](scene)
        if args.reference_run:
            reference_manifest = json.loads((args.reference_run / "run.json").read_text())
            reference = torch.load(args.reference_run / "prediction.pt", map_location="cpu", weights_only=False)
            for key in ("scenario_token", "seed", "checkpoint_sha256", "pipeline_sha256"):
                if manifest[key] != reference_manifest[key]:
                    raise RuntimeError(f"Reference mismatch: {key}")
            if not torch.equal(scene.tensor.cpu(), reference["scene_tensor_gt"]):
                raise RuntimeError("Reference scene tensors differ")
            if not torch.equal(scene.validity.cpu(), reference["validity"]):
                raise RuntimeError("Reference validity differs")
            if not torch.equal(task_mask.cpu(), reference["task_mask"]):
                raise RuntimeError("Reference random conditioning masks differ")
            print("REFERENCE_SCENE_AND_MASK_OK", flush=True)
        task_mask[:, 0] = 1.0
        with torch.no_grad():
            predictions = model.forward_inference({"scene_tensor": scene, "task_mask": task_mask}, noise)
        print("EGO_CONDITIONED=True", flush=True)
    else:
        predictions = pipeline["infer_model"](model, scene, noise, args.task)
    torch.cuda.synchronize()
    manifest["inference_seconds"] = time.perf_counter() - started
    sampled = predictions["sampled_tensor"]
    if sampled.shape != scene.tensor.shape or not torch.isfinite(sampled).all().item():
        raise RuntimeError("Prediction has an unexpected shape or non-finite values")
    torch.save({
        "sampled_tensor": sampled.detach().cpu(),
        "scene_tensor_gt": scene.tensor.detach().cpu(),
        "validity": scene.validity.detach().cpu(),
        "task_mask": predictions["task_mask"].detach().cpu(),
    }, run_dir / "prediction.pt")

    print("[5/5] Writing and checking visualization GIFs", flush=True)
    # Official visualization uses relative scripts/pipeline/<task>_<name>.gif paths.
    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        pipeline["visualize_scene_tensor"](scene, predictions, args.task)
    finally:
        os.chdir(previous_cwd)
    gifs = {}
    for name in ("pred", "gt", "condition"):
        path = gif_dir / f"{args.task}_{name}.gif"
        with Image.open(path) as image:
            frames = image.n_frames
            if frames < 2:
                raise RuntimeError(f"GIF is not animated: {path}")
            image.seek(frames - 1)
            image.load()
        gifs[name] = {"path": str(path), "frames": frames, "bytes": path.stat().st_size}
        print(f"GIF: {path} ({frames} frames)", flush=True)
    manifest.update(status="complete", gifs=gifs)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NEXUS_DEMO_OK\nRUN_DIR={run_dir}", flush=True)


if __name__ == "__main__":
    main()
