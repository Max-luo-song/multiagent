# nuPlan single-scene generation reproduction

This procedure uses the existing Nexus environment, shared nuPlan database, and
official `nuplan.ckpt`. It is a functional reproduction and visualization, not
the paper's quantitative evaluation. No training is performed.

## Observed failures and controls

The original `scene_gen` demo randomly conditions on complete agent trajectories
with probability 0.3. Unlike `Nexus.generate_task_mask` during training, it does
not always condition on ego. The seed-0 run for scenario `044e2df7a9375653` did not
condition on ego. Adding ego alone did not remove the extreme discontinuities.

The default `full_sequence` diffusion schedule also sets the first five time
steps to zero, even for unconditioned agents in scene generation. Initial noise
in those historical states therefore enters the sampler with a clean-state
time label. This is appropriate only when those states are known context.

`scripts/pipeline/run_repro_demo.py` is derived from the existing work-directory
helper. The original helper and the core model source are preserved. It adds:

- `--condition-ego`: always condition on the complete ego trajectory.
- `--denoise-history`: initialize the full-sequence schedule with all time steps
  eligible for denoising; the existing task-mask logic still fixes known context.
  This option rejects other scheduling modes.
- `--reference-run`: before inference, verify the scene tensor, validity mask,
  original random task mask, token, seed, checkpoint hash, and pipeline hash
  against a saved run. Requires using the ego-conditioned branch.

These are explicit local reproduction adjustments. They are not claimed to be
the authors' official scene-generation evaluation settings.

## Run

In the existing activated environment:

```bash
source /inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/cache_env.sh
cd "$NEXUS_WORKDIR/Nexus"
"$NEXUS_WORKDIR/envs/nexus/bin/python" -u scripts/pipeline/run_repro_demo.py \
  --task scene_gen --seed 0 --condition-ego --denoise-history \
  --reference-run "$NEXUS_WORKDIR/runs/demo_20261020_082514_215622"
```

Each run creates a fresh `runs/demo_<timestamp>/` directory. `run.json` records
both optional adjustments, provenance, and output GIFs. `config.yaml` records the
Hydra configuration; the sampling override is applied at runtime and recorded
in `run.json` as `denoise_history`. Importing libraries from shared storage can
take several minutes before model loading begins.

## Validation

```bash
"$NEXUS_WORKDIR/envs/nexus/bin/python" scripts/pipeline/check_repro_schedule.py
```

This checks the actual sampler methods without importing the entire application:
unknown history receives a full noise schedule, known context stays fixed, and
BP's effective schedule and initial state are unchanged.

Use `scripts/pipeline/compare_repro_runs.py RUN1 RUN2 RUN3 --output OUTPUT_DIR`
to save diagnostics and a side-by-side final-frame image. It compares only valid
unconditioned non-ego tokens shared across all supplied runs. Displacements use
the repository's normalization scale and 0.5-second data sampling interval.
These continuity diagnostics do not measure ADE/FDE, collision rate, map
compliance, or the paper benchmark. Smooth trajectories alone do not establish
realistic behavior.

## Verified results (2026-10-20)

All three runs completed on scenario `044e2df7a9375653` with seed 0 and the same
checkpoint. The adjusted runner verified equality of the original scene tensors
and random conditioning masks before adding ego as context.

| Run directory under `../runs` | Ego fixed | Unknown history denoised | Adjacent displacement p95 / max (m) |
| --- | --- | --- | --- |
| `demo_20261020_082514_215622` | No | No | 137.23 / 229.91 |
| `demo_20261020_085115_213174` | Yes | No | 137.33 / 230.03 |
| `demo_20261020_120001_008217` | Yes | Yes | 1.31 / 2.65 |

The comparison covers the same 524 valid unconditioned non-ego agent/time
tokens. The nonpositive length/width fraction dropped from 1.91% to 0%; all
generated values are finite. Conditioned valid state values are preserved
exactly. The corrected run contains three readable 81-frame GIFs, with
`status: complete` in its manifest. The final-frame comparison visibly reduces
the scattered trajectories, although behavioral realism and map compliance
remain unmeasured.

Files:

- [Final-frame comparison](../artifacts/scene_gen_reproduction/comparison.png)
- [Numeric diagnostics](../artifacts/scene_gen_reproduction/diagnostics.json)
- Corrected GIF: `../runs/demo_20261020_120001_008217/scripts/pipeline/scene_gen_pred.gif`
  (relative to the repository root, in the existing work-directory results).

Recreate the comparison:

```bash
"$NEXUS_WORKDIR/envs/nexus/bin/python" -u scripts/pipeline/compare_repro_runs.py \
  ../runs/demo_20261020_082514_215622 \
  ../runs/demo_20261020_085115_213174 \
  ../runs/demo_20261020_120001_008217 \
  --output artifacts/scene_gen_reproduction
```

Validation passed: runner/diagnostic script syntax, the schedule checks above,
strict checkpoint loading, reference scene/mask equality, finite outputs, and
animated GIF readability. The `inference_seconds` recorded by this diagnostic
runner includes reference verification and should not be used as benchmark
timing.
