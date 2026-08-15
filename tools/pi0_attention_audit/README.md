# π0 Attention Audit

This directory is an isolated OpenPI tool for comparing how two π0 policy
checkpoints allocate visual and action-expert attention. It supports:

- SigLIP attention rollout over visual tokens;
- action-expert attention from denoising/action queries to image keys;
- deployment episodes with joint state stored in `state.pkl`;
- LeRobot video/parquet episodes with `observation.state`;
- region metrics, overlays, comparison grids, and portable checkpoint manifests.

## Why this is a nested tool

The audit reproduces a historical Transformers-based π0 analysis environment.
Its NumPy, PyTorch, and Transformers requirements intentionally differ from the
OpenPI root environment, so it remains a separate `uv` project under `tools/`
and is not an OpenPI workspace member. Run commands with `--project
tools/pi0_attention_audit` from the repository root.

This is a migration of the maintained, reusable implementation from the former
private `pi0-attention-audit` repository at source commit
`613071cbb04ff23bb94ec153450c2152f53fcc7c`. Private checkpoints, datasets,
machine inventories, reports, and case-specific provenance were deliberately
excluded from the public OpenPI fork.

## Install

```bash
uv sync --project tools/pi0_attention_audit --frozen --extra dev
```

Add `--extra analysis` only on a compatible GPU analysis host. The base project
requires only NumPy, so configuration validation and manifest hashing do not
install a GPU stack.

## Configure

Copy `configs/attention_comparison.example.toml` and provide runtime locations
through environment variables:

```bash
export PI0_WITH_HAND_CHECKPOINT=/models/pi0-with-hand/model.safetensors
export PI0_WITHOUT_HAND_CHECKPOINT=/models/pi0-without-hand/model.safetensors
export PI0_AUDIT_DEPLOY_ROOT=/data/deployment
export PI0_AUDIT_LEROBOT_ROOT=/data/lerobot
export PI0_AUDIT_OUTPUT=/data/audit-output
```

Do not replace the environment placeholders with machine-specific absolute
paths in committed configuration.

## Validate and run

```bash
uv run --project tools/pi0_attention_audit pi0-attention validate \
  --config tools/pi0_attention_audit/configs/attention_comparison.example.toml \
  --source deployment \
  --allow-missing

uv run --project tools/pi0_attention_audit pi0-attention action-expert \
  --config tools/pi0_attention_audit/configs/attention_comparison.example.toml \
  --source deployment \
  --max-samples 2
```

Both policy variants receive the same seeded initial denoising noise for each
sample. The maintained action-expert command uses real state, ten Euler
denoising steps, and region fractions computed from raw softmax attention.

## Verification

```bash
uv run --project tools/pi0_attention_audit pytest \
  tools/pi0_attention_audit/tests
uv run --project tools/pi0_attention_audit ruff check \
  tools/pi0_attention_audit/src tools/pi0_attention_audit/tests
```

The lightweight tests cover configuration/path preflight, checkpoint key
remapping and hashing, metrics, repository boundaries, and the missing-input
CLI path. Real-checkpoint analysis remains a separate GPU acceptance test.

## Evidence boundary

Attention is a diagnostic, not causal proof of robot behavior. A finite,
non-zero attention map establishes that an extraction path ran; it does not show
that an attended region caused a policy action or task outcome.

Runtime inputs and generated outputs never belong in Git. Keep checkpoints,
optimizer state, episodes, datasets, caches, galleries, and private provenance
in their approved external stores with immutable digests.
