# LV Robotics Lab Mobile ALOHA fine-tuning

The `pi05_mobile_aloha_lora` training configuration preserves the workstation
recipe used for the LV Robotics Lab Mobile ALOHA dataset while keeping it in
the organization fork that owns the OpenPI-specific adaptation.

Dataset:
`robotics-lv/aloha-mobile-dummy-lerobot-v3`

The dataset uses three cameras (`cam_high`, `cam_left_wrist`, and
`cam_right_wrist`), a 14-dimensional arm/gripper state, and a 14-dimensional
action. Its `base_action` field is always zero and is intentionally excluded
from this recipe.

First compute normalization statistics for the exact dataset revision, then
run the LoRA configuration:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_mobile_aloha_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_mobile_aloha_lora \
  --exp-name=mobile_aloha_lora \
  --overwrite
```

The migrated workstation configuration uses PI0.5, LoRA variants for both the
PaliGemma and action-expert components, batch size 8, 20,000 steps, task-text
prompts, and the upstream Trossen normalization assets. Verify the generated
normalization statistics and the camera/joint contract before training or
deployment.

Provenance: migrated on 2026-08-14 from the only local modification in
`/home/feibo/workspace/aloha_mobile_dummy_lerobot_v3/training/openpi` at
upstream commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
