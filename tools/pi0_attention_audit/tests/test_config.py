import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pi0_attention_audit.cli import main
from pi0_attention_audit.config import load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "attention_comparison.example.toml"


class ConfigTest(unittest.TestCase):
    def test_example_config_expands_paths_and_validates_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with_checkpoint = root / "with.safetensors"
            without_checkpoint = root / "without.safetensors"
            with_checkpoint.touch()
            without_checkpoint.touch()
            deployment = root / "deployment"
            for episode in (0, 2):
                episode_dir = deployment / f"{episode:04d}"
                episode_dir.mkdir(parents=True)
                (episode_dir / "cam_top_rgb.mp4").touch()
                (episode_dir / "state.pkl").touch()
            environment = {
                "PI0_WITH_HAND_CHECKPOINT": str(with_checkpoint),
                "PI0_WITHOUT_HAND_CHECKPOINT": str(without_checkpoint),
                "PI0_AUDIT_DEPLOY_ROOT": str(deployment),
                "PI0_AUDIT_LEROBOT_ROOT": str(root / "lerobot"),
                "PI0_AUDIT_OUTPUT": str(root / "outputs"),
            }
            with patch.dict(os.environ, environment, clear=False):
                config = load_config(EXAMPLE)
                self.assertEqual(config.analysis.patch_grid, 16)
                self.assertEqual(config.models.with_hand, with_checkpoint)
                self.assertTrue(all(check.exists for check in config.path_checks("deployment")))

    def test_cli_allow_missing_is_a_structural_smoke(self) -> None:
        environment = {
            "PI0_WITH_HAND_CHECKPOINT": "/missing/with.safetensors",
            "PI0_WITHOUT_HAND_CHECKPOINT": "/missing/without.safetensors",
            "PI0_AUDIT_DEPLOY_ROOT": "/missing/deployment",
            "PI0_AUDIT_LEROBOT_ROOT": "/missing/lerobot",
            "PI0_AUDIT_OUTPUT": "/tmp/pi0-attention-output",
        }
        with patch.dict(os.environ, environment, clear=False):
            code = main(
                [
                    "validate",
                    "--config",
                    str(EXAMPLE),
                    "--source",
                    "deployment",
                    "--allow-missing",
                ]
            )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
