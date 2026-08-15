import tempfile
import unittest
from pathlib import Path

from pi0_attention_audit.checkpoint import (
    checkpoint_manifest,
    remap_lerobot_pi0_key,
    sha256_file,
)


class CheckpointTest(unittest.TestCase):
    def test_key_remapping(self) -> None:
        self.assertEqual(
            remap_lerobot_pi0_key("model.action_in_proj.weight"),
            "embed_action_time.action_in_proj.weight",
        )
        self.assertEqual(
            remap_lerobot_pi0_key("model.paligemma_with_expert.gemma_expert.model.layers.0.weight"),
            "model.dit.layers.0.weight",
        )
        self.assertIsNone(
            remap_lerobot_pi0_key("model.paligemma_with_expert.paligemma.lm_head.weight")
        )

    def test_manifest_hashes_without_loading_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            path.write_bytes(b"checkpoint-fixture")
            self.assertEqual(
                sha256_file(path),
                "5d4fa22c80243cdb189dacdbf82968cd517e5153f568cb38f1592da625fdd9e0",
            )
            self.assertEqual(
                checkpoint_manifest([path]),
                [
                    {
                        "path": str(path),
                        "bytes": 18,
                        "sha256": (
                            "5d4fa22c80243cdb189dacdbf82968cd517e5153f568cb38f1592da625fdd9e0"
                        ),
                    }
                ],
            )
            self.assertEqual(
                checkpoint_manifest(
                    [path],
                    relative_to=directory,
                    uri_prefix="hf://example-org/example-model",
                )[0]["uri"],
                "hf://example-org/example-model/tiny.safetensors",
            )


if __name__ == "__main__":
    unittest.main()
