import unittest
from pathlib import Path

from pi0_attention_audit.attention import ACTION_EXPERT_METHOD_ID

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTest(unittest.TestCase):
    def test_maintained_modules_are_present(self) -> None:
        package = ROOT / "src" / "pi0_attention_audit"
        expected = {
            "attention.py",
            "checkpoint.py",
            "cli.py",
            "config.py",
            "data.py",
            "metrics.py",
            "modeling.py",
            "runners.py",
            "visualize.py",
        }
        self.assertTrue(expected.issubset({path.name for path in package.glob("*.py")}))

    def test_git_payload_contains_no_runtime_weights_or_private_evidence(self) -> None:
        forbidden_suffixes = {".safetensors", ".pt", ".pth", ".ckpt"}
        forbidden_roots = {"huggingface", "manifests", "provenance", "release", "reports"}
        ignored_roots = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        payload = [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not ignored_roots.intersection(path.relative_to(ROOT).parts)
        ]
        self.assertFalse([path for path in payload if path.suffix.lower() in forbidden_suffixes])
        self.assertFalse([path for path in payload if forbidden_roots.intersection(path.parts)])

    def test_text_payload_has_no_lab_host_paths_or_private_archive_ids(self) -> None:
        forbidden = ("/home/", "/mnt/", "100.64.", "robotics-lv/pi0-attention-audit")
        ignored = {"uv.lock", Path(__file__).name}
        ignored_parts = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        findings: list[tuple[Path, str]] = []
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if (
                not path.is_file()
                or path.name in ignored
                or ignored_parts.intersection(relative.parts)
            ):
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for marker in forbidden:
                if marker in text:
                    findings.append((relative, marker))
        self.assertFalse(findings)

    def test_example_config_uses_environment_paths(self) -> None:
        text = (ROOT / "configs" / "attention_comparison.example.toml").read_text()
        self.assertIn("${PI0_AUDIT_DEPLOY_ROOT}", text)
        self.assertIn("${PI0_AUDIT_LEROBOT_ROOT}", text)
        self.assertNotIn("DRAG_DEMO", text)

    def test_tool_is_not_an_openpi_workspace_member(self) -> None:
        repository_pyproject = ROOT.parents[1] / "pyproject.toml"
        if not repository_pyproject.is_file():
            self.skipTest("standalone source checkout")
        text = repository_pyproject.read_text()
        self.assertIn('members = ["packages/*"]', text)
        self.assertNotIn("tools/pi0_attention_audit", text)

    def test_canonical_action_expert_method_is_explicit(self) -> None:
        self.assertEqual(
            ACTION_EXPERT_METHOD_ID,
            "action-expert-v3-real-state-multistep-raw",
        )


if __name__ == "__main__":
    unittest.main()
