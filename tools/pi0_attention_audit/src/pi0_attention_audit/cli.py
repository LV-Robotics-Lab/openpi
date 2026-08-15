"""Command-line interface for Pi0 attention audits."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .checkpoint import checkpoint_manifest
from .config import load_config
from .modeling import MissingAnalysisDependency
from .runners import run_action_expert, run_siglip


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi0-attention")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="preflight configured paths")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--source", choices=("deployment", "lerobot", "siglip"), required=True)
    validate.add_argument("--allow-missing", action="store_true")
    validate.add_argument("--json", action="store_true", dest="json_output")

    action = subcommands.add_parser("action-expert", help="run action-expert attention")
    action.add_argument("--config", type=Path, required=True)
    action.add_argument("--source", choices=("deployment", "lerobot"), required=True)
    action.add_argument("--max-samples", type=int)
    action.add_argument("--device")

    siglip = subcommands.add_parser("siglip", help="run SigLIP attention rollout")
    siglip.add_argument("--config", type=Path, required=True)
    siglip.add_argument("--max-samples", type=int)
    siglip.add_argument("--device")

    manifest = subcommands.add_parser("manifest", help="hash checkpoint files")
    manifest.add_argument("paths", type=Path, nargs="+")
    manifest.add_argument("--output", type=Path)
    manifest.add_argument("--relative-to", type=Path)
    manifest.add_argument("--uri-prefix")
    return parser


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checks = config.path_checks(args.source)
    missing = [check for check in checks if not check.exists]
    if args.json_output:
        print(json.dumps([check.as_dict() for check in checks], indent=2))
    else:
        for check in checks:
            status = "ok" if check.exists else "missing"
            print(f"{status:7} {check.label}: {check.path}")
    return 0 if not missing or args.allow_missing else 2


def _manifest(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": 1,
        "files": checkpoint_manifest(
            args.paths,
            relative_to=args.relative_to,
            uri_prefix=args.uri_prefix,
        ),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)
    else:
        print(rendered, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            code = _validate(args)
        elif args.command == "manifest":
            code = _manifest(args)
        elif args.command == "action-expert":
            output = run_action_expert(
                load_config(args.config),
                source=args.source,
                max_samples=args.max_samples,
                device_name=args.device,
            )
            print(output)
            code = 0
        elif args.command == "siglip":
            output = run_siglip(
                load_config(args.config),
                max_samples=args.max_samples,
                device_name=args.device,
            )
            print(output)
            code = 0
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (FileNotFoundError, MissingAnalysisDependency, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    if argv is None:
        raise SystemExit(code)
    return code
