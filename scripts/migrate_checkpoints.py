#!/usr/bin/env python3
"""Migrate legacy FluxFlow checkpoints to versioned format.

This script migrates checkpoints without version metadata to the new versioned format,
adding model_metadata.json with version and architecture information.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import fluxflow
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fluxflow.models.versioning import load_versioned_checkpoint, save_versioned_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def migrate_checkpoint(
    input_path: Path,
    output_path: Path,
    model_version: str = "0.3.0",
    force: bool = False,
) -> None:
    """
    Migrate legacy checkpoint to versioned format.

    Args:
        input_path: Path to legacy checkpoint file
        output_path: Path to save versioned checkpoint directory
        model_version: Target model version
        force: Overwrite existing output directory
    """
    if not input_path.exists():
        logger.error(f"Input checkpoint not found: {input_path}")
        sys.exit(1)

    if output_path.exists() and not force:
        logger.error(f"Output path already exists: {output_path}. Use --force to overwrite.")
        sys.exit(1)

    logger.info(f"Loading legacy checkpoint: {input_path}")
    try:
        model = load_versioned_checkpoint(input_path, device="cpu")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    logger.info(f"Saving as versioned checkpoint: {output_path}")
    try:
        save_versioned_checkpoint(
            model,
            output_path,
            model_version=model_version,
            training_info={
                "migrated_from": str(input_path),
                "migration_tool": "migrate_checkpoints.py",
            },
        )
    except Exception as e:
        logger.error(f"Failed to save versioned checkpoint: {e}")
        sys.exit(1)

    logger.info("Migration complete!")
    logger.info(f"  Input:  {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Version: {model_version}")
    logger.info("")
    logger.info("Files created:")
    logger.info(f"  - {output_path / 'model.safetensors'}")
    logger.info(f"  - {output_path / 'model_metadata.json'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy FluxFlow checkpoints to versioned format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a checkpoint
  python migrate_checkpoints.py old_model.safetensors versioned_model/

  # Migrate with specific version
  python migrate_checkpoints.py checkpoint.pt output/ --version 0.3.1

  # Force overwrite existing output
  python migrate_checkpoints.py checkpoint.pt output/ --force
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to legacy checkpoint file (.safetensors or .pt)",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Path to output directory for versioned checkpoint",
    )

    parser.add_argument(
        "--version",
        default="0.3.0",
        help="Model version to assign (default: 0.3.0)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    migrate_checkpoint(
        input_path=args.input,
        output_path=args.output,
        model_version=args.version,
        force=args.force,
    )


if __name__ == "__main__":
    main()
