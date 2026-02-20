#!/usr/bin/env python3
"""
Download Prima model weights from Google Drive and set up the test folder structure.

Usage (run from repo root):
  python tools/download_models_and_setup_test.py

Or from anywhere:
  python tools/download_models_and_setup_test.py --repo-root /path/to/Prima

Requires: pip install gdown
"""

import argparse
import json
import os
import sys
from pathlib import Path


# Google Drive file IDs (from shared view links)
FULL_MODEL_ID = "119kKMcdk1GPww69IQAf6JkXuNMIEEAIk"
TOKENIZER_ID = "11EitVfPVXmdPSJviQQ5ZKasFNbQqD5Bt"


def get_repo_root() -> Path:
    """Repo root is parent of the directory containing this script."""
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Prima weights and set up test folder structure."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to Prima repo root (default: parent of tools/).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only create folders and configs; do not download weights.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root or get_repo_root()
    repo_root = repo_root.resolve()

    test_dir = repo_root / "test"
    trained = test_dir / "trained_models"
    tokenizer_dir = trained / "tokenizer_model"
    output_dir = test_dir / "test_output"
    mri_case_dir = test_dir / "test_mri_case"

    print(f"Repo root: {repo_root}")
    print("Creating test folder structure...")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    mri_case_dir.mkdir(parents=True, exist_ok=True)

    # Write configs (so paths are correct after download)
    prima_config = trained / "prima_config.json"
    prima_config.write_text(
        json.dumps({"full_model_ckpt": "primafullmodel107.pt"}, indent=2)
    )
    print(f"  Wrote {prima_config}")

    tokenizer_config = tokenizer_dir / "config.json"
    # Path relative to repo root so pipeline (run from repo root) finds the checkpoint
    ckpt_path_rel = "test/trained_models/tokenizer_model/vqvae_model_step16799.pth"
    tokenizer_config.write_text(
        json.dumps(
            {
                "vqvae_config": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_res_layers": 2,
                    "downsample_parameters": [[2, 4, 1, 1], [2, 2, 1, 0]],
                    "upsample_parameters": [[2, 4, 1, 1, 0], [2, 2, 1, 0, 0]],
                    "num_channels": [256, 256],
                    "num_res_channels": [256, 256],
                    "num_embeddings": 8192,
                    "embedding_dim": 2,
                    "ckpt_path": ckpt_path_rel,
                }
            },
            indent=2,
        )
    )
    print(f"  Wrote {tokenizer_config}")

    if args.skip_download:
        print("Skipping download (--skip-download).")
        print_next_steps(repo_root, mri_case_dir)
        return 0

    try:
        import gdown
    except ImportError:
        print(
            "Error: gdown is required. Install with: pip install gdown",
            file=sys.stderr,
        )
        return 1

    full_model_path = trained / "primafullmodel107.pt"
    tokenizer_path = tokenizer_dir / "vqvae_model_step16799.pth"

    def download(file_id: str, output: Path, name: str) -> bool:
        if output.exists():
            print(f"  {name} already exists at {output}; skipping.")
            return True
        print(f"  Downloading {name}...")
        try:
            gdown.download(
                id=file_id,
                output=str(output),
                quiet=False,
                fuzzy=True,
            )
            if not output.exists():
                print(f"  Failed to create {output}", file=sys.stderr)
                return False
            print(f"  Saved to {output}")
            return True
        except Exception as e:
            print(f"  Download failed: {e}", file=sys.stderr)
            return False

    ok1 = download(FULL_MODEL_ID, full_model_path, "Full PRIMA model")
    ok2 = download(TOKENIZER_ID, tokenizer_path, "Tokenizer (VQ-VAE)")
    if not ok1 or not ok2:
        return 1

    print_next_steps(repo_root, mri_case_dir)
    return 0


def print_next_steps(repo_root: Path, mri_case_dir: Path) -> None:
    print()
    print("Setup complete. Next steps:")
    print(f"  1. Place a DICOM study folder under: {mri_case_dir}")
    print("     Example: test/test_mri_case/MY_STUDY_ID/")
    print("  2. From the repo root, run:")
    print()
    print("     python end-to-end_inference_pipeline/pipeline.py --config configs/test_pipeline_config.yaml")
    print()
    print("     (Edit configs/test_pipeline_config.yaml to set study_dir to your study path, e.g. test/test_mri_case/MY_STUDY_ID)")
    print()


if __name__ == "__main__":
    sys.exit(main())
