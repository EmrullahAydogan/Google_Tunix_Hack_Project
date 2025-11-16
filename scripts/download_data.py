#!/usr/bin/env python3
"""
Download GSM8K dataset for Google Tunix Hack

GSM8K (Grade School Math 8K) is a dataset of 8,500 grade school math problems
that require multi-step reasoning to solve.

Usage:
    python scripts/download_data.py --dataset gsm8k
    python scripts/download_data.py --dataset gsm8k --output_dir data/raw
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please run: pip install datasets tqdm")
    exit(1)


def download_gsm8k(output_dir: str = "data/raw/gsm8k") -> None:
    """
    Download GSM8K dataset from HuggingFace datasets

    Args:
        output_dir: Directory to save the dataset
    """
    print(f"📥 Downloading GSM8K dataset...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("gsm8k", "main")

        print(f"✅ Dataset loaded successfully!")
        print(f"   - Train samples: {len(dataset['train'])}")
        print(f"   - Test samples: {len(dataset['test'])}")

        # Save train split
        train_path = output_path / "train.json"
        print(f"\n💾 Saving train split to {train_path}...")
        train_data = []
        for example in tqdm(dataset['train'], desc="Processing train"):
            train_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        # Save test split
        test_path = output_path / "test.json"
        print(f"💾 Saving test split to {test_path}...")
        test_data = []
        for example in tqdm(dataset['test'], desc="Processing test"):
            test_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        # Save dataset info
        info_path = output_path / "dataset_info.json"
        info = {
            "dataset_name": "gsm8k",
            "num_train": len(train_data),
            "num_test": len(test_data),
            "description": "Grade School Math 8K - math word problems requiring multi-step reasoning",
            "splits": {
                "train": str(train_path),
                "test": str(test_path)
            }
        }

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        print(f"\n✅ Dataset downloaded successfully!")
        print(f"📁 Files saved to: {output_path}")
        print(f"   - {train_path.name}: {len(train_data)} samples")
        print(f"   - {test_path.name}: {len(test_data)} samples")
        print(f"   - {info_path.name}: metadata")

        # Show example
        print(f"\n📝 Example from dataset:")
        example = train_data[0]
        print(f"Question: {example['question'][:100]}...")
        print(f"Answer: {example['answer'][:100]}...")

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise


def download_math_dataset(output_dir: str = "data/raw/math") -> None:
    """
    Download MATH dataset (optional, for advanced problems)

    Args:
        output_dir: Directory to save the dataset
    """
    print(f"📥 Downloading MATH dataset...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load MATH dataset
        dataset = load_dataset("hendrycks/math", "all")

        print(f"✅ Dataset loaded successfully!")
        print(f"   - Train samples: {len(dataset['train'])}")
        print(f"   - Test samples: {len(dataset['test'])}")

        # Save splits
        for split in ['train', 'test']:
            split_path = output_path / f"{split}.json"
            print(f"\n💾 Saving {split} split to {split_path}...")

            split_data = []
            for example in tqdm(dataset[split], desc=f"Processing {split}"):
                split_data.append({
                    "problem": example["problem"],
                    "solution": example["solution"],
                    "level": example["level"],
                    "type": example["type"]
                })

            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ MATH dataset downloaded successfully!")

    except Exception as e:
        print(f"❌ Error downloading MATH dataset: {e}")
        print(f"ℹ️  MATH dataset is optional. Continuing without it.")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Google Tunix Hack"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "all"],
        help="Dataset to download (default: gsm8k)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for datasets (default: data/raw)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📚 Dataset Downloader - Google Tunix Hack")
    print("=" * 70)

    if args.dataset == "gsm8k" or args.dataset == "all":
        gsm8k_dir = os.path.join(args.output_dir, "gsm8k")
        download_gsm8k(gsm8k_dir)

    if args.dataset == "math" or args.dataset == "all":
        math_dir = os.path.join(args.output_dir, "math")
        download_math_dataset(math_dir)

    print("\n" + "=" * 70)
    print("✅ All done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
