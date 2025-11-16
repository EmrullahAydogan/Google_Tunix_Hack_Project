#!/usr/bin/env python3
"""
Preprocess GSM8K dataset for training

This script:
1. Loads raw GSM8K data
2. Preprocesses and formats for chain-of-thought training
3. Creates train/val/test splits
4. Saves processed data

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input data/raw/gsm8k --output data/processed
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tunix_project.data.dataset import GSM8KDataset
from tunix_project.data.preprocessing import (
    preprocess_dataset,
    create_validation_split,
    prepare_training_example
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GSM8K dataset for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gsm8k",
        help="Input directory with raw data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GSM8K DATASET PREPROCESSING")
    print("=" * 70)

    # Load dataset
    print(f"\n📂 Loading dataset from {args.input}")
    loader = GSM8KDataset(args.input)
    dataset = loader.load()

    train_data = dataset['train']
    test_data = dataset['test']

    print(f"✅ Loaded:")
    print(f"   Train: {len(train_data)} examples")
    print(f"   Test: {len(test_data)} examples")

    # Preprocess training data
    print(f"\n🔄 Preprocessing training data...")
    train_examples = []
    for example in train_data:
        train_examples.append(example)

    processed_train = preprocess_dataset(
        train_examples,
        max_length=args.max_length,
        add_step_markers=True
    )

    print(f"✅ Preprocessed {len(processed_train)} training examples")

    # Create validation split
    print(f"\n✂️ Creating validation split (ratio: {args.val_ratio})...")
    final_train, val_data = create_validation_split(
        processed_train,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Preprocess test data
    print(f"\n🔄 Preprocessing test data...")
    test_examples = []
    for example in test_data:
        test_examples.append(example)

    processed_test = preprocess_dataset(
        test_examples,
        max_length=args.max_length,
        add_step_markers=True
    )

    print(f"✅ Preprocessed {len(processed_test)} test examples")

    # Save processed data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving processed data to {output_dir}")

    # Save splits
    splits = {
        'train': final_train,
        'validation': val_data,
        'test': processed_test
    }

    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {split_name}.json: {len(split_data)} examples")

    # Save metadata
    metadata = {
        'dataset': 'gsm8k',
        'num_train': len(final_train),
        'num_val': len(val_data),
        'num_test': len(processed_test),
        'val_ratio': args.val_ratio,
        'max_length': args.max_length,
        'seed': args.seed,
        'preprocessing': {
            'add_step_markers': True,
            'format': 'chain_of_thought'
        }
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ metadata.json")

    # Show example
    print(f"\n📝 Example from preprocessed data:")
    print("=" * 70)
    example = final_train[0]
    print(f"Input:\n{example['input']}\n")
    print(f"Target:\n{example['target']}")
    print("=" * 70)

    # Summary
    print(f"\n✅ Preprocessing complete!")
    print(f"📊 Final dataset sizes:")
    print(f"   Training: {len(final_train)}")
    print(f"   Validation: {len(val_data)}")
    print(f"   Test: {len(processed_test)}")
    print(f"   Total: {len(final_train) + len(val_data) + len(processed_test)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
