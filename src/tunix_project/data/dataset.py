"""
Dataset loading and management for GSM8K and other reasoning datasets
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict


class GSM8KDataset:
    """
    GSM8K (Grade School Math 8K) dataset handler

    This dataset contains math word problems requiring multi-step reasoning.
    Each example has a question and an answer with step-by-step solution.
    """

    def __init__(self, data_dir: str = "data/raw/gsm8k"):
        """
        Initialize GSM8K dataset

        Args:
            data_dir: Directory containing train.json and test.json
        """
        self.data_dir = Path(data_dir)
        self.train_data: Optional[List[Dict]] = None
        self.test_data: Optional[List[Dict]] = None

    def load(self) -> DatasetDict:
        """
        Load GSM8K dataset from JSON files

        Returns:
            DatasetDict with train and test splits
        """
        print(f"📂 Loading GSM8K dataset from {self.data_dir}")

        # Load train split
        train_path = self.data_dir / "train.json"
        if not train_path.exists():
            raise FileNotFoundError(
                f"Train file not found: {train_path}\n"
                f"Please run: python scripts/download_data.py --dataset gsm8k"
            )

        with open(train_path, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)

        # Load test split
        test_path = self.data_dir / "test.json"
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        with open(test_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        print(f"✅ Loaded {len(self.train_data)} train samples")
        print(f"✅ Loaded {len(self.test_data)} test samples")

        # Convert to HuggingFace Dataset format
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(self.train_data),
            'test': Dataset.from_list(self.test_data)
        })

        return dataset_dict

    @staticmethod
    def extract_answer(answer_text: str) -> str:
        """
        Extract the final numerical answer from GSM8K answer text

        GSM8K answers are in format:
        "Step 1: ...\\nStep 2: ...\\n#### 42"

        Args:
            answer_text: Full answer text with reasoning

        Returns:
            Final numerical answer as string
        """
        # Extract answer after ####
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        if match:
            # Remove commas from numbers (e.g., "1,000" -> "1000")
            answer = match.group(1).replace(',', '')
            return answer

        # Fallback: try to find last number in text
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
        if numbers:
            return numbers[-1].replace(',', '')

        return ""

    @staticmethod
    def extract_reasoning_steps(answer_text: str) -> str:
        """
        Extract reasoning steps from GSM8K answer text

        Args:
            answer_text: Full answer text with reasoning

        Returns:
            Reasoning steps without the final answer marker
        """
        # Remove the final answer (everything after ####)
        reasoning = re.split(r'####', answer_text)[0].strip()
        return reasoning

    @staticmethod
    def parse_example(example: Dict) -> Tuple[str, str, str]:
        """
        Parse a GSM8K example into question, reasoning, and answer

        Args:
            example: Dictionary with 'question' and 'answer' keys

        Returns:
            Tuple of (question, reasoning_steps, final_answer)
        """
        question = example['question']
        answer_text = example['answer']

        reasoning = GSM8KDataset.extract_reasoning_steps(answer_text)
        final_answer = GSM8KDataset.extract_answer(answer_text)

        return question, reasoning, final_answer

    def get_example(self, split: str = 'train', index: int = 0) -> Dict:
        """
        Get a single example from the dataset

        Args:
            split: 'train' or 'test'
            index: Index of example

        Returns:
            Dictionary with parsed example
        """
        data = self.train_data if split == 'train' else self.test_data

        if data is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        example = data[index]
        question, reasoning, answer = self.parse_example(example)

        return {
            'question': question,
            'reasoning': reasoning,
            'answer': answer,
            'full_answer_text': example['answer']
        }

    def print_example(self, split: str = 'train', index: int = 0) -> None:
        """
        Pretty print an example from the dataset

        Args:
            split: 'train' or 'test'
            index: Index of example
        """
        example = self.get_example(split, index)

        print("=" * 70)
        print(f"📝 GSM8K Example (split={split}, index={index})")
        print("=" * 70)
        print(f"\n❓ Question:\n{example['question']}\n")
        print(f"💭 Reasoning Steps:\n{example['reasoning']}\n")
        print(f"✅ Answer: {example['answer']}")
        print("=" * 70)


def load_dataset(
    dataset_name: str = "gsm8k",
    data_dir: str = "data/raw"
) -> DatasetDict:
    """
    Load dataset by name

    Args:
        dataset_name: Name of dataset ('gsm8k', 'math', etc.)
        data_dir: Base directory for datasets

    Returns:
        DatasetDict with train/test splits
    """
    if dataset_name == "gsm8k":
        dataset_path = Path(data_dir) / "gsm8k"
        loader = GSM8KDataset(str(dataset_path))
        return loader.load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Example usage
if __name__ == "__main__":
    # Load dataset
    loader = GSM8KDataset("data/raw/gsm8k")
    dataset = loader.load()

    # Print example
    loader.print_example(split='train', index=0)

    # Show dataset info
    print(f"\n📊 Dataset Info:")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
