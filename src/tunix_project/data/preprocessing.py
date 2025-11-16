"""
Data preprocessing for chain-of-thought reasoning training
"""

import re
from typing import Dict, List, Optional, Tuple


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def format_number(num_str: str) -> str:
    """
    Format numerical answer consistently

    Args:
        num_str: Number as string

    Returns:
        Formatted number string
    """
    # Remove commas
    num_str = num_str.replace(',', '')

    # Try to convert to float then back to handle decimals
    try:
        num = float(num_str)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        return num_str


def split_reasoning_steps(reasoning: str) -> List[str]:
    """
    Split reasoning text into individual steps

    Args:
        reasoning: Reasoning text

    Returns:
        List of individual reasoning steps
    """
    # Split by common patterns: <<...>>, newlines, sentence endings
    # First, split by newlines
    steps = reasoning.split('\n')

    # Clean each step
    cleaned_steps = []
    for step in steps:
        step = clean_text(step)
        if step and len(step) > 5:  # Filter very short steps
            cleaned_steps.append(step)

    return cleaned_steps


def create_chain_of_thought_example(
    question: str,
    reasoning: str,
    answer: str,
    add_step_markers: bool = True
) -> str:
    """
    Create a formatted chain-of-thought example

    Args:
        question: The question/problem
        reasoning: Step-by-step reasoning
        answer: Final answer
        add_step_markers: Whether to add "Step 1:", "Step 2:", etc.

    Returns:
        Formatted example string
    """
    # Split reasoning into steps
    steps = split_reasoning_steps(reasoning)

    # Format steps
    if add_step_markers:
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            formatted_steps.append(f"Step {i}: {step}")
        reasoning_text = "\n".join(formatted_steps)
    else:
        reasoning_text = "\n".join(steps)

    # Create full example
    example = f"Question: {question}\n\n"
    example += f"Let's solve this step by step:\n{reasoning_text}\n\n"
    example += f"Answer: {answer}"

    return example


def prepare_training_example(
    question: str,
    reasoning: str,
    answer: str,
    prompt_template: Optional[str] = None
) -> Dict[str, str]:
    """
    Prepare a training example with input and target

    Args:
        question: The question/problem
        reasoning: Step-by-step reasoning
        answer: Final answer
        prompt_template: Optional custom prompt template

    Returns:
        Dictionary with 'input' and 'target' keys
    """
    if prompt_template is None:
        prompt_template = "Question: {question}\n\nLet's solve this step by step:"

    # Create input (question + prompt)
    input_text = prompt_template.format(question=question)

    # Create target (reasoning + answer)
    steps = split_reasoning_steps(reasoning)
    formatted_steps = [f"Step {i}: {step}" for i, step in enumerate(steps, 1)]

    target_text = "\n".join(formatted_steps)
    target_text += f"\n\nAnswer: {answer}"

    return {
        'input': input_text,
        'target': target_text,
        'question': question,
        'answer': answer
    }


def preprocess_dataset(
    examples: List[Dict],
    max_length: Optional[int] = None,
    add_step_markers: bool = True
) -> List[Dict]:
    """
    Preprocess a list of dataset examples

    Args:
        examples: List of examples with 'question' and 'answer' keys
        max_length: Maximum length for examples (optional)
        add_step_markers: Whether to add step markers

    Returns:
        List of preprocessed examples
    """
    from .dataset import GSM8KDataset

    preprocessed = []

    for example in examples:
        # Parse example
        question, reasoning, answer = GSM8KDataset.parse_example(example)

        # Clean text
        question = clean_text(question)
        reasoning = clean_text(reasoning)
        answer = format_number(answer)

        # Create training example
        training_example = prepare_training_example(
            question, reasoning, answer
        )

        # Check length if specified
        if max_length is not None:
            total_length = len(training_example['input']) + len(training_example['target'])
            if total_length > max_length:
                continue  # Skip examples that are too long

        preprocessed.append(training_example)

    return preprocessed


def create_validation_split(
    train_data: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create validation split from training data

    Args:
        train_data: Training data
        val_ratio: Ratio of validation data
        seed: Random seed

    Returns:
        Tuple of (new_train_data, val_data)
    """
    import random

    # Set seed for reproducibility
    random.seed(seed)

    # Shuffle data
    shuffled_data = train_data.copy()
    random.shuffle(shuffled_data)

    # Split
    val_size = int(len(shuffled_data) * val_ratio)
    val_data = shuffled_data[:val_size]
    new_train_data = shuffled_data[val_size:]

    print(f"📊 Split created:")
    print(f"   Train: {len(new_train_data)} examples")
    print(f"   Validation: {len(val_data)} examples")

    return new_train_data, val_data


# Example usage
if __name__ == "__main__":
    # Example GSM8K data
    example = {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
    }

    from .dataset import GSM8KDataset

    # Parse example
    question, reasoning, answer = GSM8KDataset.parse_example(example)

    # Create chain-of-thought example
    cot_example = create_chain_of_thought_example(question, reasoning, answer)
    print(cot_example)

    print("\n" + "="*70 + "\n")

    # Create training example
    training_example = prepare_training_example(question, reasoning, answer)
    print("Input:", training_example['input'])
    print("\nTarget:", training_example['target'])
