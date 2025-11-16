"""
Evaluation metrics for reasoning quality assessment
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def exact_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer exactly matches ground truth

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if exact match, False otherwise
    """
    # Normalize strings
    pred_clean = predicted.strip().lower()
    truth_clean = ground_truth.strip().lower()

    return pred_clean == truth_clean


def numerical_match(predicted: str, ground_truth: str, tolerance: float = 1e-4) -> bool:
    """
    Check if predicted numerical answer matches ground truth

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Numerical tolerance

    Returns:
        True if numbers match within tolerance
    """
    def extract_number(text):
        # Remove commas
        text = text.replace(',', '')
        # Find number
        match = re.search(r'-?\d+(?:\.\d+)?', text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None

    pred_num = extract_number(predicted)
    truth_num = extract_number(ground_truth)

    if pred_num is None or truth_num is None:
        return False

    return abs(pred_num - truth_num) < tolerance


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy over a list of predictions

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = sum(
        numerical_match(pred, truth)
        for pred, truth in zip(predictions, ground_truths)
    )

    return correct / len(predictions) if len(predictions) > 0 else 0.0


def count_reasoning_steps(response: str) -> int:
    """
    Count number of reasoning steps in response

    Args:
        response: Model response

    Returns:
        Number of steps
    """
    # Look for "Step X:" pattern
    steps = re.findall(r'Step \d+:', response, re.IGNORECASE)
    return len(steps)


def has_explicit_answer(response: str) -> bool:
    """
    Check if response has explicit answer marker

    Args:
        response: Model response

    Returns:
        True if has "Answer:" marker
    """
    return bool(re.search(r'Answer:', response, re.IGNORECASE))


def compute_reasoning_metrics(responses: List[str]) -> Dict[str, float]:
    """
    Compute reasoning quality metrics

    Args:
        responses: List of model responses

    Returns:
        Dictionary of metrics
    """
    num_responses = len(responses)

    if num_responses == 0:
        return {
            'avg_num_steps': 0.0,
            'pct_with_steps': 0.0,
            'pct_with_answer_marker': 0.0,
            'avg_response_length': 0.0
        }

    # Count steps
    num_steps = [count_reasoning_steps(resp) for resp in responses]
    avg_steps = np.mean(num_steps)
    pct_with_steps = sum(1 for n in num_steps if n > 0) / num_responses

    # Check for answer markers
    has_answer = [has_explicit_answer(resp) for resp in responses]
    pct_with_answer_marker = sum(has_answer) / num_responses

    # Response length
    lengths = [len(resp) for resp in responses]
    avg_length = np.mean(lengths)

    return {
        'avg_num_steps': avg_steps,
        'pct_with_steps': pct_with_steps * 100,
        'pct_with_answer_marker': pct_with_answer_marker * 100,
        'avg_response_length': avg_length
    }


def evaluate_model(
    predictions: List[str],
    ground_truths: List[str],
    responses: List[str]
) -> Dict[str, float]:
    """
    Comprehensive model evaluation

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        responses: List of full model responses

    Returns:
        Dictionary of evaluation metrics
    """
    # Accuracy
    accuracy = compute_accuracy(predictions, ground_truths)

    # Reasoning metrics
    reasoning_metrics = compute_reasoning_metrics(responses)

    # Combine all metrics
    metrics = {
        'accuracy': accuracy * 100,  # Convert to percentage
        **reasoning_metrics
    }

    return metrics


def print_evaluation_report(metrics: Dict[str, float], title: str = "Evaluation Report"):
    """
    Print formatted evaluation report

    Args:
        metrics: Dictionary of metrics
        title: Report title
    """
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

    # Accuracy
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.2f}%")

    # Reasoning metrics
    print(f"\n💭 Reasoning Quality:")
    print(f"  Average Steps: {metrics.get('avg_num_steps', 0):.1f}")
    print(f"  Responses with Steps: {metrics.get('pct_with_steps', 0):.1f}%")
    print(f"  Explicit Answer Marker: {metrics.get('pct_with_answer_marker', 0):.1f}%")

    # Length
    print(f"\n📝 Response Characteristics:")
    print(f"  Average Length: {metrics.get('avg_response_length', 0):.0f} chars")

    print("=" * 70 + "\n")


def compute_metrics(predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
    """
    Compute metrics for HuggingFace evaluate compatibility

    Args:
        predictions: List of prediction dictionaries
        references: List of reference dictionaries

    Returns:
        Dictionary of metrics
    """
    # Extract answers
    pred_answers = [p.get('answer', '') for p in predictions]
    ref_answers = [r.get('answer', '') for r in references]

    # Extract responses
    pred_responses = [p.get('response', '') for p in predictions]

    # Evaluate
    metrics = evaluate_model(pred_answers, ref_answers, pred_responses)

    return metrics


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION METRICS DEMO")
    print("=" * 70)

    # Sample data
    predictions = ["18", "42", "100"]
    ground_truths = ["18", "42", "99"]

    responses = [
        "Step 1: Calculate eggs\nStep 2: Multiply by price\nAnswer: 18",
        "Step 1: Add numbers\nAnswer: 42",
        "The answer is 100"
    ]

    # Evaluate
    metrics = evaluate_model(predictions, ground_truths, responses)

    # Print report
    print_evaluation_report(metrics)

    # Individual metrics
    print("Detailed Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
