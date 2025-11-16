"""
Reward function for Chain-of-Thought reasoning

This is the CRITICAL component that teaches the model to show its work.
The reward function evaluates:
1. Correctness - Is the final answer correct?
2. Reasoning Quality - Are the reasoning steps logical and clear?
3. Clarity - Is the explanation understandable?
"""

import re
from typing import Dict, List, Optional, Tuple


def extract_final_answer(response: str) -> str:
    """
    Extract the final answer from model response

    Args:
        response: Model's generated response

    Returns:
        Extracted answer string
    """
    # Look for "Answer: X" pattern
    answer_pattern = r'Answer:\s*([^\n]+)'
    match = re.search(answer_pattern, response, re.IGNORECASE)

    if match:
        answer = match.group(1).strip()
        # Extract number from answer
        number_match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer)
        if number_match:
            return number_match.group(0).replace(',', '')

    # Fallback: try to find last number in response
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""


def extract_reasoning_steps(response: str) -> List[str]:
    """
    Extract reasoning steps from model response

    Args:
        response: Model's generated response

    Returns:
        List of reasoning steps
    """
    # Look for "Step X:" pattern
    step_pattern = r'Step \d+:(.+?)(?=Step \d+:|Answer:|$)'
    steps = re.findall(step_pattern, response, re.DOTALL | re.IGNORECASE)

    # Clean steps
    cleaned_steps = [step.strip() for step in steps if step.strip()]

    # If no explicit steps found, try splitting by newlines
    if not cleaned_steps:
        lines = response.split('\n')
        cleaned_steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

    return cleaned_steps


def check_answer_correctness(
    predicted_answer: str,
    ground_truth_answer: str,
    tolerance: float = 1e-4
) -> bool:
    """
    Check if predicted answer matches ground truth

    Args:
        predicted_answer: Model's predicted answer
        ground_truth_answer: Correct answer
        tolerance: Numerical tolerance for float comparison

    Returns:
        True if answers match, False otherwise
    """
    # Extract numbers
    pred_num = extract_number(predicted_answer)
    truth_num = extract_number(ground_truth_answer)

    if pred_num is None or truth_num is None:
        # String comparison if not numbers
        return predicted_answer.strip().lower() == ground_truth_answer.strip().lower()

    # Numerical comparison
    return abs(pred_num - truth_num) < tolerance


def extract_number(text: str) -> Optional[float]:
    """
    Extract number from text

    Args:
        text: Text containing a number

    Returns:
        Float number or None if no valid number found
    """
    # Remove commas
    text = text.replace(',', '')

    # Try to find number
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None

    return None


def score_reasoning_quality(steps: List[str], question: str = "") -> float:
    """
    Score the quality of reasoning steps

    Criteria:
    - Number of steps (more steps = more detailed, up to a point)
    - Step length (not too short, not too long)
    - Contains calculations (indicated by numbers, operators)
    - Logical flow (each step builds on previous)

    Args:
        steps: List of reasoning steps
        question: Original question (optional, for context)

    Returns:
        Score between 0.0 and 1.0
    """
    if not steps:
        return 0.0

    score = 0.0
    num_steps = len(steps)

    # 1. Number of steps (25 points)
    # Ideal: 2-8 steps for GSM8K
    if 2 <= num_steps <= 8:
        step_score = 0.25
    elif num_steps == 1:
        step_score = 0.1
    elif num_steps > 8:
        step_score = max(0.25 - 0.02 * (num_steps - 8), 0.1)
    else:
        step_score = 0.0

    score += step_score

    # 2. Step length (25 points)
    # Steps should be substantial but not overly verbose
    avg_length = sum(len(step) for step in steps) / num_steps
    if 20 <= avg_length <= 150:
        length_score = 0.25
    elif avg_length < 20:
        length_score = 0.1
    else:
        length_score = max(0.25 - 0.001 * (avg_length - 150), 0.1)

    score += length_score

    # 3. Contains calculations (25 points)
    calculation_count = 0
    for step in steps:
        # Look for mathematical operations
        if re.search(r'\d+\s*[+\-*/×÷]\s*\d+', step):
            calculation_count += 1

    if calculation_count > 0:
        calc_score = min(0.25, calculation_count * 0.1)
    else:
        calc_score = 0.0

    score += calc_score

    # 4. Step completeness (25 points)
    # Each step should have some substance
    complete_steps = sum(1 for step in steps if len(step) > 15 and any(c.isdigit() for c in step))
    completeness_score = min(0.25, (complete_steps / max(num_steps, 1)) * 0.25)

    score += completeness_score

    return min(score, 1.0)


def score_clarity(response: str, steps: List[str]) -> float:
    """
    Score the clarity and readability of the response

    Criteria:
    - Proper formatting (step markers, answer marker)
    - Clear language (not too complex, not too simple)
    - Coherent structure

    Args:
        response: Full model response
        steps: Extracted reasoning steps

    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0

    # 1. Has explicit step markers (30 points)
    has_step_markers = bool(re.search(r'Step \d+:', response, re.IGNORECASE))
    if has_step_markers:
        score += 0.3

    # 2. Has explicit answer marker (30 points)
    has_answer_marker = bool(re.search(r'Answer:', response, re.IGNORECASE))
    if has_answer_marker:
        score += 0.3

    # 3. Proper punctuation and grammar (20 points)
    # Check for basic punctuation
    has_punctuation = any(char in response for char in '.!?')
    if has_punctuation:
        score += 0.2

    # 4. Not repetitive (20 points)
    # Check for excessive repetition
    unique_steps = len(set(steps))
    if len(steps) > 0:
        repetition_score = min(0.2, (unique_steps / len(steps)) * 0.2)
        score += repetition_score

    return min(score, 1.0)


def compute_reward(
    response: str,
    ground_truth_answer: str,
    question: str = "",
    correctness_weight: float = 0.5,
    reasoning_weight: float = 0.3,
    clarity_weight: float = 0.2
) -> Dict[str, float]:
    """
    Compute comprehensive reward for a model response

    This is the main reward function used for training.

    Args:
        response: Model's generated response
        ground_truth_answer: Correct answer
        question: Original question (optional)
        correctness_weight: Weight for correctness (default: 0.5)
        reasoning_weight: Weight for reasoning quality (default: 0.3)
        clarity_weight: Weight for clarity (default: 0.2)

    Returns:
        Dictionary with reward components and total reward
    """
    # Extract components from response
    predicted_answer = extract_final_answer(response)
    reasoning_steps = extract_reasoning_steps(response)

    # 1. Correctness score
    is_correct = check_answer_correctness(predicted_answer, ground_truth_answer)
    correctness_score = 1.0 if is_correct else 0.0

    # 2. Reasoning quality score
    reasoning_score = score_reasoning_quality(reasoning_steps, question)

    # 3. Clarity score
    clarity_score = score_clarity(response, reasoning_steps)

    # Compute weighted total
    total_reward = (
        correctness_weight * correctness_score +
        reasoning_weight * reasoning_score +
        clarity_weight * clarity_score
    )

    return {
        'total_reward': total_reward,
        'correctness_score': correctness_score,
        'reasoning_score': reasoning_score,
        'clarity_score': clarity_score,
        'is_correct': is_correct,
        'num_steps': len(reasoning_steps),
        'predicted_answer': predicted_answer,
        'ground_truth_answer': ground_truth_answer
    }


def batch_compute_rewards(
    responses: List[str],
    ground_truth_answers: List[str],
    questions: Optional[List[str]] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute rewards for a batch of responses

    Args:
        responses: List of model responses
        ground_truth_answers: List of correct answers
        questions: List of original questions (optional)
        **kwargs: Additional arguments for compute_reward

    Returns:
        List of reward dictionaries
    """
    if questions is None:
        questions = [""] * len(responses)

    rewards = []
    for response, truth, question in zip(responses, ground_truth_answers, questions):
        reward = compute_reward(response, truth, question, **kwargs)
        rewards.append(reward)

    return rewards


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("REWARD FUNCTION DEMO")
    print("=" * 70)

    # Example question and answer
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    ground_truth = "18"

    # Good response with clear reasoning
    good_response = """Step 1: Janet's ducks lay 16 eggs per day
Step 2: She eats 3 eggs for breakfast
Step 3: She uses 4 eggs for muffins
Step 4: Total eggs used: 3 + 4 = 7 eggs
Step 5: Remaining eggs to sell: 16 - 7 = 9 eggs
Step 6: Revenue from selling eggs: 9 × $2 = $18

Answer: 18"""

    # Poor response without clear steps
    poor_response = """The answer is 18."""

    # Medium response with some reasoning
    medium_response = """Janet has 16 eggs. She uses some for breakfast and muffins, leaving 9 eggs. She sells them for $2 each.
Answer: 18"""

    # Test each response
    for name, response in [("Good", good_response), ("Medium", medium_response), ("Poor", poor_response)]:
        print(f"\n{name} Response:")
        print("-" * 70)
        print(response)
        print("-" * 70)

        reward = compute_reward(response, ground_truth, question)

        print(f"\nReward Breakdown:")
        print(f"  Total Reward: {reward['total_reward']:.3f}")
        print(f"  ├─ Correctness: {reward['correctness_score']:.3f} (weight: 0.5)")
        print(f"  ├─ Reasoning Quality: {reward['reasoning_score']:.3f} (weight: 0.3)")
        print(f"  └─ Clarity: {reward['clarity_score']:.3f} (weight: 0.2)")
        print(f"\nDetails:")
        print(f"  Is Correct: {reward['is_correct']}")
        print(f"  Number of Steps: {reward['num_steps']}")
        print(f"  Predicted Answer: {reward['predicted_answer']}")
        print("=" * 70)
