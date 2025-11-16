"""
Advanced Reward Function with Multiple Sophisticated Criteria

This extends the basic reward function with:
1. Step coherence - Do steps follow logically?
2. Mathematical rigor - Are calculations correct?
3. Explanation quality - Is it pedagogical?
4. Progressive correctness - Partial credit for intermediate steps
5. Efficiency - Optimal number of steps?

These improvements can boost training effectiveness significantly!
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def score_step_coherence(steps: List[str], question: str = "") -> float:
    """
    Score how well steps flow logically from one to another

    Criteria:
    - Each step references previous steps or question
    - Steps build progressively toward answer
    - No logical jumps or inconsistencies

    Args:
        steps: List of reasoning steps
        question: Original question (for context)

    Returns:
        Coherence score (0-1)
    """
    if not steps or len(steps) < 2:
        return 0.0

    score = 0.0
    num_steps = len(steps)

    # 1. Check for connective words/phrases (30%)
    connectives = [
        'therefore', 'thus', 'so', 'then', 'next', 'now',
        'since', 'because', 'as', 'this means', 'from this'
    ]

    steps_with_connectives = sum(
        1 for step in steps[1:]  # Skip first step
        if any(conn in step.lower() for conn in connectives)
    )

    if num_steps > 1:
        score += 0.3 * (steps_with_connectives / (num_steps - 1))

    # 2. Reference to previous results (30%)
    # Look for patterns like "from step 1", "using the above", numbers from prev steps
    references = 0
    for i, step in enumerate(steps[1:], 1):
        # Check if step mentions results from previous steps
        # Look for numbers that appeared in previous steps
        prev_numbers = set()
        for prev_step in steps[:i]:
            nums = re.findall(r'\d+', prev_step)
            prev_numbers.update(nums)

        curr_numbers = set(re.findall(r'\d+', step))

        # If current step uses numbers from previous steps
        if prev_numbers & curr_numbers:
            references += 1

    if num_steps > 1:
        score += 0.3 * (references / (num_steps - 1))

    # 3. Progressive complexity (20%)
    # Early steps should be simpler than later steps
    lengths = [len(step) for step in steps]
    is_progressive = all(
        lengths[i] <= lengths[i + 1] * 1.5  # Allow some flexibility
        for i in range(len(lengths) - 1)
    )
    if is_progressive:
        score += 0.2

    # 4. No contradictions (20%)
    # Check if steps contradict each other (simple heuristic)
    contradictions = 0
    for i in range(len(steps) - 1):
        for j in range(i + 1, len(steps)):
            # Very simple check: if same numbers appear with different operations
            # This is a placeholder - can be made more sophisticated
            pass

    # Assume no contradictions for now (can be enhanced)
    score += 0.2

    return min(score, 1.0)


def score_mathematical_rigor(steps: List[str]) -> float:
    """
    Score mathematical correctness and rigor

    Criteria:
    - Calculations are shown explicitly
    - Operations are correct (when verifiable)
    - Units/context are maintained
    - No mathematical errors

    Args:
        steps: List of reasoning steps

    Returns:
        Mathematical rigor score (0-1)
    """
    if not steps:
        return 0.0

    score = 0.0
    num_steps = len(steps)

    # 1. Explicit calculations (40%)
    # Look for pattern: "X op Y = Z"
    calc_pattern = r'\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+'
    steps_with_calcs = sum(1 for step in steps if re.search(calc_pattern, step))

    if num_steps > 0:
        score += 0.4 * (steps_with_calcs / num_steps)

    # 2. Verify calculations (40%)
    verified_calcs = 0
    total_calcs = 0

    for step in steps:
        # Find calculations like "3 + 4 = 7"
        calculations = re.findall(
            r'(\d+)\s*([+\-*/×÷])\s*(\d+)\s*=\s*(\d+)',
            step
        )

        for calc in calculations:
            total_calcs += 1
            try:
                left = float(calc[0])
                op = calc[1]
                right = float(calc[2])
                result = float(calc[3])

                # Verify
                if op in ['+']:
                    expected = left + right
                elif op in ['-']:
                    expected = left - right
                elif op in ['*', '×']:
                    expected = left * right
                elif op in ['/', '÷']:
                    expected = left / right if right != 0 else None
                else:
                    continue

                if expected is not None and abs(expected - result) < 0.01:
                    verified_calcs += 1

            except:
                pass

    if total_calcs > 0:
        score += 0.4 * (verified_calcs / total_calcs)
    else:
        # No verifiable calculations, give partial credit
        score += 0.2

    # 3. Units consistency (20%)
    # Check if units/context are maintained (simple heuristic)
    # Look for currency symbols, units, etc.
    has_units = any(
        re.search(r'(\$|eggs|dollars|items|pounds|kg)', step, re.IGNORECASE)
        for step in steps
    )
    if has_units:
        score += 0.2

    return min(score, 1.0)


def score_explanation_quality(steps: List[str], question: str = "") -> float:
    """
    Score pedagogical quality of explanation

    Criteria:
    - Clear language
    - Avoids jargon
    - Provides intuition
    - Easy to follow

    Args:
        steps: List of reasoning steps
        question: Original question

    Returns:
        Explanation quality score (0-1)
    """
    if not steps:
        return 0.0

    score = 0.0

    # 1. Natural language (30%)
    # Steps should have verbs, proper sentences
    verbs = [
        'calculate', 'find', 'determine', 'add', 'subtract',
        'multiply', 'divide', 'solve', 'use', 'get', 'need'
    ]

    steps_with_verbs = sum(
        1 for step in steps
        if any(verb in step.lower() for verb in verbs)
    )

    if len(steps) > 0:
        score += 0.3 * (steps_with_verbs / len(steps))

    # 2. Complete sentences (30%)
    # Check for punctuation
    steps_with_punctuation = sum(
        1 for step in steps
        if any(p in step for p in ['.', '!', '?'])
    )

    if len(steps) > 0:
        score += 0.3 * (steps_with_punctuation / len(steps))

    # 3. Not too technical (20%)
    # Avoid overly complex words
    complex_words = ['algorithm', 'heuristic', 'optimization', 'derivation']
    has_complexity = any(
        word in ' '.join(steps).lower()
        for word in complex_words
    )

    if not has_complexity:
        score += 0.2

    # 4. Provides context (20%)
    # Mentions what we're trying to find
    context_phrases = ['we need', 'we want', 'we are looking for', 'to find']
    has_context = any(
        phrase in ' '.join(steps).lower()
        for phrase in context_phrases
    )

    if has_context:
        score += 0.2

    return min(score, 1.0)


def score_partial_correctness(
    steps: List[str],
    ground_truth: str,
    intermediate_checks: Optional[List[Tuple[str, str]]] = None
) -> float:
    """
    Score partial correctness - give credit for correct intermediate steps

    Args:
        steps: List of reasoning steps
        ground_truth: Final correct answer
        intermediate_checks: List of (step_pattern, correct_value) tuples

    Returns:
        Partial correctness score (0-1)
    """
    if not steps:
        return 0.0

    # If no intermediate checks provided, return basic score
    if intermediate_checks is None:
        # Check if any intermediate calculations lead toward correct answer
        try:
            target = float(ground_truth)

            # Extract all numbers from steps
            all_numbers = []
            for step in steps:
                nums = re.findall(r'\d+(?:\.\d+)?', step)
                all_numbers.extend([float(n) for n in nums])

            # Check if target appears in intermediate steps
            if target in all_numbers:
                return 1.0

            # Check if we're getting close (within 20%)
            if all_numbers:
                closest = min(all_numbers, key=lambda x: abs(x - target))
                if abs(closest - target) / max(target, 1) < 0.2:
                    return 0.5

            return 0.0
        except:
            return 0.0

    # With intermediate checks
    correct_intermediates = 0
    for pattern, correct_value in intermediate_checks:
        # Check if this intermediate value appears correctly
        for step in steps:
            if re.search(pattern, step) and correct_value in step:
                correct_intermediates += 1
                break

    return correct_intermediates / len(intermediate_checks) if intermediate_checks else 0.0


def score_efficiency(steps: List[str], question: str = "") -> float:
    """
    Score efficiency - optimal number of steps, no redundancy

    Args:
        steps: List of reasoning steps
        question: Original question

    Returns:
        Efficiency score (0-1)
    """
    if not steps:
        return 0.0

    num_steps = len(steps)
    score = 0.0

    # 1. Optimal length (40%)
    # For GSM8K, ideal is 3-6 steps
    if 3 <= num_steps <= 6:
        score += 0.4
    elif 2 <= num_steps <= 8:
        score += 0.3
    elif num_steps < 2:
        score += 0.1
    else:
        # Penalize too many steps
        score += max(0.4 - 0.05 * (num_steps - 6), 0.0)

    # 2. No redundancy (30%)
    # Check for repeated calculations or info
    unique_steps = len(set(steps))
    redundancy_ratio = unique_steps / num_steps
    score += 0.3 * redundancy_ratio

    # 3. Conciseness (30%)
    # Steps shouldn't be overly verbose
    avg_length = np.mean([len(step) for step in steps])

    if avg_length < 200:  # Reasonable length
        score += 0.3
    else:
        score += max(0.3 - 0.001 * (avg_length - 200), 0.0)

    return min(score, 1.0)


def compute_advanced_reward(
    response: str,
    ground_truth: str,
    question: str = "",
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive advanced reward

    Args:
        response: Model's generated response
        ground_truth: Correct answer
        question: Original question
        weights: Optional custom weights for each component

    Returns:
        Dictionary with all reward components
    """
    from .reward import (
        extract_final_answer,
        extract_reasoning_steps,
        check_answer_correctness,
        score_reasoning_quality,
        score_clarity
    )

    # Default weights
    if weights is None:
        weights = {
            'correctness': 0.30,         # Slightly reduced from 0.50
            'reasoning_quality': 0.15,   # Slightly reduced from 0.30
            'clarity': 0.10,             # Reduced from 0.20
            'coherence': 0.15,           # NEW
            'mathematical_rigor': 0.15,  # NEW
            'explanation_quality': 0.05, # NEW
            'partial_correctness': 0.05, # NEW
            'efficiency': 0.05,          # NEW
        }

    # Extract components
    predicted_answer = extract_final_answer(response)
    steps = extract_reasoning_steps(response)

    # Basic scores (from original reward function)
    is_correct = check_answer_correctness(predicted_answer, ground_truth)
    correctness_score = 1.0 if is_correct else 0.0
    reasoning_score = score_reasoning_quality(steps, question)
    clarity_score = score_clarity(response, steps)

    # Advanced scores
    coherence_score = score_step_coherence(steps, question)
    rigor_score = score_mathematical_rigor(steps)
    explanation_score = score_explanation_quality(steps, question)
    partial_score = score_partial_correctness(steps, ground_truth)
    efficiency_score = score_efficiency(steps, question)

    # Weighted combination
    total_reward = (
        weights['correctness'] * correctness_score +
        weights['reasoning_quality'] * reasoning_score +
        weights['clarity'] * clarity_score +
        weights['coherence'] * coherence_score +
        weights['mathematical_rigor'] * rigor_score +
        weights['explanation_quality'] * explanation_score +
        weights['partial_correctness'] * partial_score +
        weights['efficiency'] * efficiency_score
    )

    return {
        'total_reward': total_reward,

        # Basic components
        'correctness_score': correctness_score,
        'reasoning_score': reasoning_score,
        'clarity_score': clarity_score,

        # Advanced components
        'coherence_score': coherence_score,
        'mathematical_rigor_score': rigor_score,
        'explanation_quality_score': explanation_score,
        'partial_correctness_score': partial_score,
        'efficiency_score': efficiency_score,

        # Metadata
        'is_correct': is_correct,
        'num_steps': len(steps),
        'predicted_answer': predicted_answer,
        'ground_truth': ground_truth,
        'weights': weights
    }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED REWARD FUNCTION")
    print("=" * 70)

    # Test response
    test_response = """Step 1: First, calculate total eggs used for breakfast and muffins
Step 2: Janet uses 3 eggs for breakfast
Step 3: She uses 4 eggs for muffins
Step 4: Therefore, total used = 3 + 4 = 7 eggs
Step 5: Subtract from total: 16 - 7 = 9 eggs remaining
Step 6: Calculate revenue: 9 eggs × $2 = $18

Answer: 18"""

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast and uses four for muffins. How much does she make selling the rest at $2 each?"

    # Compute reward
    reward = compute_advanced_reward(test_response, "18", question)

    print("\n🎯 Advanced Reward Breakdown:")
    print(f"   Total Reward: {reward['total_reward']:.3f}\n")

    print("   Basic Components:")
    print(f"   ├─ Correctness: {reward['correctness_score']:.3f} (weight: {reward['weights']['correctness']})")
    print(f"   ├─ Reasoning: {reward['reasoning_score']:.3f} (weight: {reward['weights']['reasoning_quality']})")
    print(f"   └─ Clarity: {reward['clarity_score']:.3f} (weight: {reward['weights']['clarity']})\n")

    print("   Advanced Components:")
    print(f"   ├─ Coherence: {reward['coherence_score']:.3f} (weight: {reward['weights']['coherence']})")
    print(f"   ├─ Math Rigor: {reward['mathematical_rigor_score']:.3f} (weight: {reward['weights']['mathematical_rigor']})")
    print(f"   ├─ Explanation: {reward['explanation_quality_score']:.3f} (weight: {reward['weights']['explanation_quality']})")
    print(f"   ├─ Partial Correct: {reward['partial_correctness_score']:.3f} (weight: {reward['weights']['partial_correctness']})")
    print(f"   └─ Efficiency: {reward['efficiency_score']:.3f} (weight: {reward['weights']['efficiency']})")

    print("\n" + "=" * 70)
    print("✅ Advanced reward function provides richer training signal!")
    print("=" * 70)
