"""
Process Reward Modeling (PRM)

Instead of only rewarding final answer, reward EACH STEP in the reasoning process.
This provides much richer learning signal!

Inspired by OpenAI's process supervision approach.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


class ProcessRewardModel:
    """
    Assign rewards to individual reasoning steps

    This allows the model to learn which intermediate steps are good,
    not just whether the final answer is correct.
    """

    def __init__(
        self,
        step_correctness_weight: float = 0.4,
        step_necessity_weight: float = 0.3,
        step_clarity_weight: float = 0.3
    ):
        """
        Initialize process reward model

        Args:
            step_correctness_weight: Weight for step correctness
            step_necessity_weight: Weight for step necessity
            step_clarity_weight: Weight for step clarity
        """
        self.step_correctness_weight = step_correctness_weight
        self.step_necessity_weight = step_necessity_weight
        self.step_clarity_weight = step_clarity_weight

    def evaluate_step(
        self,
        step: str,
        step_index: int,
        all_steps: List[str],
        question: str = "",
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single reasoning step

        Args:
            step: The step to evaluate
            step_index: Index of this step
            all_steps: All reasoning steps
            question: Original question
            ground_truth: Ground truth answer (if available)

        Returns:
            Dictionary with step-level rewards
        """
        # 1. Step correctness - is this step logically sound?
        correctness_score = self._score_step_correctness(
            step, step_index, all_steps
        )

        # 2. Step necessity - is this step needed?
        necessity_score = self._score_step_necessity(
            step, step_index, all_steps, question
        )

        # 3. Step clarity - is this step clear?
        clarity_score = self._score_step_clarity(step)

        # Combined reward
        total_reward = (
            self.step_correctness_weight * correctness_score +
            self.step_necessity_weight * necessity_score +
            self.step_clarity_weight * clarity_score
        )

        return {
            'step_reward': total_reward,
            'correctness': correctness_score,
            'necessity': necessity_score,
            'clarity': clarity_score
        }

    def _score_step_correctness(
        self,
        step: str,
        step_index: int,
        all_steps: List[str]
    ) -> float:
        """
        Score whether step is mathematically/logically correct

        Args:
            step: Step text
            step_index: Step index
            all_steps: All steps

        Returns:
            Correctness score (0-1)
        """
        score = 0.5  # Default: neutral

        # Check if step contains a calculation
        calc_pattern = r'(\d+)\s*([+\-*/×÷])\s*(\d+)\s*=\s*(\d+)'
        match = re.search(calc_pattern, step)

        if match:
            try:
                left = float(match.group(1))
                op = match.group(2)
                right = float(match.group(3))
                result = float(match.group(4))

                # Verify calculation
                if op in ['+']:
                    expected = left + right
                elif op in ['-']:
                    expected = left - right
                elif op in ['*', '×']:
                    expected = left * right
                elif op in ['/', '÷']:
                    expected = left / right if right != 0 else None
                else:
                    return score

                if expected is not None and abs(expected - result) < 0.01:
                    score = 1.0  # Correct calculation
                else:
                    score = 0.0  # Incorrect calculation

            except:
                pass

        return score

    def _score_step_necessity(
        self,
        step: str,
        step_index: int,
        all_steps: List[str],
        question: str
    ) -> float:
        """
        Score whether step is necessary for solving the problem

        Args:
            step: Step text
            step_index: Step index
            all_steps: All steps
            question: Original question

        Returns:
            Necessity score (0-1)
        """
        # Heuristic: steps that introduce new numbers or operations are necessary
        # Steps that just restate are not necessary

        # Extract numbers from step
        step_numbers = set(re.findall(r'\d+', step))

        # Extract numbers from previous steps
        prev_numbers = set()
        for prev_step in all_steps[:step_index]:
            prev_numbers.update(re.findall(r'\d+', prev_step))

        # If step introduces new numbers (from calculations), it's probably necessary
        new_numbers = step_numbers - prev_numbers

        if new_numbers or any(op in step for op in ['+', '-', '*', '/', '×', '÷', '=']):
            return 1.0  # Likely necessary
        else:
            return 0.3  # Might be redundant

    def _score_step_clarity(self, step: str) -> float:
        """
        Score how clear the step is

        Args:
            step: Step text

        Returns:
            Clarity score (0-1)
        """
        score = 0.0

        # Has reasonable length
        if 10 < len(step) < 200:
            score += 0.3

        # Contains action verb
        verbs = ['calculate', 'find', 'add', 'subtract', 'multiply', 'divide', 'use', 'get']
        if any(verb in step.lower() for verb in verbs):
            score += 0.3

        # Has punctuation
        if any(p in step for p in ['.', '!', '?', ':']):
            score += 0.2

        # Contains numbers (doing actual math)
        if re.search(r'\d+', step):
            score += 0.2

        return min(score, 1.0)

    def compute_process_rewards(
        self,
        response: str,
        question: str = "",
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Compute rewards for all steps in response

        Args:
            response: Full model response
            question: Original question
            ground_truth: Ground truth answer

        Returns:
            Dictionary with process rewards
        """
        # Extract steps
        from .reward import extract_reasoning_steps

        steps = extract_reasoning_steps(response)

        if not steps:
            return {
                'process_rewards': [],
                'avg_step_reward': 0.0,
                'num_steps': 0
            }

        # Evaluate each step
        step_rewards = []
        for i, step in enumerate(steps):
            step_eval = self.evaluate_step(
                step, i, steps, question, ground_truth
            )
            step_rewards.append(step_eval)

        # Aggregate
        avg_reward = np.mean([r['step_reward'] for r in step_rewards])

        # Progressive bonus: later steps should build on earlier ones
        progressive_bonus = 0.0
        for i in range(1, len(step_rewards)):
            if step_rewards[i]['step_reward'] >= step_rewards[i-1]['step_reward']:
                progressive_bonus += 0.1

        progressive_bonus = min(progressive_bonus, 0.3) / len(steps) if len(steps) > 1 else 0.0

        return {
            'process_rewards': step_rewards,
            'avg_step_reward': avg_reward,
            'progressive_bonus': progressive_bonus,
            'total_process_reward': avg_reward + progressive_bonus,
            'num_steps': len(steps)
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("PROCESS REWARD MODELING")
    print("=" * 70)

    # Test response
    test_response = """Step 1: Janet's ducks lay 16 eggs per day
Step 2: She uses 3 eggs for breakfast and 4 for muffins
Step 3: Total used = 3 + 4 = 7 eggs
Step 4: Remaining eggs = 16 - 7 = 9 eggs
Step 5: Revenue = 9 × 2 = 18 dollars

Answer: 18"""

    # Create PRM
    prm = ProcessRewardModel()

    # Compute process rewards
    result = prm.compute_process_rewards(test_response)

    print(f"\n📊 Process Reward Analysis:")
    print(f"   Total steps: {result['num_steps']}")
    print(f"   Avg step reward: {result['avg_step_reward']:.3f}")
    print(f"   Progressive bonus: {result['progressive_bonus']:.3f}")
    print(f"   Total process reward: {result['total_process_reward']:.3f}")

    print(f"\n🔍 Individual Step Rewards:")
    for i, step_reward in enumerate(result['process_rewards'], 1):
        print(f"   Step {i}: {step_reward['step_reward']:.3f}")
        print(f"     ├─ Correctness: {step_reward['correctness']:.3f}")
        print(f"     ├─ Necessity: {step_reward['necessity']:.3f}")
        print(f"     └─ Clarity: {step_reward['clarity']:.3f}")

    print("\n" + "=" * 70)
    print("✅ Process rewards provide step-level learning signal!")
    print("=" * 70)
