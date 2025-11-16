"""
Curriculum Learning for Progressive Training

Start with easy examples, gradually increase difficulty.
This improves training stability and final performance.

Strategy: Easy (1-3 steps) → Medium (3-5 steps) → Hard (5+ steps)
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


class CurriculumLearning:
    """
    Curriculum learning strategy for chain-of-thought training

    Progressively increases problem difficulty during training
    """

    def __init__(
        self,
        difficulty_metric: str = 'num_steps',
        num_phases: int = 3,
        phase_epochs: Optional[List[int]] = None
    ):
        """
        Initialize curriculum learning

        Args:
            difficulty_metric: How to measure difficulty
                - 'num_steps': Number of reasoning steps
                - 'answer_magnitude': Size of the answer
                - 'question_length': Length of question
                - 'num_operations': Number of mathematical operations
            num_phases: Number of curriculum phases
            phase_epochs: Epochs per phase (default: equal split)
        """
        self.difficulty_metric = difficulty_metric
        self.num_phases = num_phases
        self.phase_epochs = phase_epochs

    def estimate_difficulty(self, example: Dict) -> float:
        """
        Estimate difficulty of an example

        Args:
            example: Training example with 'question', 'answer', 'reasoning'

        Returns:
            Difficulty score (higher = harder)
        """
        if self.difficulty_metric == 'num_steps':
            return self._count_steps(example.get('target', ''))

        elif self.difficulty_metric == 'answer_magnitude':
            return self._get_answer_magnitude(example.get('answer', '0'))

        elif self.difficulty_metric == 'question_length':
            return len(example.get('question', ''))

        elif self.difficulty_metric == 'num_operations':
            return self._count_operations(example.get('target', ''))

        else:
            # Default: number of steps
            return self._count_steps(example.get('target', ''))

    def _count_steps(self, text: str) -> int:
        """Count number of steps in reasoning"""
        steps = re.findall(r'Step \d+:', text, re.IGNORECASE)
        return len(steps)

    def _get_answer_magnitude(self, answer: str) -> float:
        """Get numerical magnitude of answer"""
        try:
            # Extract number
            match = re.search(r'-?\d+(?:\.\d+)?', answer)
            if match:
                return abs(float(match.group(0)))
        except:
            pass
        return 0.0

    def _count_operations(self, text: str) -> int:
        """Count mathematical operations"""
        operations = re.findall(r'[+\-*/×÷=]', text)
        return len(operations)

    def create_curriculum(
        self,
        dataset: List[Dict],
        shuffle_within_phase: bool = True
    ) -> List[List[Dict]]:
        """
        Create curriculum phases from dataset

        Args:
            dataset: List of training examples
            shuffle_within_phase: Shuffle examples within each phase

        Returns:
            List of phases, each containing a subset of examples
        """
        # Estimate difficulty for all examples
        difficulties = [
            (i, self.estimate_difficulty(ex))
            for i, ex in enumerate(dataset)
        ]

        # Sort by difficulty
        difficulties.sort(key=lambda x: x[1])

        # Split into phases
        phase_size = len(dataset) // self.num_phases
        phases = []

        for phase_idx in range(self.num_phases):
            start_idx = phase_idx * phase_size
            if phase_idx == self.num_phases - 1:
                # Last phase gets remaining examples
                end_idx = len(difficulties)
            else:
                end_idx = (phase_idx + 1) * phase_size

            # Get examples for this phase
            phase_indices = [idx for idx, _ in difficulties[start_idx:end_idx]]
            phase_examples = [dataset[i] for i in phase_indices]

            if shuffle_within_phase:
                np.random.shuffle(phase_examples)

            phases.append(phase_examples)

        return phases

    def get_phase_info(self, phases: List[List[Dict]]) -> List[Dict]:
        """
        Get statistics about each phase

        Args:
            phases: List of curriculum phases

        Returns:
            List of phase information dicts
        """
        phase_info = []

        for i, phase in enumerate(phases):
            difficulties = [self.estimate_difficulty(ex) for ex in phase]

            info = {
                'phase': i + 1,
                'num_examples': len(phase),
                'avg_difficulty': np.mean(difficulties),
                'min_difficulty': np.min(difficulties),
                'max_difficulty': np.max(difficulties),
                'std_difficulty': np.std(difficulties)
            }

            phase_info.append(info)

        return phase_info

    def print_curriculum_summary(self, phases: List[List[Dict]]):
        """Print curriculum summary"""
        phase_info = self.get_phase_info(phases)

        print(f"\n{'='*70}")
        print(f"CURRICULUM LEARNING SUMMARY")
        print(f"{'='*70}")
        print(f"Difficulty metric: {self.difficulty_metric}")
        print(f"Number of phases: {self.num_phases}")
        print(f"Total examples: {sum(len(p) for p in phases)}\n")

        for info in phase_info:
            print(f"Phase {info['phase']}:")
            print(f"  Examples: {info['num_examples']}")
            print(f"  Difficulty: {info['avg_difficulty']:.2f} ± {info['std_difficulty']:.2f}")
            print(f"  Range: [{info['min_difficulty']:.2f}, {info['max_difficulty']:.2f}]")
            print()

        print(f"{'='*70}\n")


class AdaptiveCurriculum:
    """
    Adaptive curriculum that adjusts based on training performance

    If model struggles on current difficulty, stay longer.
    If model does well, advance faster.
    """

    def __init__(
        self,
        base_curriculum: CurriculumLearning,
        performance_threshold: float = 0.7,
        patience: int = 2
    ):
        """
        Initialize adaptive curriculum

        Args:
            base_curriculum: Base curriculum learning object
            performance_threshold: Accuracy threshold to advance
            patience: Number of evaluations to wait before advancing
        """
        self.base_curriculum = base_curriculum
        self.performance_threshold = performance_threshold
        self.patience = patience

        self.current_phase = 0
        self.wait_counter = 0
        self.performance_history = []

    def should_advance(self, current_performance: float) -> bool:
        """
        Decide whether to advance to next phase

        Args:
            current_performance: Current accuracy on validation set

        Returns:
            Whether to advance
        """
        self.performance_history.append(current_performance)

        if current_performance >= self.performance_threshold:
            self.wait_counter += 1
        else:
            self.wait_counter = 0

        # Advance if consistently good performance
        if self.wait_counter >= self.patience:
            self.wait_counter = 0
            return True

        return False

    def get_current_phase_data(
        self,
        curriculum_phases: List[List[Dict]]
    ) -> List[Dict]:
        """
        Get data for current curriculum phase

        Args:
            curriculum_phases: All curriculum phases

        Returns:
            Current phase data
        """
        if self.current_phase >= len(curriculum_phases):
            # Use all data (post-curriculum)
            return [ex for phase in curriculum_phases for ex in phase]

        return curriculum_phases[self.current_phase]

    def advance_phase(self, curriculum_phases: List[List[Dict]]) -> bool:
        """
        Advance to next phase

        Args:
            curriculum_phases: All curriculum phases

        Returns:
            Whether advancement was successful
        """
        if self.current_phase < len(curriculum_phases) - 1:
            self.current_phase += 1
            print(f"📈 Advanced to curriculum phase {self.current_phase + 1}")
            return True
        else:
            print(f"✅ Curriculum complete! Using all data.")
            return False


def create_difficulty_balanced_batches(
    dataset: List[Dict],
    batch_size: int,
    difficulty_metric: str = 'num_steps'
) -> List[List[Dict]]:
    """
    Create batches balanced by difficulty

    Each batch contains mix of easy/medium/hard examples

    Args:
        dataset: Training examples
        batch_size: Batch size
        difficulty_metric: How to measure difficulty

    Returns:
        List of balanced batches
    """
    curriculum = CurriculumLearning(difficulty_metric=difficulty_metric)

    # Estimate difficulties
    examples_with_diff = [
        (ex, curriculum.estimate_difficulty(ex))
        for ex in dataset
    ]

    # Sort by difficulty
    examples_with_diff.sort(key=lambda x: x[1])

    # Create balanced batches
    batches = []
    num_batches = len(dataset) // batch_size

    for batch_idx in range(num_batches):
        batch = []

        # Sample from different difficulty ranges
        for i in range(batch_size):
            # Interleave difficulties
            source_idx = (batch_idx + i * num_batches) % len(examples_with_diff)
            batch.append(examples_with_diff[source_idx][0])

        batches.append(batch)

    return batches


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CURRICULUM LEARNING")
    print("=" * 70)

    # Create dummy dataset
    dummy_data = []
    for i in range(100):
        num_steps = np.random.randint(1, 10)
        target = "Step 1: ...\n" * num_steps + "Answer: 42"

        dummy_data.append({
            'question': f'Question {i}',
            'target': target,
            'answer': '42'
        })

    # Create curriculum
    curriculum = CurriculumLearning(
        difficulty_metric='num_steps',
        num_phases=3
    )

    # Generate phases
    phases = curriculum.create_curriculum(dummy_data, shuffle_within_phase=True)

    # Print summary
    curriculum.print_curriculum_summary(phases)

    # Adaptive curriculum example
    print("🔄 Adaptive Curriculum Example:")
    adaptive = AdaptiveCurriculum(
        base_curriculum=curriculum,
        performance_threshold=0.75,
        patience=2
    )

    # Simulate training
    for epoch in range(10):
        # Simulate performance
        perf = 0.6 + epoch * 0.05

        print(f"Epoch {epoch + 1}: Performance = {perf:.2f}, Phase = {adaptive.current_phase + 1}")

        if adaptive.should_advance(perf):
            adaptive.advance_phase(phases)

    print("\n" + "=" * 70)
