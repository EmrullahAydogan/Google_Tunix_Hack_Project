"""
Self-Consistency for Chain-of-Thought Reasoning

Self-consistency samples multiple reasoning paths for the same question,
then selects the most consistent answer via majority voting.

This technique can boost accuracy by 5-10% without any additional training!

Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
Wang et al., 2022 (https://arxiv.org/abs/2203.11171)
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np


class SelfConsistency:
    """
    Self-consistency inference for improved reasoning accuracy

    Key idea: Sample multiple reasoning paths, take majority vote on answers
    """

    def __init__(
        self,
        num_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        diversity_penalty: float = 0.0
    ):
        """
        Initialize self-consistency

        Args:
            num_samples: Number of reasoning paths to generate
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling parameter
            diversity_penalty: Penalty for similar reasoning paths
        """
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.diversity_penalty = diversity_penalty

    def generate_multiple_paths(
        self,
        model,
        tokenizer,
        question: str,
        max_new_tokens: int = 512
    ) -> List[str]:
        """
        Generate multiple reasoning paths for a question

        Args:
            model: Language model
            tokenizer: Tokenizer
            question: Input question
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated responses
        """
        responses = []

        # Format question
        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"

        for i in range(self.num_samples):
            # Generate with sampling
            # Note: Actual implementation depends on your model API
            # This is a placeholder structure

            response = self._generate_single(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )

            responses.append(response)

        return responses

    def _generate_single(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> str:
        """
        Generate a single response

        Note: This is a placeholder. Replace with actual model inference.
        """
        # Placeholder for actual model inference
        # In real implementation:
        # inputs = tokenizer(prompt, return_tensors="pt")
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     do_sample=do_sample
        # )
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return "PLACEHOLDER - implement model.generate()"

    def extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        # Look for "Answer: X" pattern
        match = re.search(r'Answer:\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Extract number
            number_match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer)
            if number_match:
                return number_match.group(0).replace(',', '')

        # Fallback: last number
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
        return numbers[-1].replace(',', '') if numbers else ""

    def majority_vote(self, answers: List[str]) -> Tuple[str, float]:
        """
        Select answer by majority voting

        Args:
            answers: List of candidate answers

        Returns:
            (most_common_answer, confidence_score)
        """
        if not answers:
            return "", 0.0

        # Count occurrences
        answer_counts = Counter(answers)

        # Most common answer
        most_common_answer, count = answer_counts.most_common(1)[0]

        # Confidence = fraction of votes for winner
        confidence = count / len(answers)

        return most_common_answer, confidence

    def select_best_reasoning(
        self,
        responses: List[str],
        answers: List[str],
        target_answer: str,
        score_function: Optional[Callable] = None
    ) -> str:
        """
        Select the best reasoning path among those with correct answer

        Args:
            responses: List of full responses
            answers: List of extracted answers
            target_answer: The majority-voted answer
            score_function: Optional function to score reasoning quality

        Returns:
            Best reasoning path
        """
        # Filter responses with the target answer
        correct_responses = [
            resp for resp, ans in zip(responses, answers)
            if ans == target_answer
        ]

        if not correct_responses:
            # No correct answers, return most common response
            return responses[0] if responses else ""

        # If no score function, return first correct one
        if score_function is None:
            return correct_responses[0]

        # Score each correct response
        scored = [(resp, score_function(resp)) for resp in correct_responses]

        # Return highest scoring
        best_response = max(scored, key=lambda x: x[1])[0]

        return best_response

    def aggregate_with_weighted_voting(
        self,
        answers: List[str],
        scores: List[float]
    ) -> Tuple[str, float]:
        """
        Weighted majority voting based on reasoning quality scores

        Args:
            answers: List of candidate answers
            scores: Quality scores for each answer

        Returns:
            (best_answer, confidence)
        """
        if not answers or not scores:
            return "", 0.0

        # Weight votes by scores
        answer_weights = {}
        for answer, score in zip(answers, scores):
            if answer not in answer_weights:
                answer_weights[answer] = 0.0
            answer_weights[answer] += score

        # Best answer
        best_answer = max(answer_weights.items(), key=lambda x: x[1])[0]

        # Confidence
        total_weight = sum(answer_weights.values())
        confidence = answer_weights[best_answer] / total_weight if total_weight > 0 else 0.0

        return best_answer, confidence

    def __call__(
        self,
        model,
        tokenizer,
        question: str,
        ground_truth: Optional[str] = None,
        return_all_paths: bool = False,
        score_function: Optional[Callable] = None
    ) -> Dict:
        """
        Perform self-consistency inference

        Args:
            model: Language model
            tokenizer: Tokenizer
            question: Input question
            ground_truth: Ground truth answer (for evaluation)
            return_all_paths: Whether to return all reasoning paths
            score_function: Optional function to score reasoning quality

        Returns:
            Dictionary with results
        """
        # Generate multiple paths
        responses = self.generate_multiple_paths(model, tokenizer, question)

        # Extract answers
        answers = [self.extract_answer(resp) for resp in responses]

        # Majority vote
        if score_function is None:
            # Simple majority voting
            final_answer, confidence = self.majority_vote(answers)
        else:
            # Weighted voting by reasoning quality
            scores = [score_function(resp) for resp in responses]
            final_answer, confidence = self.aggregate_with_weighted_voting(answers, scores)

        # Select best reasoning
        best_reasoning = self.select_best_reasoning(
            responses,
            answers,
            final_answer,
            score_function
        )

        # Prepare result
        result = {
            'question': question,
            'final_answer': final_answer,
            'confidence': confidence,
            'best_reasoning': best_reasoning,
            'num_samples': len(responses),
            'answer_distribution': dict(Counter(answers))
        }

        if ground_truth is not None:
            result['is_correct'] = (final_answer == ground_truth)
            result['ground_truth'] = ground_truth

        if return_all_paths:
            result['all_responses'] = responses
            result['all_answers'] = answers

        return result


# Utility functions for analysis

def analyze_consistency(results: List[Dict]) -> Dict:
    """
    Analyze self-consistency results

    Args:
        results: List of self-consistency results

    Returns:
        Analysis dictionary
    """
    total = len(results)

    # Aggregate statistics
    avg_confidence = np.mean([r['confidence'] for r in results])

    # Accuracy (if ground truth available)
    correct_results = [r for r in results if 'is_correct' in r]
    accuracy = np.mean([r['is_correct'] for r in correct_results]) if correct_results else None

    # Confidence vs correctness correlation
    if correct_results:
        confidences = [r['confidence'] for r in correct_results]
        correctness = [float(r['is_correct']) for r in correct_results]
        correlation = np.corrcoef(confidences, correctness)[0, 1]
    else:
        correlation = None

    # Distribution of answer diversity
    diversities = []
    for r in results:
        dist = r['answer_distribution']
        # Diversity = number of unique answers / total samples
        diversity = len(dist) / r['num_samples']
        diversities.append(diversity)

    avg_diversity = np.mean(diversities)

    return {
        'total_samples': total,
        'avg_confidence': avg_confidence,
        'accuracy': accuracy,
        'confidence_correctness_correlation': correlation,
        'avg_answer_diversity': avg_diversity
    }


def calibration_analysis(results: List[Dict], bins: int = 10) -> Dict:
    """
    Analyze calibration: does confidence match accuracy?

    Args:
        results: List of results with confidence and correctness
        bins: Number of confidence bins

    Returns:
        Calibration data
    """
    results = [r for r in results if 'is_correct' in r]

    if not results:
        return {}

    # Create bins
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(bins):
        # Results in this bin
        bin_results = [
            r for r in results
            if bin_edges[i] <= r['confidence'] < bin_edges[i + 1]
        ]

        if bin_results:
            avg_conf = np.mean([r['confidence'] for r in bin_results])
            avg_acc = np.mean([r['is_correct'] for r in bin_results])
            count = len(bin_results)

            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
            bin_counts.append(count)

    # Expected calibration error (ECE)
    ece = 0.0
    total = len(results)
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        ece += (count / total) * abs(conf - acc)

    return {
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'expected_calibration_error': ece
    }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SELF-CONSISTENCY IMPLEMENTATION")
    print("=" * 70)

    # Initialize
    sc = SelfConsistency(
        num_samples=10,
        temperature=0.7,
        top_p=0.9
    )

    print(f"\n✅ Self-Consistency initialized:")
    print(f"   Samples per question: {sc.num_samples}")
    print(f"   Temperature: {sc.temperature}")
    print(f"   Top-p: {sc.top_p}")

    # Test majority voting
    print(f"\n🧪 Test majority voting:")
    test_answers = ["42", "42", "42", "43", "42", "41", "42", "42"]
    final_answer, confidence = sc.majority_vote(test_answers)
    print(f"   Answers: {test_answers}")
    print(f"   Final answer: {final_answer}")
    print(f"   Confidence: {confidence:.2%}")

    print("\n" + "=" * 70)
    print("⚠️ Note: Actual model inference needs to be implemented")
    print("This module provides the self-consistency logic")
    print("=" * 70)
