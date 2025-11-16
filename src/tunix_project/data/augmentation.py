"""
Data Augmentation for Math Reasoning Tasks

Expand training data by:
1. Number variation - Change specific numbers while maintaining structure
2. Question paraphrasing - Rephrase question different ways
3. Step reordering - Present same logic in different order
4. Operation variation - Use equivalent mathematical expressions

This can effectively 3-5x your training data!
"""

import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class MathDataAugmenter:
    """
    Data augmentation specifically for math word problems
    """

    def __init__(self, seed: int = 42):
        """
        Initialize augmenter

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

    def augment_example(
        self,
        example: Dict,
        num_variations: int = 2,
        methods: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Create augmented versions of an example

        Args:
            example: Original example with 'question', 'answer', 'target'
            num_variations: Number of variations to create
            methods: List of augmentation methods to use
                - 'number_variation'
                - 'question_paraphrase'
                - 'operation_variation'
                - 'context_variation'

        Returns:
            List of augmented examples (including original)
        """
        if methods is None:
            methods = ['number_variation', 'context_variation']

        augmented = [example]  # Include original

        for _ in range(num_variations):
            # Randomly select method
            method = random.choice(methods)

            if method == 'number_variation':
                aug = self.number_variation(example)
            elif method == 'question_paraphrase':
                aug = self.question_paraphrase(example)
            elif method == 'operation_variation':
                aug = self.operation_variation(example)
            elif method == 'context_variation':
                aug = self.context_variation(example)
            else:
                aug = example

            if aug is not None:
                augmented.append(aug)

        return augmented

    def number_variation(self, example: Dict) -> Optional[Dict]:
        """
        Vary numbers while maintaining problem structure

        Example:
        "Janet has 16 eggs..." → "Janet has 20 eggs..."

        Args:
            example: Original example

        Returns:
            Augmented example with different numbers
        """
        question = example['question']
        answer_text = example.get('full_answer_text', example.get('answer', ''))

        # Extract all numbers
        numbers = re.findall(r'\d+', question)

        if len(numbers) < 2:
            # Not enough numbers to vary
            return None

        # Create variation scaling factor (0.8 to 1.2)
        scale = random.uniform(0.8, 1.2)

        # Track number mappings
        number_map = {}

        # Replace numbers
        new_question = question
        for num_str in set(numbers):
            num = int(num_str)
            # Scale and round
            new_num = max(1, int(num * scale))

            number_map[num] = new_num
            new_question = new_question.replace(str(num), str(new_num), 1)

        # Recompute answer (this is problem-specific, might need manual adjustment)
        # For now, return None as we'd need to re-solve the problem
        # In production, you'd use an LLM to regenerate the solution

        # Placeholder: mark that answer needs recomputation
        return {
            'question': new_question,
            'answer': 'RECOMPUTE',  # Needs to be solved again
            'target': 'RECOMPUTE',
            'augmentation_method': 'number_variation',
            'original': example
        }

    def question_paraphrase(self, example: Dict) -> Optional[Dict]:
        """
        Paraphrase question in different words

        This requires an LLM - return placeholder for now

        Args:
            example: Original example

        Returns:
            Paraphrased example
        """
        # Placeholder - in production, use LLM to paraphrase
        # Example:
        # "How many eggs does Janet sell?" →
        # "What is the number of eggs Janet sells?"

        paraphrase_templates = [
            "What is the {entity}?",
            "How much is the {entity}?",
            "Calculate the {entity}.",
            "Find the {entity}."
        ]

        # This is a simple placeholder
        return None  # Implement with LLM

    def operation_variation(self, example: Dict) -> Optional[Dict]:
        """
        Vary how operations are expressed

        Example:
        "3 + 4" → "the sum of 3 and 4"
        "10 - 5" → "10 minus 5"

        Args:
            example: Original example

        Returns:
            Example with varied operation expressions
        """
        target = example.get('target', '')

        # Operation variations
        variations = {
            r'(\d+)\s*\+\s*(\d+)': [
                r'\1 plus \2',
                r'the sum of \1 and \2',
                r'\1 added to \2'
            ],
            r'(\d+)\s*-\s*(\d+)': [
                r'\1 minus \2',
                r'the difference between \1 and \2',
                r'\2 subtracted from \1'
            ],
            r'(\d+)\s*\*\s*(\d+)': [
                r'\1 times \2',
                r'the product of \1 and \2',
                r'\1 multiplied by \2'
            ],
        }

        new_target = target
        modified = False

        for pattern, replacements in variations.items():
            matches = re.finditer(pattern, target)
            for match in matches:
                if random.random() < 0.5:  # 50% chance to vary
                    replacement = random.choice(replacements)
                    new_target = re.sub(
                        pattern,
                        replacement,
                        new_target,
                        count=1
                    )
                    modified = True

        if not modified:
            return None

        return {
            **example,
            'target': new_target,
            'augmentation_method': 'operation_variation'
        }

    def context_variation(self, example: Dict) -> Optional[Dict]:
        """
        Change context/story while keeping math the same

        Example:
        "Janet's eggs" → "Sarah's apples"
        "sells at market" → "gives to friends"

        Args:
            example: Original example

        Returns:
            Example with different context
        """
        question = example['question']

        # Common substitutions
        substitutions = {
            # Names
            r'\bJanet\b': ['Sarah', 'Maria', 'Emma', 'Lisa'],
            r'\bJohn\b': ['Mike', 'David', 'Tom', 'Alex'],

            # Objects
            r'\beggs?\b': ['apples', 'oranges', 'cookies', 'candies'],
            r'\bducks?\b': ['chickens', 'hens', 'geese'],
            r'\bmuffins?\b': ['cakes', 'pies', 'cookies', 'donuts'],

            # Locations
            r'\bmarket\b': ['store', 'shop', 'bazaar', 'stand'],
            r'\bfarm\b': ['ranch', 'garden', 'orchard'],

            # Actions
            r'\bsells?\b': ['trades', 'gives', 'donates', 'distributes'],
            r'\bbakes?\b': ['makes', 'prepares', 'cooks'],
        }

        new_question = question
        new_target = example.get('target', '')
        modified = False

        for pattern, replacements in substitutions.items():
            if re.search(pattern, question, re.IGNORECASE):
                replacement = random.choice(replacements)

                # Replace in question
                new_question = re.sub(
                    pattern,
                    replacement,
                    new_question,
                    flags=re.IGNORECASE
                )

                # Replace in target/reasoning too
                new_target = re.sub(
                    pattern,
                    replacement,
                    new_target,
                    flags=re.IGNORECASE
                )

                modified = True

        if not modified:
            return None

        return {
            **example,
            'question': new_question,
            'target': new_target,
            'augmentation_method': 'context_variation'
        }

    def augment_dataset(
        self,
        dataset: List[Dict],
        augmentation_factor: int = 2,
        methods: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Augment entire dataset

        Args:
            dataset: Original dataset
            augmentation_factor: How many variations per example (total = factor * original_size)
            methods: Augmentation methods to use

        Returns:
            Augmented dataset
        """
        augmented_dataset = []

        print(f"🔄 Augmenting dataset...")
        print(f"   Original size: {len(dataset)}")
        print(f"   Target size: {len(dataset) * (augmentation_factor + 1)}")
        print(f"   Methods: {methods or 'default'}\n")

        for i, example in enumerate(dataset):
            if (i + 1) % 100 == 0:
                print(f"   Processed: {i + 1}/{len(dataset)}")

            # Get augmented examples
            augmented = self.augment_example(
                example,
                num_variations=augmentation_factor,
                methods=methods
            )

            augmented_dataset.extend(augmented)

        print(f"\n✅ Augmentation complete!")
        print(f"   Final size: {len(augmented_dataset)}")
        print(f"   Augmentation ratio: {len(augmented_dataset) / len(dataset):.1f}x\n")

        return augmented_dataset


class BackTranslation:
    """
    Back-translation data augmentation

    Translate question to another language and back to get paraphrased version
    Requires translation API (Google Translate, etc.)
    """

    def __init__(self, intermediate_language: str = 'es'):
        """
        Args:
            intermediate_language: Language code for intermediate translation
        """
        self.intermediate_language = intermediate_language

    def augment(self, text: str) -> str:
        """
        Back-translate text

        Args:
            text: Original text

        Returns:
            Back-translated text
        """
        # Placeholder - implement with translation API
        # 1. Translate text → intermediate_language
        # 2. Translate back → English
        # This gives a paraphrased version

        return text  # Placeholder


class SyntheticExampleGenerator:
    """
    Generate completely synthetic examples similar to existing ones

    Uses templates and rules to create new problems
    """

    def __init__(self):
        """Initialize generator"""
        self.templates = self._load_templates()

    def _load_templates(self) -> List[Dict]:
        """
        Load problem templates

        Returns:
            List of problem templates
        """
        templates = [
            {
                'template': "{name} has {num1} {item}. {pronoun} {action1} {num2} and {action2} {num3}. How many {item} are left?",
                'solution_template': "Step 1: Start with {num1} {item}\nStep 2: {action1_past} {num2}: {num1} - {num2} = {intermediate}\nStep 3: {action2_past} {num3}: {intermediate} - {num3} = {answer}\n\nAnswer: {answer}",
                'variables': {
                    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                    'item': ['apples', 'cookies', 'marbles', 'pencils'],
                    'action1': ['eats', 'gives away', 'loses', 'uses'],
                    'action2': ['eats', 'gives away', 'loses', 'uses'],
                    'pronoun': ['She', 'He', 'They']
                }
            },
            # Add more templates...
        ]

        return templates

    def generate(self, num_examples: int = 10) -> List[Dict]:
        """
        Generate synthetic examples

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of synthetic examples
        """
        examples = []

        for _ in range(num_examples):
            # Select random template
            template = random.choice(self.templates)

            # Fill in variables
            filled = self._fill_template(template)

            examples.append(filled)

        return examples

    def _fill_template(self, template: Dict) -> Dict:
        """Fill template with random values"""
        # Select random values for variables
        values = {}
        for var, options in template['variables'].items():
            values[var] = random.choice(options)

        # Generate random numbers
        values['num1'] = random.randint(10, 50)
        values['num2'] = random.randint(1, values['num1'] // 2)
        values['num3'] = random.randint(1, (values['num1'] - values['num2']) // 2)

        # Compute answer
        values['intermediate'] = values['num1'] - values['num2']
        values['answer'] = values['intermediate'] - values['num3']

        # Fill templates
        question = template['template'].format(**values)
        solution = template['solution_template'].format(**values)

        return {
            'question': question,
            'target': solution,
            'answer': str(values['answer']),
            'augmentation_method': 'synthetic_generation'
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DATA AUGMENTATION")
    print("=" * 70)

    # Test example
    example = {
        'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. How much does she make at the farmers' market if she sells the remainder for $2 per egg?",
        'answer': '18',
        'target': "Step 1: Calculate eggs used\nStep 2: 3 + 4 = 7\nStep 3: 16 - 7 = 9\nAnswer: 18"
    }

    # Create augmenter
    augmenter = MathDataAugmenter(seed=42)

    # Test context variation
    print("\n🔄 Context Variation Test:")
    print("-" * 70)
    aug = augmenter.context_variation(example)
    if aug:
        print(f"Original: {example['question'][:80]}...")
        print(f"Augmented: {aug['question'][:80]}...")

    # Test operation variation
    print("\n🔄 Operation Variation Test:")
    print("-" * 70)
    aug = augmenter.operation_variation(example)
    if aug:
        print(f"Original: {example['target']}")
        print(f"Augmented: {aug['target']}")

    print("\n" + "=" * 70)
    print("✅ Data augmentation can significantly expand training data!")
    print("=" * 70)
