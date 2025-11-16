"""
Chain-of-Thought prompt templates for reasoning tasks

These templates guide the model to show step-by-step reasoning
"""

from typing import Dict, List, Optional


class PromptTemplate:
    """Base class for prompt templates"""

    def __init__(self, template: str, name: str = "default"):
        self.template = template
        self.name = name

    def format(self, **kwargs) -> str:
        """Format the template with given arguments"""
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}')"


# ============================================================================
# CHAIN-OF-THOUGHT PROMPTS FOR GSM8K
# ============================================================================

BASIC_COT_PROMPT = PromptTemplate(
    template="""Question: {question}

Let's solve this step by step:
""",
    name="basic_cot"
)


DETAILED_COT_PROMPT = PromptTemplate(
    template="""Question: {question}

Let me break this down step by step to find the answer:

Step 1:""",
    name="detailed_cot"
)


INSTRUCTIONAL_COT_PROMPT = PromptTemplate(
    template="""You are a helpful math tutor. Solve the following problem step by step, showing all your work.

Question: {question}

Solution:
Let's think through this carefully:
""",
    name="instructional_cot"
)


STRUCTURED_COT_PROMPT = PromptTemplate(
    template="""Problem: {question}

To solve this, I will:
1. Identify the key information
2. Break down the problem into steps
3. Calculate each step
4. Provide the final answer

Let's begin:
""",
    name="structured_cot"
)


CONVERSATIONAL_COT_PROMPT = PromptTemplate(
    template="""Q: {question}

A: Great question! Let me work through this step by step.

""",
    name="conversational_cot"
)


# ============================================================================
# FEW-SHOT PROMPTS
# ============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": """Step 1: Start with the initial number of cars: 3 cars
Step 2: Add the cars that arrived: 3 + 2 = 5 cars
Step 3: The total is 5 cars""",
        "answer": "5"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "reasoning": """Step 1: Calculate the total cost of bagels: 5 bagels × $3 = $15
Step 2: Subtract from the money she had: $23 - $15 = $8
Step 3: She has $8 left""",
        "answer": "8"
    }
]


def create_few_shot_prompt(question: str, examples: Optional[List[Dict]] = None) -> str:
    """
    Create a few-shot prompt with examples

    Args:
        question: The question to answer
        examples: List of example dictionaries (optional, uses default if None)

    Returns:
        Formatted few-shot prompt
    """
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    prompt = "Here are some examples of how to solve math problems step by step:\n\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n\n"
        prompt += f"{example['reasoning']}\n\n"
        prompt += f"Answer: {example['answer']}\n\n"
        prompt += "-" * 70 + "\n\n"

    prompt += f"Now solve this problem:\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Let's solve this step by step:\n"

    return prompt


# ============================================================================
# ZERO-SHOT COT PROMPTS
# ============================================================================

ZERO_SHOT_COT_PROMPT = PromptTemplate(
    template="""Question: {question}

Let's think step by step.
""",
    name="zero_shot_cot"
)


ZERO_SHOT_COT_DETAILED = PromptTemplate(
    template="""Question: {question}

Let's approach this systematically and think step by step to find the answer.
""",
    name="zero_shot_cot_detailed"
)


# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

def format_response(reasoning_steps: List[str], answer: str) -> str:
    """
    Format model response with reasoning steps and answer

    Args:
        reasoning_steps: List of reasoning steps
        answer: Final answer

    Returns:
        Formatted response string
    """
    response = ""

    for i, step in enumerate(reasoning_steps, 1):
        response += f"Step {i}: {step}\n"

    response += f"\nAnswer: {answer}"

    return response


def parse_response(response: str) -> Dict[str, any]:
    """
    Parse model response to extract reasoning and answer

    Args:
        response: Model's generated response

    Returns:
        Dictionary with 'reasoning_steps' and 'answer'
    """
    import re

    # Extract steps
    step_pattern = r'Step \d+:(.+?)(?=Step \d+:|Answer:|$)'
    steps = re.findall(step_pattern, response, re.DOTALL)
    reasoning_steps = [step.strip() for step in steps]

    # Extract answer
    answer_pattern = r'Answer:\s*(.+?)(?:\n|$)'
    answer_match = re.search(answer_pattern, response)
    answer = answer_match.group(1).strip() if answer_match else ""

    return {
        'reasoning_steps': reasoning_steps,
        'answer': answer,
        'num_steps': len(reasoning_steps)
    }


# ============================================================================
# PROMPT SELECTION
# ============================================================================

PROMPT_REGISTRY = {
    'basic': BASIC_COT_PROMPT,
    'detailed': DETAILED_COT_PROMPT,
    'instructional': INSTRUCTIONAL_COT_PROMPT,
    'structured': STRUCTURED_COT_PROMPT,
    'conversational': CONVERSATIONAL_COT_PROMPT,
    'zero_shot': ZERO_SHOT_COT_PROMPT,
    'zero_shot_detailed': ZERO_SHOT_COT_DETAILED,
}


def get_prompt_template(name: str = 'basic') -> PromptTemplate:
    """
    Get a prompt template by name

    Args:
        name: Name of the template

    Returns:
        PromptTemplate object

    Raises:
        ValueError: If template name is not found
    """
    if name not in PROMPT_REGISTRY:
        available = ', '.join(PROMPT_REGISTRY.keys())
        raise ValueError(
            f"Unknown prompt template: {name}. "
            f"Available templates: {available}"
        )

    return PROMPT_REGISTRY[name]


def list_available_prompts() -> List[str]:
    """
    List all available prompt templates

    Returns:
        List of template names
    """
    return list(PROMPT_REGISTRY.keys())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example question
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print("=" * 70)
    print("PROMPT TEMPLATES DEMO")
    print("=" * 70)

    # Test different prompts
    for name in list_available_prompts():
        template = get_prompt_template(name)
        prompt = template.format(question=question)

        print(f"\n{'='*70}")
        print(f"Template: {name}")
        print(f"{'='*70}")
        print(prompt)

    # Few-shot example
    print(f"\n{'='*70}")
    print("Few-Shot Prompt")
    print(f"{'='*70}")
    few_shot = create_few_shot_prompt(question)
    print(few_shot)

    # Parse response example
    print(f"\n{'='*70}")
    print("Response Parsing Example")
    print(f"{'='*70}")

    sample_response = """Step 1: Janet's ducks lay 16 eggs per day
Step 2: She eats 3 eggs for breakfast
Step 3: She uses 4 eggs for muffins
Step 4: Total eggs used: 3 + 4 = 7 eggs
Step 5: Remaining eggs to sell: 16 - 7 = 9 eggs
Step 6: Revenue from selling eggs: 9 × $2 = $18

Answer: 18"""

    parsed = parse_response(sample_response)
    print(f"Parsed response:")
    print(f"  Number of steps: {parsed['num_steps']}")
    print(f"  Answer: {parsed['answer']}")
    print(f"  Reasoning steps:")
    for i, step in enumerate(parsed['reasoning_steps'], 1):
        print(f"    {i}. {step[:50]}...")
