"""
Gemma model loading and initialization utilities
"""

from typing import Optional, Tuple

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
except ImportError as e:
    print(f"Warning: Some dependencies not installed: {e}")
    print("Please install: pip install transformers jax flax")


def load_gemma_model(
    model_name: str = "google/gemma-3-1b",
    use_8bit: bool = False,
    use_flash_attention: bool = True,
    device: str = "tpu"
) -> Tuple:
    """
    Load Gemma model and tokenizer

    Args:
        model_name: HuggingFace model identifier
        use_8bit: Load in 8-bit quantization for memory efficiency
        use_flash_attention: Use flash attention if available
        device: Device to load model on ('tpu', 'gpu', 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"📥 Loading Gemma model: {model_name}")
    print(f"   Device: {device}")
    print(f"   8-bit: {use_8bit}")
    print(f"   Flash Attention: {use_flash_attention}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Tokenizer loaded")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Max length: {tokenizer.model_max_length}")

    # Load model
    # Note: For Tunix, we'll need to convert to Flax/JAX format
    # This is a placeholder - actual Tunix integration will differ

    try:
        if use_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )

        print(f"✅ Model loaded")

        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {num_params / 1e9:.2f}B")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Note: For actual Tunix training, model will be loaded via Tunix API")
        raise


def get_model_config(model_name: str = "google/gemma-3-1b") -> dict:
    """
    Get model configuration

    Args:
        model_name: Model name

    Returns:
        Model configuration dictionary
    """
    configs = {
        "google/gemma-3-1b": {
            "hidden_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "intermediate_size": 16384,
            "vocab_size": 256000,
            "max_position_embeddings": 32768,
            "num_key_value_heads": 1,
        },
        "google/gemma-2-2b": {
            "hidden_size": 2304,
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "intermediate_size": 9216,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "num_key_value_heads": 4,
        }
    }

    if model_name in configs:
        return configs[model_name]
    else:
        print(f"⚠️ Config for {model_name} not found, using defaults")
        return configs["google/gemma-3-1b"]


def prepare_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> dict:
    """
    Prepare LoRA configuration for parameter-efficient fine-tuning

    Args:
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: Modules to apply LoRA to

    Returns:
        LoRA configuration dictionary
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }


def count_trainable_parameters(model) -> Tuple[int, int]:
    """
    Count trainable and total parameters in model

    Args:
        model: Model to count parameters

    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return trainable_params, total_params


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("GEMMA MODEL LOADER DEMO")
    print("=" * 70)

    # Get model config
    config = get_model_config("google/gemma-3-1b")
    print("\nGemma 3 1B Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # LoRA config
    print("\nLoRA Configuration:")
    lora_config = prepare_lora_config()
    for key, value in lora_config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Note: Actual model loading requires GPU/TPU environment")
    print("=" * 70)
