"""
Configuration management utilities
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ Config saved to: {output_path}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries

    Args:
        base_config: Base configuration
        override_config: Configuration to override with

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_env_or_default(env_var: str, default: Any) -> Any:
    """
    Get environment variable or default value

    Args:
        env_var: Environment variable name
        default: Default value if env var not set

    Returns:
        Environment variable value or default
    """
    return os.getenv(env_var, default)


class Config:
    """Configuration object with dot notation access"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default"""
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config("configs/training/grpo_gemma3_1b.yaml")
    print("Loaded config:")
    print(yaml.dump(config, indent=2))

    # Use Config object
    cfg = Config(config)
    print(f"\nExperiment name: {cfg.experiment_name}")
    print(f"Learning rate: {cfg.training.learning_rate}")
