"""
Logging utilities for training and evaluation
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "tunix_project",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    Setup logger with optional file and console output

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
        use_rich: Use rich formatting for console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Logger for training metrics and progress"""

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None
    ):
        """
        Initialize training logger

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity/username
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        log_file = self.log_dir / f"{experiment_name}.log"
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file)
        )

        # Setup W&B if requested
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "google-tunix-hack",
                    entity=wandb_entity,
                    name=experiment_name,
                    config={}
                )
                self.wandb = wandb
                self.logger.info("✅ Weights & Biases initialized")
            except ImportError:
                self.logger.warning("⚠️ wandb not installed. Skipping W&B logging.")
                self.use_wandb = False

        self.step = 0

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics to console, file, and W&B

        Args:
            metrics: Dictionary of metrics
            step: Training step (optional, auto-increments if None)
        """
        if step is None:
            step = self.step
            self.step += 1

        # Log to console/file
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")

        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def log_example(self, question: str, response: str, reward: float, step: Optional[int] = None):
        """
        Log an example prediction

        Args:
            question: Input question
            response: Model response
            reward: Computed reward
            step: Training step
        """
        if step is None:
            step = self.step

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Example at step {step}")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Question: {question[:100]}...")
        self.logger.info(f"Response: {response[:200]}...")
        self.logger.info(f"Reward: {reward:.3f}")
        self.logger.info(f"{'='*70}\n")

        # Log to W&B as table
        if self.use_wandb:
            self.wandb.log({
                "examples": self.wandb.Table(
                    columns=["step", "question", "response", "reward"],
                    data=[[step, question, response, reward]]
                )
            })

    def finish(self):
        """Finish logging session"""
        if self.use_wandb:
            self.wandb.finish()
        self.logger.info("✅ Training session finished")


# Example usage
if __name__ == "__main__":
    # Basic logger
    logger = setup_logger("test_logger", log_level="INFO")
    logger.info("This is an info message")
    logger.warning("This is a warning")

    # Training logger
    training_logger = TrainingLogger(
        experiment_name="test_experiment",
        use_wandb=False
    )

    # Log metrics
    training_logger.log_metrics({
        'loss': 0.5,
        'reward': 0.8,
        'accuracy': 0.75
    })

    # Log example
    training_logger.log_example(
        question="What is 2+2?",
        response="Step 1: Add 2 and 2\nAnswer: 4",
        reward=0.95
    )

    training_logger.finish()
