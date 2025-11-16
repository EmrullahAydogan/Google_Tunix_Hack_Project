# Google Tunix Hack - Train a Model to Show Its Work

![Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)
![Prize](https://img.shields.io/badge/Prize-$100K-gold)
![Deadline](https://img.shields.io/badge/Deadline-Jan%2012%202026-red)

Fine-tuning Gemma models with Tunix to teach them step-by-step reasoning.

## 🎯 Project Overview

This project participates in the [Google Tunix Hack](https://www.kaggle.com/competitions/google-tunix-hackathon) competition, where we train language models to not just provide answers, but to **show their reasoning process** step-by-step.

### Goal
Train a Gemma model using Tunix to:
- Solve complex problems (mathematics, logic, reasoning)
- Explain its thought process clearly
- Show step-by-step reasoning (Chain-of-Thought)
- Be transparent and interpretable

## 🏗️ Project Structure

```
Google_Tunix_Hack_Project/
├── notebooks/                    # Kaggle notebooks
│   └── main_training_notebook.ipynb  # Main submission notebook
│
├── src/tunix_project/           # Source code
│   ├── data/                    # Data loading & preprocessing
│   ├── models/                  # Model definitions
│   ├── training/                # Training loops & trainers
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Utility functions
│
├── configs/                     # Configuration files
│   ├── training/                # Training configs
│   └── model/                   # Model configs
│
├── data/                        # Dataset storage
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
│
├── models/                      # Trained models
│   ├── checkpoints/             # Training checkpoints
│   └── final/                   # Final models
│
├── scripts/                     # Utility scripts
│   ├── download_data.py         # Download datasets
│   ├── preprocess.py            # Preprocess data
│   └── evaluate.py              # Evaluate models
│
├── docs/                        # Documentation
│   ├── writeup/                 # Kaggle writeup drafts
│   └── video/                   # Video script & materials
│
├── assets/                      # Media assets
│   ├── images/                  # Cover image, charts
│   └── videos/                  # Demo videos
│
├── tests/                       # Unit tests
│
├── RESEARCH_NOTES.md            # Research findings
├── SUBMISSION_REQUIREMENTS.md   # Submission checklist
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- TPU access (Kaggle or Google Cloud)
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/EmrullahAydogan/Google_Tunix_Hack_Project.git
cd Google_Tunix_Hack_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
python scripts/download_data.py --dataset gsm8k
```

### Run Training

**Option 1: Kaggle Notebook (Recommended)**
1. Upload `notebooks/main_training_notebook.ipynb` to Kaggle
2. Enable TPU accelerator
3. Run all cells

**Option 2: Local/Cloud**
```bash
python scripts/train.py --config configs/training/grpo_gemma3_1b.yaml
```

## 📊 Approach

### Model Selection
- **Gemma 3 1B** - Chosen for efficiency and 32K context window

### Training Method
- **Tunix GRPO** - Group Relative Policy Optimization
- **Dataset:** GSM8K (Grade School Math 8K problems)
- **Focus:** Step-by-step mathematical reasoning

### Key Features
1. Chain-of-Thought prompting
2. Multi-criteria reward function (8 components)
3. Self-consistency evaluation (+5-10% accuracy boost!)
4. Curriculum learning (progressive difficulty)
5. Data augmentation (3-5x training data)
6. Process reward modeling (step-level learning)
7. Ensemble methods (+2-5% accuracy boost!)

## 🚀 Advanced Techniques (NEW!)

This project implements **state-of-the-art techniques** to maximize competition performance:

### 1. 🎯 Self-Consistency (+5-10% Accuracy!)
**Most impactful technique - implement this first!**

```python
from tunix_project.training.self_consistency import SelfConsistency

# Generate 10 different reasoning paths
sc = SelfConsistency(num_samples=10, temperature=0.7)
result = sc(model, tokenizer, question="What is 2+2?")

# Majority vote → more robust answers!
print(result['final_answer'])  # Most common answer
print(result['confidence'])    # How confident?
```

**Why it works:** Multiple attempts catch errors, majority voting filters mistakes.

### 2. 🧠 Advanced Reward Function
**8 criteria vs. basic 3 - much richer training signal!**

```python
from tunix_project.training.advanced_reward import compute_advanced_reward

reward = compute_advanced_reward(response, ground_truth, question)

# Rewards include:
# - Correctness (30%)
# - Reasoning quality (15%)
# - Clarity (10%)
# - Step coherence (15%) ⭐ NEW
# - Mathematical rigor (15%) ⭐ NEW
# - Explanation quality (5%) ⭐ NEW
# - Partial correctness (5%) ⭐ NEW
# - Efficiency (5%) ⭐ NEW
```

### 3. 📚 Curriculum Learning
**Train smarter: easy → medium → hard**

```python
from tunix_project.data.curriculum import CurriculumLearning

curriculum = CurriculumLearning(difficulty_metric='num_steps', num_phases=3)
phases = curriculum.create_curriculum(dataset)

# Phase 1: Easy problems (1-3 steps)
# Phase 2: Medium problems (3-5 steps)
# Phase 3: Hard problems (5+ steps)

# Or use adaptive curriculum (adjusts based on performance)
from tunix_project.data.curriculum import AdaptiveCurriculum
adaptive = AdaptiveCurriculum(curriculum, performance_threshold=0.75)
```

### 4. 🔄 Data Augmentation (3-5x More Data!)
**Expand your training set without collecting more data**

```python
from tunix_project.data.augmentation import MathDataAugmenter

augmenter = MathDataAugmenter(seed=42)

# Augment dataset 3x
augmented = augmenter.augment_dataset(
    original_dataset,
    augmentation_factor=2,  # 2 variations per example
    methods=['number_variation', 'context_variation']
)

# Original: 7,500 examples → Augmented: 22,500 examples!
```

### 5. ⚡ Process Reward Modeling
**Reward EACH STEP, not just final answer**

```python
from tunix_project.training.process_reward import ProcessRewardModel

prm = ProcessRewardModel()
result = prm.compute_process_rewards(response, question, ground_truth)

# Get rewards for each reasoning step
for i, step_reward in enumerate(result['process_rewards']):
    print(f"Step {i+1}: {step_reward['step_reward']:.3f}")
```

**Why it works:** Model learns which intermediate steps are good, not just whether final answer is correct.

### 6. 🤝 Ensemble Methods (+2-5% Accuracy!)
**Combine multiple models for better performance**

```python
from tunix_project.training.ensemble import EnsemblePredictor

# Train 3 different models (or same model, different seeds)
models = [model1, model2, model3]

ensemble = EnsemblePredictor(
    models=models,
    voting_strategy='weighted'  # or 'majority', 'confidence'
)

result = ensemble.predict(question)
print(result['ensemble_answer'])
```

## 📈 Expected Performance Impact

| Technique | Expected Improvement | Difficulty | Priority |
|-----------|---------------------|------------|----------|
| **Self-Consistency** | **+5-10%** | Easy | ⭐⭐⭐ Must have! |
| Advanced Reward | Better training | Medium | ⭐⭐⭐ High |
| Curriculum Learning | Faster convergence | Easy | ⭐⭐ Medium |
| Data Augmentation | Reduces overfitting | Medium | ⭐⭐ Medium |
| Process Rewards | Finer learning | Hard | ⭐ Nice to have |
| **Ensemble** | **+2-5%** | Easy | ⭐⭐⭐ Must have! |

**Combined Expected Improvement: +10-20% over baseline!**

From ~75% → **85-95% accuracy range** → **Top 6 contention!**

## 📈 Results

*(To be updated after training)*

- Accuracy on GSM8K test set: TBD
- Reasoning quality score: TBD
- Step-by-step clarity: TBD

## 📝 Submission Components

- ✅ **Kaggle Writeup** - [Link TBD]
- ✅ **Public Notebook** - [Link TBD]
- ✅ **YouTube Video** - [Link TBD]
- ✅ **Reproducibility** - All configs included

## 🛠️ Technologies Used

- **Tunix** - JAX-native LLM post-training library
- **JAX/Flax** - High-performance numerical computing
- **Gemma 3 1B** - Google's open-weight language model
- **Hugging Face** - Model hub and datasets
- **Kaggle TPU** - Training infrastructure

## 📚 Documentation

- [Research Notes](RESEARCH_NOTES.md) - Literature review and findings
- [Submission Requirements](SUBMISSION_REQUIREMENTS.md) - Competition checklist
- [Writeup Draft](docs/writeup/) - Kaggle writeup preparation
- [Video Script](docs/video/) - YouTube video preparation

## 🤝 Contributing

This is a competition project, but feedback and suggestions are welcome!

## 📄 License

This project is for educational and competition purposes.

## 🙏 Acknowledgments

- Google for organizing the Tunix Hack
- Kaggle for hosting the competition
- Tunix team for the amazing library
- GSM8K dataset creators

## 📧 Contact

- **Author:** Emrullah Aydogan
- **GitHub:** [@EmrullahAydogan](https://github.com/EmrullahAydogan)
- **Competition:** [Google Tunix Hack](https://www.kaggle.com/competitions/google-tunix-hackathon)

---

**Status:** 🚧 In Development
**Last Updated:** November 16, 2025
**Competition Deadline:** January 12, 2026
