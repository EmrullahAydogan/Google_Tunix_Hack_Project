# Project Structure Overview

Bu belge, projenin klasör yapısını ve her dosyanın amacını açıklar.

## 📁 Dizin Yapısı

```
Google_Tunix_Hack_Project/
│
├── 📄 README.md                          # Ana proje dokümantasyonu
├── 📄 RESEARCH_NOTES.md                  # Araştırma bulguları
├── 📄 SUBMISSION_REQUIREMENTS.md         # Submission checklist
├── 📄 PROJECT_STRUCTURE.md               # Bu dosya
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore patterns
│
├── 📂 notebooks/                         # Jupyter Notebooks
│   └── main_training_notebook.ipynb      # Ana Kaggle submission notebook
│
├── 📂 src/tunix_project/                # Python kaynak kodları
│   ├── __init__.py
│   │
│   ├── 📂 data/                         # Veri yükleme ve işleme
│   │   ├── __init__.py
│   │   ├── dataset.py                   # Dataset loading
│   │   ├── preprocessing.py             # Data preprocessing
│   │   └── prompts.py                   # Prompt templates
│   │
│   ├── 📂 models/                       # Model tanımlamaları
│   │   ├── __init__.py
│   │   ├── gemma.py                     # Gemma model loader
│   │   └── config.py                    # Model configurations
│   │
│   ├── 📂 training/                     # Training loops
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Tunix trainer wrapper
│   │   ├── reward.py                    # Reward function
│   │   └── callbacks.py                 # Training callbacks
│   │
│   ├── 📂 evaluation/                   # Değerlendirme
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Evaluation metrics
│   │   └── evaluator.py                 # Model evaluator
│   │
│   └── 📂 utils/                        # Yardımcı fonksiyonlar
│       ├── __init__.py
│       ├── logging.py                   # Logger setup
│       ├── config.py                    # Config loader
│       └── visualization.py             # Plotting utilities
│
├── 📂 configs/                          # Konfigürasyon dosyaları
│   ├── 📂 training/                     # Training configs
│   │   ├── grpo_gemma3_1b.yaml         # GRPO + Gemma 3 1B
│   │   ├── ppo_gemma2_2b.yaml          # PPO + Gemma 2 2B (opsiyonel)
│   │   └── gspo_gemma3_1b.yaml         # GSPO + Gemma 3 1B (opsiyonel)
│   │
│   └── 📂 model/                        # Model configs
│       ├── gemma3_1b.yaml              # Gemma 3 1B config
│       └── gemma2_2b.yaml              # Gemma 2 2B config (opsiyonel)
│
├── 📂 data/                             # Dataset storage
│   ├── 📂 raw/                          # Ham veri
│   │   ├── gsm8k/                       # GSM8K dataset
│   │   └── math/                        # MATH dataset (opsiyonel)
│   │
│   └── 📂 processed/                    # İşlenmiş veri
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── 📂 models/                           # Eğitilmiş modeller
│   ├── 📂 checkpoints/                  # Training checkpoints
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── checkpoint-1500/
│   │
│   └── 📂 final/                        # Final model
│       ├── model.safetensors
│       ├── config.json
│       └── tokenizer/
│
├── 📂 scripts/                          # Utility scripts
│   ├── download_data.py                 # Dataset download
│   ├── preprocess.py                    # Data preprocessing
│   ├── train.py                         # Training script
│   ├── evaluate.py                      # Evaluation script
│   ├── export_model.py                  # Model export
│   └── generate_examples.py             # Generate reasoning examples
│
├── 📂 docs/                             # Dokümantasyon
│   ├── 📂 writeup/                      # Kaggle writeup
│   │   ├── WRITEUP_TEMPLATE.md         # Writeup şablonu
│   │   ├── draft.md                     # Draft writeup
│   │   └── final.md                     # Final writeup
│   │
│   └── 📂 video/                        # Video materyalleri
│       ├── VIDEO_SCRIPT.md             # Video script
│       ├── slides.pdf                   # Presentation slides
│       └── recording_notes.md           # Recording checklist
│
├── 📂 assets/                           # Media dosyaları
│   ├── 📂 images/                       # Görseller
│   │   ├── cover_image.png             # Kaggle cover image
│   │   ├── architecture.png             # Architecture diagram
│   │   └── results_chart.png            # Results visualization
│   │
│   └── 📂 videos/                       # Video dosyaları
│       └── demo.mp4                     # Demo video
│
└── 📂 tests/                            # Unit tests (opsiyonel)
    ├── test_data.py
    ├── test_model.py
    ├── test_training.py
    └── test_evaluation.py
```

## 📝 Dosya Açıklamaları

### Ana Dizin

| Dosya | Amaç |
|-------|------|
| `README.md` | Projenin ana dokümantasyonu, kurulum ve kullanım talimatları |
| `RESEARCH_NOTES.md` | Tunix, Gemma ve veri setleri hakkında araştırma bulguları |
| `SUBMISSION_REQUIREMENTS.md` | Yarışma submission gereksinimleri checklist |
| `requirements.txt` | Python bağımlılıkları listesi |
| `.gitignore` | Git'in ignore edeceği dosya pattern'leri |

### Notebooks

**Main Training Notebook:** Kaggle'a submit edilecek ana notebook. Tüm training pipeline buradan çalışacak.

### Source Code (`src/tunix_project/`)

#### `data/`
- **dataset.py:** GSM8K ve diğer dataset'leri yükleme
- **preprocessing.py:** Veri temizleme ve formatlama
- **prompts.py:** Chain-of-thought prompt şablonları

#### `models/`
- **gemma.py:** Gemma model'ini yükleme ve initialize etme
- **config.py:** Model configuration yönetimi

#### `training/`
- **trainer.py:** Tunix trainer wrapper, GRPO/PPO/GSPO
- **reward.py:** Reward function implementation (kritik!)
- **callbacks.py:** Training sırasında kullanılacak callbacks

#### `evaluation/`
- **metrics.py:** Accuracy, reasoning quality, clarity metrikleri
- **evaluator.py:** Model evaluation pipeline

#### `utils/`
- **logging.py:** Logger setup (wandb, tensorboard)
- **config.py:** YAML config dosyalarını yükleme
- **visualization.py:** Sonuçları görselleştirme

### Configs

**Training configs:** Her training stratejisi için ayrı YAML dosyası
**Model configs:** Her model için hyperparameters ve settings

### Data

- **raw/:** İndirilen ham veri
- **processed/:** İşlenmiş ve tokenize edilmiş veri

### Models

- **checkpoints/:** Training sırasında kaydedilen checkpoint'ler
- **final/:** Final submission için kullanılacak model

### Scripts

Standalone Python scriptleri:
- Data download ve preprocessing
- Training başlatma
- Model evaluation
- Model export

### Docs

- **writeup/:** Kaggle writeup (max 1,500 kelime)
- **video/:** YouTube video script ve materyaller

### Assets

- **images/:** Cover image, charts, diagrams
- **videos/:** Demo ve presentation videoları

## 🚀 Workflow

### 1. Data Preparation
```bash
python scripts/download_data.py --dataset gsm8k
python scripts/preprocess.py --input data/raw --output data/processed
```

### 2. Training
```bash
# Option A: Local/Cloud
python scripts/train.py --config configs/training/grpo_gemma3_1b.yaml

# Option B: Kaggle Notebook
# Upload and run notebooks/main_training_notebook.ipynb
```

### 3. Evaluation
```bash
python scripts/evaluate.py --model models/final --data data/processed/test.json
```

### 4. Export & Demo
```bash
python scripts/generate_examples.py --model models/final --num_examples 10
```

### 5. Submission
1. Finalize `docs/writeup/final.md`
2. Record video using `docs/video/VIDEO_SCRIPT.md`
3. Upload notebook to Kaggle (make public)
4. Upload video to YouTube
5. Submit on Kaggle competition page

## 📦 Development Setup

```bash
# 1. Clone repo
git clone https://github.com/EmrullahAydogan/Google_Tunix_Hack_Project.git
cd Google_Tunix_Hack_Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data
python scripts/download_data.py

# 5. Run tests (optional)
pytest tests/

# 6. Start development
# Edit files in src/, test in notebooks/
```

## 🎯 Next Steps

### Immediate (Hafta 1-2)
- [ ] Implement `src/tunix_project/data/dataset.py`
- [ ] Implement `src/tunix_project/training/reward.py`
- [ ] Create basic training notebook
- [ ] Download and explore GSM8K data

### Short-term (Hafta 3-4)
- [ ] Complete training pipeline
- [ ] Run baseline experiments
- [ ] Hyperparameter tuning
- [ ] Model evaluation

### Final (Hafta 5-8)
- [ ] Final training runs
- [ ] Write Kaggle writeup
- [ ] Record YouTube video
- [ ] Submission preparation

---

**Created:** November 16, 2025
**Competition Deadline:** January 12, 2026
