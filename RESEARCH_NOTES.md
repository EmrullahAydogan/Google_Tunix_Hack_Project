# Google Tunix Hack - Araştırma Notları

## 📋 Yarışma Özeti

**Yarışma Adı:** Google Tunix Hack - Train a model to show its work
**Platform:** Kaggle
**Ödül Havuzu:** $100,000
**Son Tarih:** 12 Ocak 2026
**Yarışma Linki:** https://www.kaggle.com/competitions/google-tunix-hackathon

### Ödül Dağılımı
- 🥇 1. Yer: $30,000
- 🥈 2. Yer: $25,000
- 🥉 3. Yer: $15,000
- 4-6. Yer: Her biri $10,000

### Gereksinimler
1. ✅ Tunix kullanarak çalışan bir eğitim pipeline'ı
2. ✅ Gemma2 2B veya Gemma3 1B modeli
3. ✅ Kaggle Writeup (maksimum 1,500 kelime)
4. ✅ Public Kaggle Notebook
5. ✅ YouTube video (maksimum 3 dakika)

---

## 🔧 Tunix Kütüphanesi

### Genel Bakış
Tunix (Tune-in-JAX), Google tarafından geliştirilen JAX tabanlı bir LLM post-training kütüphanesidir.

**Resmi Kaynaklar:**
- GitHub: https://github.com/google/tunix
- Dokümantasyon: https://tunix.readthedocs.io
- PyPI: `google-tunix`

### Kurulum

#### PyPI (Önerilen)
```bash
pip install "google-tunix[prod]"
```

#### GitHub'dan
```bash
pip install git+https://github.com/google/tunix
```

#### Kaynak Koddan (Geliştirme)
```bash
git clone https://github.com/google/tunix.git
cd tunix
pip install -e ".[dev]"
```

#### SGLang-Jax Desteği (Opsiyonel)
```bash
git clone git@github.com:sgl-project/sglang-jax.git
cd sglang-jax/python
pip install -e .
```

### Ana Özellikler

#### 1. Supervised Fine-Tuning (SFT)
- Full weights fine-tuning
- Parameter-efficient yöntemler: LoRA ve QLoRA

#### 2. Reinforcement Learning
- **PPO (Proximal Policy Optimization)**
- **GRPO (Group Relative Policy Optimization)**
- **GSPO (Token-level Goal-Seeking Policy Optimization)**

#### 3. Knowledge Distillation
- Logit distillation
- Attention transfer
- Feature pooling stratejileri

### Teknik Özellikler
- ✅ JAX-native implementation
- ✅ Flax NNX ile entegrasyon
- ✅ Modüler ve composable componentler
- ✅ Multi-device ve multi-host desteği
- ✅ TPU optimizasyonu
- ✅ Distributed training stratejileri (DP, FSDP, TP)

### Mevcut Örnek Notebook'lar
1. PEFT Gemma with QLoRA
2. GRPO training on grade school math problems
3. Logit distillation using Gemma models
4. Llama3/Qwen2 training with GRPO and SGLang-Jax

### Durum
⚠️ **Erken geliştirme aşamasında** - Aktif olarak yeni özellikler ekleniyor

---

## 🤖 Gemma Modelleri

### Gemma 2 2B

**Temel Özellikler:**
- **Parametre Sayısı:** 2 milyar
- **Context Uzunluğu:** 8,192 token
- **Training Data:** ~2 trilyon token
- **Mimari:** Multi-query attention (MQA)
- **Kullanım:** Genel amaçlı dil modeli

**장점:**
- Daha fazla parametre = Potansiyel olarak daha iyi performans
- Yerleşik multi-query attention
- Geniş training data

### Gemma 3 1B

**Temel Özellikler:**
- **Parametre Sayısı:** 1 milyar
- **Context Uzunluğu:** 32,000 token (4x daha fazla!)
- **Training Data:** ~2 trilyon token
- **Mimari:** 5:1 local-to-global attention ratio
  - Local layers: 1024 token span
- **Boyut:** Gemma 2 2B'nin sadece %20'si

**장점:**
- ✅ 4x daha büyük context window (32K vs 8K)
- ✅ Daha küçük deployment size
- ✅ Daha az memory requirement
- ✅ Gemma 2 2B'den daha iyi performans (küçük olmasına rağmen!)
- ✅ Optimized KV Cache
- ✅ Mobile-friendly (4GB+ RAM)
- ✅ Quantization-aware training

**Kısıtlamalar:**
- ⚠️ Text-only (multimodal değil)
- ⚠️ Görüntü işleme desteği yok

### Model Seçimi Önerisi
**Gemma 3 1B önerilir** çünkü:
1. Daha uzun context window (reasoning için önemli)
2. Daha verimli ve hızlı
3. Daha az kaynak tüketimi
4. Daha modern mimari
5. Better performance/size ratio

---

## 🧠 Chain-of-Thought (CoT) Reasoning

### Konsept
Chain-of-thought prompting, modellerin karmaşık problemleri adım adım düşünme sürecini doğal dil olarak ifade etmesini sağlayan bir tekniktir.

### Reinforcement Learning ile Entegrasyon
- RL ile CoT birleştirilerek modeller mantıksal düşünme stratejilerini öğrenebilir
- OpenAI o1, DeepSeek R1 gibi modeller bu yaklaşımı kullanıyor
- Tunix'in PPO, GRPO, GSPO algoritmaları tam da bu amaç için tasarlanmış

---

## 📊 Önerilen Veri Setleri

### 1. GSM8K (Grade School Math 8K)
**Özet:**
- 8,500 grade school math problem
- 7,500 training + 1,000 test problem
- Her problem 2-8 adım arası çözüm gerektirir
- Temel aritmetik işlemler

**Neden Önemli:**
- ✅ Yüksek kaliteli, linguistically diverse
- ✅ Her problemin step-by-step çözümü var
- ✅ Tunix örneklerinde kullanılıyor (GRPO notebook)
- ✅ Chain-of-thought için ideal

**Başarı Metrikleri:**
- Chain-of-thought + self-consistency: %74 accuracy

### 2. MATH Dataset
**Özet:**
- GSM8K'dan daha zorlu matematik problemleri
- Üst düzey matematik konuları
- Competition-level problems

**Kullanım:**
- Daha ileri düzey reasoning için
- GSM8K'da iyi sonuç alındıktan sonra

### 3. ThoughtSource
**Özet:**
- Meta-dataset ve kütüphane
- 15 farklı dataset'i birleştiriyor:
  - 7 scientific/medical QA
  - 3 general-domain QA
  - 5 math word problems

**장점:**
- ✅ Çeşitli domain'lerden örnekler
- ✅ CoT reasoning için özel olarak hazırlanmış
- ✅ Qualitative understanding için iyi

### 4. InfinityMATH (2025 - Yeni!)
**Özet:**
- 100,000+ synthesized samples
- Program-of-Thoughts (PoT) yaklaşımı
- 7 high-quality dataset'ten sentezlenmiş

**장점:**
- ✅ Çok büyük dataset
- ✅ Modern approach (PoT)
- ✅ 2025'te yayınlandı - çok güncel

### 5. University-level Math Reasoning Dataset
**Özet:**
- 13,500+ text-only problems
- 600+ multimodal problems
- Real-world STEM problems
- Step-by-step solutions

**장점:**
- ✅ Daha zorlu problemler
- ✅ Real-world applications
- ✅ Detaylı çözümler

---

## 🎯 Proje Stratejisi

### Önerilen Yaklaşım

#### Faz 1: Temel Setup (1-2 hafta)
1. ✅ Gemma 3 1B model ile başla
2. ✅ GSM8K dataset kullan
3. ✅ Tunix GRPO trainer ile fine-tune
4. ✅ Baseline performance ölç

#### Faz 2: Optimizasyon (2-3 hafta)
1. QLoRA ile parameter-efficient training
2. Hyperparameter tuning
3. Different RL algorithms dene (PPO vs GRPO vs GSPO)
4. Self-consistency implementation

#### Faz 3: İleri Düzey (2-3 hafta)
1. MATH dataset ile extension
2. Multi-dataset training
3. Custom reasoning dataset oluştur
4. Ensemble methods

#### Faz 4: Finalizasyon (1 hafta)
1. Kaggle Writeup yaz
2. Video hazırla
3. Public notebook optimize et
4. Final submission

### Donanım Gereksinimleri
- **Minimum:** T4 GPU (Google Colab ücretsiz)
- **Önerilen:** TPU v2/v3 (Kaggle/Colab TPU)
- **Optimal:** TPU v4 (Google Cloud)

### Başarı İçin Kritik Faktörler
1. ✅ Step-by-step reasoning açıkça göstermek
2. ✅ Diverse problem types
3. ✅ Self-consistency implementation
4. ✅ Efficient training pipeline
5. ✅ İyi dokümantasyon ve açıklama

---

## 📚 Teknik Referanslar

### Tunix
- GitHub: https://github.com/google/tunix
- Docs: https://tunix.readthedocs.io
- Blog: https://developers.googleblog.com/en/introducing-tunix-a-jax-native-library-for-llm-post-training/

### Gemma
- Official Tutorial: https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_finetune
- Flax Models: https://huggingface.co/google/gemma-2-2b-jpn-it-flax

### Datasets
- GSM8K: https://github.com/openai/grade-school-math
- ThoughtSource: https://github.com/OpenBioLink/ThoughtSource
- LLM Datasets: https://github.com/mlabonne/llm-datasets

### Chain-of-Thought
- Google Research: https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/
- Awesome LLM Reasoning: https://github.com/atfortes/Awesome-LLM-Reasoning

---

## 🚀 Sonraki Adımlar

1. ✅ Proje yapısı oluştur
2. ✅ Gerekli dependencies yükle
3. ✅ GSM8K dataset indir
4. ✅ Basit bir training pipeline oluştur
5. ✅ İlk baseline model eğit
6. ✅ Evaluation framework kur

---

**Güncelleme Tarihi:** 16 Kasım 2025
**Yarışma Son Tarih:** 12 Ocak 2026 (57 gün kaldı)
