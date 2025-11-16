# Google Tunix Hack - Submission Requirements Checklist

## 📋 Zorunlu Deliverables

### ✅ 1. Kaggle Writeup
**Platform:** Kaggle Competition Page

**Gereksinimler:**
- [ ] **Title** (Çekici ve açıklayıcı başlık)
- [ ] **Subtitle** (Kısa özet)
- [ ] **Detailed Analysis** (Maksimum 1,500 kelime)
  - [ ] Problem tanımı
  - [ ] Yaklaşım ve metodoloji
  - [ ] Model mimarisi ve hyperparameters
  - [ ] Training süreç açıklaması
  - [ ] Sonuçlar ve bulgular
  - [ ] İyileştirmeler ve gelecek çalışmalar
- [ ] **Cover Image** (Media Gallery'de)
  - Önerilen boyut: 1920x1080 veya 16:9 ratio
  - Görseli projeyi temsil etmeli

**İçerik Önerileri:**
- Neden bu yaklaşımı seçtiniz?
- Hangi zorlukları aştınız?
- Model nasıl "reasoning" gösteriyor?
- Sonuçlar ne kadar iyi?

---

### ✅ 2. Public Kaggle Notebook
**Platform:** Kaggle Notebooks

**Teknik Gereksinimler:**
- [ ] **Public** olmalı (Private değil!)
- [ ] **Single Kaggle TPU session**'da çalışabilir olmalı
- [ ] **Reproducible** - Başkası çalıştırabilmeli
- [ ] **Model output** - Fine-tuned model direkt notebook'tan çıkmalı
- [ ] **Clear documentation** - Markdown cells ile açıklamalar

**Notebook İçeriği:**
```
1. Introduction & Setup
   - Problem tanımı
   - Dependencies yükleme
   - Environment setup

2. Data Loading & Preprocessing
   - Dataset yükleme
   - Data exploration
   - Preprocessing steps

3. Model Configuration
   - Gemma model yükleme (2B veya 1B)
   - Tunix configuration
   - Hyperparameters

4. Training Pipeline
   - Tunix trainer setup
   - Training loop
   - Logging & monitoring

5. Evaluation
   - Test set evaluation
   - Reasoning examples
   - Metrics & visualizations

6. Model Export
   - Save fine-tuned model
   - Export configurations
```

**Kod Kalitesi:**
- [ ] Clean, readable code
- [ ] Comments ve docstrings
- [ ] Error handling
- [ ] Memory efficient

---

### ✅ 3. YouTube Video
**Platform:** YouTube (public veya unlisted)

**Gereksinimler:**
- [ ] **Maksimum 3 dakika** (180 saniye)
- [ ] **Public veya Unlisted** (Private değil!)
- [ ] **High quality** (minimum 720p önerilir)
- [ ] **İyi ses kalitesi**

**Video İçeriği (Önerilen Yapı):**

**0:00-0:30 (30 sn) - Introduction**
- Projenin amacı
- Problem tanımı
- Kısaca yaklaşım

**0:30-1:30 (60 sn) - Methodology**
- Tunix kullanımı
- Gemma model seçimi
- Training stratejisi
- Reward function açıklaması

**1:30-2:30 (60 sn) - Demo & Results**
- Model reasoning örneği (live demo)
- "Show its work" özelliği
- Performance metrikleri
- Başarı hikayeleri

**2:30-3:00 (30 sn) - Conclusion**
- Önemli bulgular
- Sonraki adımlar
- Teşekkürler

**Teknik Öneriler:**
- [ ] Ekran kaydı (screen recording) kullan
- [ ] Ses kalitesine dikkat et (iyi mikrofon)
- [ ] Altyazı ekle (opsiyonel ama önerilir)
- [ ] Hızlı konuş ama anlaşılır ol
- [ ] Visual aids kullan (charts, examples)

---

### ✅ 4. Reproducibility Requirements
**Paylaşılması Gerekenler:**

#### Configuration Files
- [ ] **Training config** (YAML/JSON)
  ```yaml
  model:
    name: gemma-3-1b
    base_model: google/gemma-3-1b

  training:
    algorithm: GRPO  # veya PPO, GSPO
    learning_rate: 1e-5
    batch_size: 8
    num_epochs: 3
    warmup_steps: 100

  data:
    dataset: gsm8k
    train_size: 7500
    val_size: 1000
  ```

#### Reward Function
- [ ] **Reward function kodu** (açıkça tanımlanmış)
  ```python
  def compute_reward(response, ground_truth):
      """
      Reward function for reasoning quality

      Criteria:
      - Correctness: Does it get the right answer?
      - Step-by-step: Does it show reasoning steps?
      - Clarity: Is the explanation clear?
      """
      # Implementation details
  ```

#### Recipe/Pipeline
- [ ] **Complete training recipe**
  - Data preprocessing steps
  - Model initialization
  - Training hyperparameters
  - Evaluation metrics
  - Post-processing

- [ ] **Requirements.txt** veya **environment.yml**
  ```
  google-tunix[prod]
  jax
  flax
  optax
  datasets
  numpy
  pandas
  matplotlib
  ```

---

## 🎯 Teknik Gereksinimler

### Model Requirements
- [ ] **Gemma2 2B** veya **Gemma3 1B** kullanılmalı
- [ ] **Tunix library** ile fine-tuning yapılmalı
- [ ] Model **reasoning göstermeli** ("show its work")
- [ ] Step-by-step açıklamalar olmalı

### Training Requirements
- [ ] **Single Kaggle TPU session** constraint
- [ ] Training süresi: Max 9-12 saat (Kaggle TPU limiti)
- [ ] Memory efficient olmalı
- [ ] Checkpointing (ara kayıt) olmalı

### Output Format
- [ ] Model responses format:
  ```
  Question: [Problem]

  Reasoning:
  Step 1: [First step explanation]
  Step 2: [Second step explanation]
  ...
  Step N: [Final step]

  Answer: [Final answer]
  ```

---

## 📊 Judging Criteria (Tahmini)

Resmi judging criteria belirtilmemiş, ancak hackathon'larda genel olarak:

### 1. Innovation & Creativity (25%)
- Yaklaşımın yenilikçiliği
- Farklı reasoning strategies
- Unique insights

### 2. Technical Implementation (30%)
- Code quality
- Tunix kullanımı
- Model performance
- Reproducibility

### 3. Reasoning Quality (30%)
- "Show its work" ne kadar iyi
- Step-by-step açıklama kalitesi
- Accuracy vs explainability trade-off

### 4. Presentation (15%)
- Writeup kalitesi
- Video clarity
- Documentation

---

## 📝 Submission Timeline & Checklist

### 2 Hafta Önce
- [ ] Training tamamlanmış olmalı
- [ ] Model evaluation yapılmış olmalı
- [ ] Notebook cleanup başlanmalı

### 1 Hafta Önce
- [ ] Writeup draft hazır
- [ ] Video script hazır
- [ ] Configurations ve configs documented

### 3 Gün Önce
- [ ] Final notebook test (Kaggle TPU'da çalıştır)
- [ ] Video kaydı yapılmış
- [ ] Writeup finalize edilmiş

### 1 Gün Önce
- [ ] Video upload edilmiş
- [ ] Cover image hazır
- [ ] Final review yapılmış

### Submission Günü
- [ ] Tüm linkler test edilmiş
- [ ] Writeup submit edilmiş
- [ ] Notebook public yapılmış
- [ ] Video linki doğru çalışıyor

---

## 🔗 Useful Links

- **Competition Page:** https://www.kaggle.com/competitions/google-tunix-hackathon
- **Tunix GitHub:** https://github.com/google/tunix
- **Tunix Docs:** https://tunix.readthedocs.io
- **Gemma Models:** https://ai.google.dev/gemma

---

## ⚠️ Common Pitfalls (Kaçınılması Gerekenler)

- [ ] ❌ Notebook private bırakmak
- [ ] ❌ Video 3 dakikadan uzun
- [ ] ❌ Writeup 1,500 kelimeyi aşmak
- [ ] ❌ Reproducibility eksikliği
- [ ] ❌ "Show its work" özelliği zayıf
- [ ] ❌ Kaggle TPU limitlerini aşmak
- [ ] ❌ Dependencies belirtilmemiş
- [ ] ❌ Model reasoning eksik veya belirsiz

---

**Son Tarih:** 12 Ocak 2026
**Güncellenme:** 16 Kasım 2025
