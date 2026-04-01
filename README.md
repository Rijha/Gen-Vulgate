# Gen-Vulgate

### Towards Generalizing AI-based Software Vulnerability Detection

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/status-research-green)

Gen-Vulgate is a research project focused on **improving the
generalization ability of AI-based software vulnerability detection
systems**. The project builds upon the **LineVul architecture** and
extends it with improved **data mining, negative sampling strategies,
and training workflows**.

The goal is to develop models that can **detect vulnerabilities across
diverse codebases while minimizing false positives**.

------------------------------------------------------------------------

# 📌 Motivation

Most deep learning models for vulnerability detection perform well on
the dataset they were trained on but **fail to generalize to unseen
codebases**.

This project addresses that challenge by introducing:

-   **Hard Negative Mining**
-   **Improved sampling strategies**
-   **Enhanced model preprocessing**
-   **More robust training pipelines**

These improvements help the model learn **stronger decision boundaries
between vulnerable and non-vulnerable code**.

------------------------------------------------------------------------

# 🧪 Key Contributions

### 📦 VulGate Dataset
- A large-scale, high-quality dataset containing **236,663 function-level samples across 792 projects**
- Covers **180 CWE types**
- Includes data updated through **May 2025**

### 🧠 Hard Negative Mining
- Incorporates **hard negative samples** (code pairs with >90% semantic similarity but different labels)
- Forces the model to learn **deep semantic patterns instead of superficial syntax**

### ⚖️ Data Quality & Balance
- Rigorous cleaning to remove **duplicates and label noise**
- Maintains a **balanced distribution** of vulnerable and secure samples

### 📏 Context Window Optimization
- Extending context window to **1024 tokens** significantly improves:
  - Long-range dependency capture
  - Performance on complex code structures

---

# 📊 Benchmark Performance

### 🚀 Breakthrough in Generalization
- Prior work shows **40–70% performance drop** on unseen projects
- Gen-Vulgate reduces degradation to only **4–6%**

### 🏆 State-of-the-Art Results
- Fine-tuned **UniXcoder-Base-Nine** achieves:
  - **F1 Score: 94.73%**
- Evaluated on the **BigVul benchmark**
- Outperforms existing LLM-based approaches

### 🛡 Robustness
- Verified through:
  - **Multi-seed experiments**
  - **Ablation studies**
- Performance gains are **stable and data-driven**

------------------------------------------------------------------------

# 📦 Dataset Comparison (Table 1)

| Dataset | Size | Balanced | CWEs | Hard Neg.% | Quality |
|--------|------|----------|------|------------|--------|
| Devign* | 25,872 | ✓ | – | 86.41% | ★★★★ |
| BigVul | 188,636 | ✗ | 91 | 19.45% | ★★★ |
| ReVeal* | 22,734 | ✗ | – | 36.13% | ★★★ |
| D2A* | 1.30M | ✗ | – | 6.21% | ★★ |
| CVEfixes | 168,089 | ✗ | 180 | 9.36% | ★★★★ |
| DiverseVul* | 348,987 | ✗ | 150 | 25.22% | ★★★ |
| MegaVul | 353,873 | ✗ | 169 | 0.00% | ★★★★ |
| **VulGate (Ours)** | **236,663** | **✓** | **180** | **~60.95%** | **★★★★★** |


# 🏆 BigVul Benchmark Results (Table 3)

| Category | Model | Context | F1 | P | R |
|----------|------|--------|----|----|----|
| Static | Cppcheck | – | 12 | 10 | 15 |
| Static | Infer | – | 19.5 | 15 | 28 |
| ML | SySeVR | – | 27 | 15 | 74 |
| ML | IVDetect | – | 35 | 23 | 72 |
| Decoder | CodeGPT-2 | 1024 | 90.45 | 97 | 84.45 |
| Decoder | CodeLlama | 1024 | 79.73 | 79.34 | 80.13 |
| Encoder | CodeBERT | 512 | 91 | 97 | 86 |
| Encoder | UniXcoder-Base | 1024 | 94.23 | 97.28 | 91.37 |
| **Encoder** | **UniXcoder-Base-Nine** | **1024** | **94.73** | **96.74** | **92.8** |


# 📊 VulGate Dataset Results (Table 5)

| Model | F1 | P | R |
|------|----|----|----|
| Cppcheck | 29.0 | 54.0 | 20.0 |
| Infer | 42.0 | 26.0 | 98.0 |
| CodeBERT | 85.9 | 83.2 | 89.0 |
| UniXcoder-Base | 87.0 | 85.4 | 88.7 |
| **UniXcoder-Base-Nine** | **88.9** | **87.7** | **90.0** |


# 🌍 Generalization Results (Table 6)

| Dataset | Model | F1 | P | R |
|--------|------|----|----|----|
| Linux | CodeBERT | 75.3 | 69.5 | 82.1 |
| Linux | UniXcoder-Base-Nine | 76.4 | 93.7 | 64.1 |
| PrimeVul | CodeBERT | 87.0 | 87.0 | 87.0 |
| PrimeVul | UniXcoder-Base-Nine | 89.14 | 87.1 | 91.3 |
| Claude | CodeBERT | 17.25 | 53.1 | 10.3 |
| Claude | UniXcoder-Base-Nine | 64.8 | 92.0 | 50.0 |


# 🧪 Ablation Studies

### 🔻 Without Hard Negatives
| Dataset | F1 |
|--------|----|
| VulGate (In-distribution) | 98.87 |
| Linux (Unseen) | 33.62 |
| Claude (Unseen) | 41.48 |

➡️ High train performance, **catastrophic generalization failure**

### ⚖️ Impact of Dataset Balance

| Ratio (V:S) | F1 |
|------------|----|
| 90:10 | 76.67 |
| 80:20 | 83.73 |
| 70:30 | 86.35 |
| **50:50** | **88.90** |


------------------------------------------------------------------------

# 🏗 System Pipeline

The overall training workflow follows the pipeline below:

    Raw Dataset
         │
         ▼
    Data Preprocessing
         │
         ▼
    Hard Negative Mining
         │
         ▼
    Dataset Construction
         │
         ▼
    Model Training (Gen-Vulgate)
         │
         ▼
    Evaluation & Testing

The **hard negative mining stage** helps the model learn from
**difficult negative examples** that are semantically similar to
vulnerable code.

------------------------------------------------------------------------

# 📂 Repository Structure

    Gen-Vulgate/
    │
    ├── linevul_main.py
    │       Main entry point for training and evaluation
    │
    ├── linevul_model.py
    │       Custom model architecture and preprocessing logic
    │
    ├── hardnegative_mining.py
    │       Pipeline for generating hard negative samples
    │
    ├── dataset/
    │       Dataset directory (download separately)
    │
    └── README.md

------------------------------------------------------------------------

# 🚀 Running the Project

The main entry point is:

    linevul_main.py

## 1️⃣ Train Model

Start training from scratch:

``` bash
python linevul_main.py --seed 42 --do_train
```

------------------------------------------------------------------------

## 2️⃣ Resume Training

Resume training from a checkpoint:

``` bash
python linevul_main.py --seed 42 --do_train --resume
```

------------------------------------------------------------------------

## 3️⃣ Run Evaluation / Testing

Evaluate the trained model:

``` bash
python linevul_main.py --seed 42 --do_test
```

------------------------------------------------------------------------

# ⚙️ Key Parameters

  Parameter      Description
  -------------- ---------------------------------
  `--seed`       Random seed for reproducibility
  `--do_train`   Enables training
  `--resume`     Resume training from checkpoint
  `--do_test`    Evaluate the model

------------------------------------------------------------------------

# 🔗 Dataset Access

The datasets used in this research are hosted externally.

## VulGate Dataset

Google Drive:

https://drive.google.com/drive/folders/1j2AdOJxUCEKpeIkJFgTCfViE6j5nWq_u?usp=drive_link

After downloading, place the dataset in the project directory:

    dataset/

## Vulgate+ Dataset

Google Drive:

https://drive.google.com/drive/folders/1PzwJFh7DEQKdOmzYwguoMwCdSGItD00m

------------------------------------------------------------------------

# 🎯 Best Checkpoint

The best-performing model checkpoint is available here:
https://drive.google.com/drive/folders/1KgafbWQcVVx6h9_Bt-01usg971pZ1c52

Includes: codebert & unixcoder-base-nine

------------------------------------------------------------------------

# 🙏 Acknowledgment

This project builds upon the **LineVul** research repository:

https://github.com/awsm-research/LineVul

We thank the original authors for their dataset, architecture, and
benchmarks that enabled further research in **AI-based vulnerability
detection**.

------------------------------------------------------------------------

# 📚 Citation

If you use this repository in your research, please consider citing the
original LineVul work and referencing this repository.

Example:

    @software{gen_vulgate,
      title={Gen-Vulgate: Towards Generalizing AI-based Software Vulnerability Detection},
      author={Rijha, Danyail},
      year={2026},
      url={https://github.com/Rijha/Gen-Vulgate}
    }

------------------------------------------------------------------------

# 📜 License

## MIT
