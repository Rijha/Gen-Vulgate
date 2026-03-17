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

# 🧠 Core Features

-   🔍 **Hard Negative Mining Pipeline**
-   🧩 **Improved LineVul-based model architecture**
-   ⚙️ **Flexible training and evaluation workflows**
-   📊 **Reproducible experiments with fixed seeds**
-   📂 **External dataset management**

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

# 📊 Research Objectives

The project aims to:

-   Improve **cross-project vulnerability detection**
-   Reduce **false positives**
-   Improve **model generalization**
-   Enable **reproducible security research**

------------------------------------------------------------------------

# 🔗 Dataset Access

The dataset used in this research is hosted externally.

Google Drive:

https://drive.google.com/drive/folders/1j2AdOJxUCEKpeIkJFgTCfViE6j5nWq_u?usp=drive_link

After downloading, place the dataset in the project directory:

    dataset/

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

This project follows the licensing terms of the original LineVul
repository unless otherwise specified.

------------------------------------------------------------------------

# 🤝 Contributing

Contributions, issues, and research discussions are welcome.

Please open an issue or pull request for improvements.
