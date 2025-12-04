# ML-models-for-chickpea-lncRNAs-prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

---

This repository contains the complete implementation of three machine learning and deep learning models developed for my M.Sc. thesis titled:

> **“Exploring the long non-coding RNAs in chickpea through machine learning.”**

The models herein are built and evaluated for the prediction and classification of long non-coding RNAs (lncRNAs) in chickpea (*Cicer arietinum*). The project serves as a comprehensive solution for biologists and data scientists interested in computational approaches to transcriptomics, lncRNA discovery, and plant genomics.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

Long non-coding RNAs (lncRNAs) play significant roles in gene regulation and plant development. This repository explores computational strategies for lncRNA identification in chickpea using:

- Classical Machine Learning models
- Deep Learning frameworks

The project demonstrates the end-to-end pipeline, including data preparation, feature extraction, model training, evaluation, and visualization.


## Features

- **Preprocessing:** Ready-to-use scripts and notebooks for data cleaning and formatting.
- **Feature Engineering:** Extraction of biologically relevant sequence features.
- **Model Implementations:** Classical ML (e.g., SVM, RF) and Deep Learning models (e.g., CNN, LSTM).
- **Evaluation:** Model performance metrics, visualization, and comparative analysis.
- **Reproducibility:** Jupyter Notebooks with clear, step-by-step explanations.

---

## Repository Structure

```
ML-models-for-chickpea-lncRNAs-prediction/
│
├── data/                   # Datasets for training and testing
├── notebooks/              # Jupyter Notebooks for each stage/model
├── scripts/                # Standalone Python scripts (preprocessing, training, etc.)
├── results/                # Output files: model weights, figures, metrics
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

> The majority of the code is developed in **Jupyter Notebook (68.4%)**, with core logic and utilities in **Python (31.6%)**.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SUBASHPALANIMUTHU/ML-models-for-chickpea-lncRNAs-prediction.git
    cd ML-models-for-chickpea-lncRNAs-prediction
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the browser link provided to explore and execute the notebooks.

---

## Usage

- Explore the `notebooks/` directory to view and execute the full workflow:
    1. Data preprocessing and exploration
    2. Feature extraction and selection
    3. Model training (ML and DL)
    4. Performance evaluation

- Run standalone scripts from the `scripts/` directory for automation or batch processing.

**Example:**  
To train and test a model, open the relevant notebook (e.g., `notebooks/02-Train-Model.ipynb`) and follow the instructions inside.

---

## Models

This project implements and evaluates:

- **Machine Learning Models:**
    - Support Vector Machine (SVM)
    - Random Forest (RF)
    - Logistic Regression

- **Deep Learning Models:**
    - Convolutional Neural Network (CNN)
    - Long Short-Term Memory (LSTM)
    - Hybrid architectures

Each model is trained on hand-crafted features derived from RNA sequences, with thorough hyperparameter tuning and cross-validation.

---

## Data

- **Datasets:**  
    - Curated sets of chickpea RNA sequences (lncRNAs and coding RNAs)
    - Cleaned, labeled, and split into training and testing sets

- **Sources:**  
    - Publicly available databases and in-house annotations (refer to dataset documentation in the `data/` folder)

---

## Results

- The repository includes:
    - Model performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)
    - Visualizations (confusion matrices, ROC curves, etc.)
    - Comparative analysis of all implemented models

- See final results in the `results/` directory and summary tables/figures in the thesis report (or final notebook).

---

## Contributing

Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create your branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author:** Subash Palanimuthu
- **Email:** [YourEmail@example.com](mailto:YourEmail@example.com)
- **GitHub:** [SUBASHPALANIMUTHU](https://github.com/SUBASHPALANIMUTHU)

For questions, feedback, or collaborations, please open an issue or contact the author directly.

---
