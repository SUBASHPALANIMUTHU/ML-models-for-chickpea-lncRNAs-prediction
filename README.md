# ML Models for Chickpea lncRNAs Prediction

**Author:** SUBASHPALANIMUTHU  
**Thesis:** *Exploring the long non-coding RNAs in chickpea through machine learning*

---

## Introduction

Long non-coding RNAs (lncRNAs) play critical roles in regulating gene expression and plant development, yet their systematic identification in crops like chickpea (Cicer arietinum) remains a research challenge. This repository presents three distinct machine learning models developed to predict and annotate lncRNAs from chickpea genome data, forming the backbone of my M.Sc. thesis.

The project leverages Python and Jupyter Notebooks for code development, data exploration, and reproducibility, with all work performed under Ubuntu Linux.

---

## Methodology

1. **Data Collection and Preprocessing:**
   - Chickpea genome and transcript sequences were downloaded from **NCBI** databases.
   - Sequences were annotated using public datasets and verified through published literature.
   - The **plncRNA** tool was used for initial screening and identification of probable long non-coding RNAs.
   - Custom Python scripts were written for sequence feature extraction (length, GC content, ORF prediction, sequence motifs).

2. **Feature Engineering:**
   - Coding potential scores, sequence composition, and structure-based features were computed for each sequence.
   - Data was split into training (known lncRNA and mRNA) and testing sets to avoid overfitting and to allow fair performance comparisons.

3. **Model Development:**
   Three predictive models were implemented:

   - **Model 1: Random Forest Classifier**
     - Leverages ensemble decision trees to classify sequences as lncRNA vs. mRNA based on extracted features.
     - Tuned using cross-validation and feature importance analysis.
   
   - **Model 2: Deep Learning - Convolutional/ Recurrent Neural Network**
     - Utilizes keras/Tensorflow CNN (or LSTM) layers to capture complex sequence relationships.
     - Designed to learn data representations from raw sequence data.
     - Trained on labeled chickpea datasets.
   
   - **Model 3: Support Vector Machine (SVM)**
     - Utilizes SVMs with customized kernel functions to separate lncRNA and mRNA classes.
     - Applied various feature selection methods to improve predictive performance.

   (*All models were evaluated for accuracy, precision, recall, F1-score, and ROC-AUC.*)

4. **Validation and Results:**
   - Output from all models was compared against published chickpea lncRNA annotations.
   - Visualizations and evaluation metrics are provided in Jupyter Notebooks for transparency and reproducibility.

---

## Directory Structure

```
ML-models-for-chickpea-lncRNAs-prediction/
├── data/                # Raw and processed genomic datasets
├── notebooks/           # Jupyter Notebooks for development and results
├── src/                 # Python scripts (feature extraction, model building, evaluation)
├── requirements.txt     # Required Python libraries
├── README.md            # Project documentation
├── LICENSE              # Project license
```

---

## Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SUBASHPALANIMUTHU/ML-models-for-chickpea-lncRNAs-prediction.git
   cd ML-models-for-chickpea-lncRNAs-prediction
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   # Or use conda if preferred
   # conda env create -f environment.yml
   ```

3. **Access Notebooks**
   Launch Jupyter Notebook to explore code and results interactively:
   ```bash
   jupyter notebook
   ```

4. **Run Scripts**
   Feature extraction, model training, and testing can be executed by running corresponding scripts from the `src/` directory.

---

## Dependencies

- Python 3.7+  
- Jupyter Notebook  
- scikit-learn, keras/tensorflow, pandas, numpy, matplotlib, seaborn  
- plncRNA tool ([url or instructions])
- Ubuntu/Linux OS recommended for full compatibility

---

## Data Sources

- Chickpea genome and transcript datasets from [NCBI](https://www.ncbi.nlm.nih.gov/)
- Published lncRNA and mRNA datasets for supervised learning

---

## Results

- Performance metrics and comparisons among all models are plotted for clarity.
- Highest F1-score and ROC-AUC observed with [best performing model].
- All code and results are reproducible via provided notebooks.

---

## References

- [NCBI Chickpea Genomics](https://www.ncbi.nlm.nih.gov/)
- plncRNA tool documentation
- [Relevant Thesis/Publications]

---

## License

MIT License or specify appropriate terms.

---

## Contact

For questions, collaborations, or further information:  
[GitHub Profile](https://github.com/SUBASHPALANIMUTHU)

---

*If you use this code or data in your research, please cite or acknowledge this thesis and repository!*
