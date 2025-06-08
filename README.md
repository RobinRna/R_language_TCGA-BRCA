# R_language_TCGA-BRCA

This repository contains the code and final report for the final homework of the *Data Science: R Basics* course, at the Coll&Univ. T.C. Coll Rod.Zone (CTCCR) in University of Science and Technology Beijing (USTB).

---

## Systematic Evaluation of Machine Learning Models for PAM50-Based Breast Cancer Subtyping

This study conducts a comprehensive comparative analysis of various machine learning models for classifying breast cancer into PAM50 molecular subtypes. The models are trained and evaluated using gene copy number variation (CNV) data from The Cancer Genome Atlas (TCGA).

### Abstract

This study conducts a systematic evaluation and comparative analysis of various machine learning models—**LightGBM, XGBoost, Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and Support Vector Machine (SVM)**—for PAM50-based breast cancer subtyping. We construct a dataset from TCGA-BRCA gene copy number profiles and apply a structured train/validation/test split for evaluation. A manual grid search is employed to optimize key hyperparameters for each model.

Experimental results demonstrate that the gradient boosting models, **XGBoost and LightGBM, achieve superior overall performance** in terms of accuracy and F1-score on this CNV-based classification task. The deep learning models (MLP, CNN) and SVM showed less competitive performance under the specific conditions of this study, highlighting the challenge of applying these architectures to high-dimensional genomic data with limited sample sizes. This work underscores the effectiveness of ensemble tree methods for tabular genomic data and provides a framework for model selection in cancer subtype classification.

### Models Implemented

The following models are evaluated in this project:
-   **LightGBM** (`xena_lightgbm`)
-   **XGBoost** (`xena_xgboost`)
-   **Support Vector Machine (SVM)** with RBF Kernel (`xena_svm`)
-   **Multilayer Perceptron (MLP)** (`xena_mlp`)
-   **1D Convolutional Neural Network (CNN)** (`xena_cnn`)

### Dataset

-   **Source**: [UCSC Xena Hub](https://tcga.xenahubs.net/)
-   **Project**: TCGA Breast Invasive Carcinoma (TCGA-BRCA)
-   **Data Type**: **Gene-level Copy Number Variation (CNV)**, processed using the GISTIC2 algorithm.
    -   *File*: `Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes`
-   **Labels**: **PAM50 Subtypes** from the clinical matrix, specifically using the `PAM50_mRNA_nature2012` attribute.
    -   *File*: `TCGA.BRCA.sampleMap_BRCA_clinicalMatrix`

### File Structure

```
.
├── LICENSE.txt
├── README.md
├── 面向乳腺癌PAM50亚型的多模型机器学习方法.pdf
├── xena_cnn/
│   ├── data/
│   ├── data_splits/
│   ├── model_cnn/
│   ├── results_cnn/
│   ├── 02_data_preprocessing.R
│   ├── 03_data_splitting.R
│   └── 06_cnn_classification.R
├── xena_lightgbm/
│   ├── ... (similar structure)
├── xena_mlp/
│   ├── ... (similar structure)
├── xena_svm/
│   ├── ... (similar structure)
└── xena_xgboost/
    ├── data/
    ├── data_splits/
    ├── model/
    ├── results/
    ├── 01_data_download.R
    ├── 02_data_preprocessing.R
    ├── 03_data_splitting.R
    └── 04_model_training_evaluation.R
```
Each `xena_<model_name>` directory is a self-contained experiment, sharing the same data preprocessing logic.

### Methodology

The project follows a systematic workflow encapsulated in the R scripts:

1.  **Data Download (`01_data_download.R`)**: This script provides instructions and URLs for manually downloading the required CNV and clinical data from the UCSC Xena Hub.

2.  **Data Preprocessing (`02_data_preprocessing.R`)**:
    -   Loads the raw CNV and clinical data.
    -   Filters for primary tumor samples.
    -   Extracts the PAM50 subtype labels and removes samples with missing or rare subtypes (count < 10).
    -   Merges the datasets.
    -   Performs feature selection by retaining the top ~25,000 genes with the highest variance.
    -   Standardizes feature names to be compatible with all R models.
    -   Saves the final processed dataset as `processed_TCGA_BRCA_CNV_subtypes.rds`.

3.  **Data Splitting (`03_data_splitting.R`)**:
    -   Loads the processed data.
    -   Performs stratified splitting based on the PAM50 subtype to create **training (60%)**, **validation (20%)**, and **test (20%)** sets.
    -   Saves the three data splits into the `data_splits/` directory.

4.  **Model Training & Evaluation (`04_...`, `05_...`, `06_...`)**:
    -   Each script trains a specific model.
    -   A **manual grid search** strategy is used for hyperparameter tuning.
    -   The best model configuration is identified based on performance on the validation set (or via cross-validation).
    -   A comprehensive evaluation is performed on the independent test set.
    -   All results, including performance metrics, confusion matrices, training history plots, and the final trained model object, are saved to the `results/` and `model/` directories of each experiment.

### Requirements & Setup

#### R Environment
Install the required R packages.
```R
install.packages(c(
  "lightgbm", "xgboost", "caret", "dplyr", "ggplot2", 
  "MLmetrics", "reshape2", "readr", "stringr", "kernlab", "e1071",
  "keras", "tensorflow"
))
```

#### Python Environment for MLP/CNN
The Keras/TensorFlow models require a specific Python environment managed via `reticulate`.

1.  **Install Python**: Ensure you have a compatible Python version installed (e.g., Python 3.8-3.10).

2.  **Create Virtual Environment**: In an R session, create the virtual environment. This project was tested with TensorFlow 2.10.
    ```R
    # Run this in R
    library(tensorflow)
    install_tensorflow(
      version = "2.10.0", 
      method = "virtualenv", 
      gpu = TRUE, # Set to FALSE if you don't have a compatible NVIDIA GPU
      envname = "tf210_gpu_venv"
    )
    ```

3.  **Install Specific NumPy Version**: TensorFlow 2.10 requires a specific version of NumPy.
    ```R
    # Run this in R
    library(reticulate)
    # Make sure you are using the correct environment
    use_virtualenv("tf210_gpu_venv", required = TRUE)
    py_install("numpy==1.23.4", pip = TRUE, force = TRUE)
    ```

### How to Run

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/R_language_TCGA-BRCA.git
    ```

2.  **Download Data**:
    -   Follow the instructions in `01_data_download.R` to download the CNV and Clinical Matrix files.
    -   Place the downloaded files into the `data/` directory of one of the project folders (e.g., `xena_xgboost/data/`).

3.  **Prepare Data**:
    -   Open an R console.
    -   Set your working directory to the location of the script you want to run (e.g., `cd R_language_TCGA-BRCA/xena_xgboost`).
    -   Run the preprocessing and splitting scripts. They only need to be run once.
        ```R
        source("02_data_preprocessing.R")
        source("03_data_splitting.R")
        ```
    -   Copy the generated `data_splits` folder to the other model directories.

4.  **Train and Evaluate a Model**:
    -   Run the desired model training script. For example, to run the XGBoost evaluation:
        ```R
        # In R, with working directory set to xena_xgboost
        source("04_model_training_evaluation.R") 
        ```
    -   To run the MLP evaluation, navigate to its directory (`cd ../xena_mlp`) and run its script:
        ```R
        # In R, with working directory set to xena_mlp
        source("05_mlp_classification_subtypes.R")
        ```

### Results Summary

The models were evaluated on the independent test set. The Gradient Boosting models demonstrated significantly better performance than the neural network and SVM models for this task.

| Model           | Accuracy | Macro F1-Score | Key Challenge(s)                                   |
| --------------- | :------: | :------------: | -------------------------------------------------- |
| **XGBoost**     | **0.657**| **0.661**      | Best overall performance.                            |
| **LightGBM**    |  0.618   |     0.616      | Slightly lower than XGBoost but still strong.        |
| **SVM (RBF)**   |  0.451   |     0.447      | Performance heavily dependent on feature selection.  |
| **MLP**         |  0.462   |     0.339      | Struggled to generalize from the limited samples.      |
| **CNN (1D)**    |  0.414   |     0.258      | Lowest performance; gene order lacks spatial meaning.  |

*Note: Performance metrics are based on the paper's final evaluation and the output from the provided scripts. MLP/CNN scores are based on a 5-class problem, while others are on a 4-class problem after removing the rare 'Normal-like' subtype.*

The primary conclusion is that for tabular, high-dimensional CNV data, **ensemble tree-based models like XGBoost and LightGBM are highly effective and robust**. Deep learning models may require larger datasets or more sophisticated feature engineering to be competitive.

### License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.

<a href="https://github.com/RobinRna/R_language_TCGA-BRCA">R_language_TCGA-BRCA</a> © 2025 by <a href="https://github.com/RobinRna">Lifeng Zhang</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="height: 12px; vertical-align: middle; margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="height: 12px; vertical-align: middle; margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" style="height: 12px; vertical-align: middle; margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" style="height: 12px; vertical-align: middle; margin-left: .2em;">