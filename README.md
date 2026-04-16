# Machine Learning Playground

<p align="center">
  <img src="https://img.shields.io/badge/Focus-Machine%20Learning-0f766e?style=for-the-badge" alt="Focus badge" />
  <img src="https://img.shields.io/badge/Language-Python-1d4ed8?style=for-the-badge" alt="Language badge" />
  <img src="https://img.shields.io/badge/Mode-Learning%20Lab-f59e0b?style=for-the-badge" alt="Mode badge" />
</p>

<p align="center">
  <b>A handcrafted Machine Learning notebook</b>
</p>

<p align="center">
  <i>Algorithms, experiments, and mini-projects collected while learning how models think, fit, predict, and improve.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Style-Modern%20%2B%20Minimal-111827?style=flat-square" alt="Style badge" />
  <img src="https://img.shields.io/badge/Workflow-Learn%20%7C%20Build%20%7C%20Repeat-065f46?style=flat-square" alt="Workflow badge" />
  <img src="https://img.shields.io/badge/Focus%20Area-Only%20Machine%20Learning-7c3aed?style=flat-square" alt="Machine Learning only badge" />
</p>

<p align="center">
  <b>Signature:</b> Curated with GitHub Copilot
</p>

> [!NOTE]
> This folder is dedicated to Machine Learning only. It does not include AI-specific or general AI/ML branding.

---

## About This Repository

This repository is my working notebook for Machine Learning. It is intentionally practical: a place to implement algorithms from scratch, test ideas, compare approaches, and keep small projects that help build intuition.

Instead of trying to be a polished production library, this collection focuses on learning by doing. You will find scripts for regression, classification, tree-based methods, support vector machines, neural network experiments, and a few dataset-driven explorations.

<p align="center">
  <img src="https://img.shields.io/badge/Regression-Training%20%26%20Prediction-2563eb?style=flat-square" alt="Regression badge" />
  <img src="https://img.shields.io/badge/Classification-Decision%20Making-f97316?style=flat-square" alt="Classification badge" />
  <img src="https://img.shields.io/badge/Trees-Structure%20%26%20Splits-15803d?style=flat-square" alt="Trees badge" />
  <img src="https://img.shields.io/badge/Neural%20Nets-Pattern%20Learning-7c3aed?style=flat-square" alt="Neural nets badge" />
</p>

## What’s Inside

- Core Machine Learning implementations written in Python
- Introductory experiments with regression and classification
- Decision tree, random forest, KNN, Naive Bayes, SVM, and MLP practice code
- Small visualizations and output artifacts
- Dataset work, including Titanic-related exploration under the pandas folder

<p align="center">
  <img src="https://img.shields.io/badge/Progress-Hands--On%20Learning%20Log-0ea5e9?style=flat-square" alt="Progress badge" />
</p>

## Repository Map

The repository is now grouped by topic so each experiment lives where it belongs:

```text
Machine-Learning/
├── anomaly_detection/
├── boosting/
├── clustering/
├── data_preprocessing/
├── decision_trees/
├── dimensionality_reduction/
├── feature_selection/
├── hyperparameter_tuning/
├── imbalanced_learning/
├── KNN/
├── model_interpretability/
├── model_evaluation/
├── model_persistence/
├── naive_bayes/
├── neural_networks/
├── outputs/
├── pandas/
├── random_forest/
├── regression/
└── svm/
```

### Top-Level Folders

- `regression/` holds the general linear and logistic regression experiments.
- `svm/` holds SVM classification and regression demos in separate subfolders.
- `naive_bayes/` holds Naive Bayes classification demos and a binned-target regression-style example.
- `random_forest/` holds Random Forest classification and regression demos in separate subfolders.
- `anomaly_detection/` holds outlier and anomaly detection workflows.
- `boosting/` holds gradient boosting and XGBoost-style boosting alternatives.
- `clustering/` holds unsupervised clustering experiments such as KMeans and DBSCAN.
- `data_preprocessing/` holds tabular preprocessing and pipeline examples.
- `decision_trees/` holds decision tree classifier and regressor scripts.
- `dimensionality_reduction/` holds PCA-based experiments and pipelines.
- `feature_selection/` holds feature ranking and selection workflows.
- `hyperparameter_tuning/` holds model search and tuning workflows.
- `imbalanced_learning/` holds class-imbalance handling examples.
- `model_evaluation/` holds confusion matrix, ROC-AUC, and cross-validation scripts.
- `model_interpretability/` holds feature importance and interpretation workflows.
- `model_persistence/` holds save/load deployment-ready model examples.
- `neural_networks/` holds the perceptron and MLP experiments.
- `KNN/` contains organized classification, regression, visualization, data, and docs subfolders.
- `pandas/` holds the Titanic dataset exploration work.
- `outputs/` holds generated artifacts like plots, HTML previews, and lab output files.

### Notable Files

- `regression/linear_regression.py`
- `regression/logistic_regression.py`
- `svm/classification/svm_classification.py`
- `svm/regression/svm_regression.py`
- `naive_bayes/classification/naive_bayes.py`
- `naive_bayes/regression/naive_bayes_regression.py`
- `random_forest/classification/random_forest_classification.py`
- `random_forest/regression/random_forest_regression.py`
- `data_preprocessing/tabular_preprocessing_pipeline.py`
- `hyperparameter_tuning/grid_and_random_search.py`
- `decision_trees/decision_tree_classifier.py`
- `clustering/kmeans_clustering.py`
- `clustering/dbscan_clustering.py`
- `anomaly_detection/isolation_forest_anomaly_detection.py`
- `dimensionality_reduction/pca_classification_pipeline.py`
- `boosting/gradient_boosting_classification.py`
- `boosting/gradient_boosting_regression.py`
- `boosting/xgboost_style_hist_gradient_boosting.py`
- `feature_selection/mutual_info_feature_selection.py`
- `feature_selection/rfe_feature_selection.py`
- `model_interpretability/permutation_importance_interpretability.py`
- `model_evaluation/confusion_matrix_evaluation.py`
- `model_evaluation/roc_auc_evaluation.py`
- `model_evaluation/cross_validation_benchmark.py`
- `imbalanced_learning/class_weight_and_threshold.py`
- `model_persistence/save_and_load_model.py`
- `neural_networks/mlp.py`
- `KNN/classification/knn.py`
- `KNN/regression/reg.py`
- `outputs/tree.png`

## Featured Areas

### Regression
Regression scripts in this repository explore both linear and logistic-style modeling. They are useful for understanding how a model fits data, how predictions are produced, and how different implementations compare.

### Classification
Several scripts focus on classification problems, including Naive Bayes, SVM, KNN, decision trees, and random forest. These files are a good way to study the behavior of common supervised learning techniques.

### Topic Folders
The `svm/`, `naive_bayes/`, and `random_forest/` folders keep each topic separate and make the classification and regression examples easy to find side by side.

### Neural Networks
The `mlp.py` file captures experimentation with multilayer perceptron ideas. It is a good starting point for understanding how a simple feed-forward model is assembled and trained.

### Data Exploration
The pandas and Titanic dataset folders support hands-on analysis, preprocessing, and model-building workflows using real data.

### Additional Models
The standalone topic folders `clustering/`, `dimensionality_reduction/`, `boosting/`, `feature_selection/`, and `model_evaluation/` cover core ML areas beyond baseline supervised learning.

### Full Pipeline Topics
The repository also includes `data_preprocessing/`, `hyperparameter_tuning/`, `anomaly_detection/`, `imbalanced_learning/`, `model_interpretability/`, and `model_persistence/` so the ML workflow is covered from data preparation to evaluation and deployment-ready model saving.

## Learning Snapshot

<table>
  <tr>
    <th align="left">Theme</th>
    <th align="left">What it captures</th>
  </tr>
  <tr>
    <td>Regression</td>
    <td>Predicting continuous outcomes and understanding fit</td>
  </tr>
  <tr>
    <td>Classification</td>
    <td>Separating classes and comparing model behavior</td>
  </tr>
  <tr>
    <td>Trees</td>
    <td>Rules, splits, and interpretable decision paths</td>
  </tr>
  <tr>
    <td>Support Vector Machines</td>
    <td>Margin-based classification experiments</td>
  </tr>
  <tr>
    <td>Neural Networks</td>
    <td>Layered pattern learning and model training practice</td>
  </tr>
</table>

## How To Use This Repository

1. Open the file or folder you want to study.
2. Run the script in Python and observe the output.
3. Modify the data, parameters, or logic and compare results.
4. Keep notes on what each experiment teaches you.

If you want to run anything locally, make sure Python is installed and the required packages are available in your environment. Many of these scripts are lightweight and can be tested individually.

<p align="center">
  <img src="https://img.shields.io/badge/Tip-Run%20One%20Script%20at%20a%20Time-14b8a6?style=flat-square" alt="Tip badge" />
</p>

## Suggested Learning Path

If you are exploring the repository for the first time, a practical order is:

1. Start with `regression/linear_regression.py` and `regression/logistic_regression.py`
2. Move into `decision_trees/`, `svm/`, `naive_bayes/`, and `random_forest/` to compare classic supervised methods
3. Explore `KNN/` once the baseline models feel familiar
4. Move to `clustering/`, `dimensionality_reduction/`, `boosting/`, `feature_selection/`, and `model_evaluation/`
5. Add `data_preprocessing/`, `hyperparameter_tuning/`, and `anomaly_detection/` for practical production-style workflows
6. Study `imbalanced_learning/`, `model_interpretability/`, and `model_persistence/` to round out advanced practice
7. Open `neural_networks/` for the perceptron and MLP experiments
8. Use `pandas/` and `outputs/` when you want to study data prep and generated results

## Notes

- This is a learning repository, so some files may reflect experiments, drafts, or alternate implementations.
- The root is intentionally kept clean, with topic folders carrying the code.
- New projects can be added as more topics are explored.
- The theme here is strictly Machine Learning, so the README intentionally avoids AI-only framing.

## Purpose

The goal of this repository is simple: build intuition through implementation. Every script is part of the process of learning how Machine Learning models behave, how they are trained, and how they can be adapted to solve different problems.

---

<p align="center">
  <i>Created for learning, experimentation, and continuous improvement.</i><br />
  <b>GitHub Copilot</b> · <b>Machine Learning Playground</b>
</p>