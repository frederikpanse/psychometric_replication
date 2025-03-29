# Replication of Kim (2010): Comparative Simulation Study on Classification Methods

This repository contains code and documentation for replicating and slightly extending the simulation study introduced by Kim (2010). The original study compares several classification algorithms—Decision Trees, Logistic Regression, and Artificial Neural Networks (ANNs)—across multiple data-generation scenarios. Here, we replicate these comparisons and also include Random Forests as an additional model. 

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Files](#key-files)  
3. [Installation and Requirements](#installation-and-requirements)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Methodology Summary](#methodology-summary)  
7. [How to Cite / References](#how-to-cite--references)  
8. [License](#license)

---

## Overview

### Original Paper

Kim (2010) conducts a simulation study to evaluate classification performance in scenarios with:
- Different numbers of classes in the dependent variable (2, 3, 4).
- Different numbers of features (3, 5, 7).
- A mix of continuous and categorical features.
- Varying sample sizes (100, 500, 1000, 10,000).

### Our Extensions
1. We replicate the original three model types:
   - **Decision Trees** (with Gini, Entropy, and an F-Test criterion).
   - **Logistic Regression** (both logit and probit link functions).
   - **Artificial Neural Networks** (custom PyTorch implementation).
2. We add a **Random Forest** model.
3. We record multiple performance metrics (accuracy, precision, recall, F1-score, and misclassification rate).
4. All code is in Python, with minimal external dependencies beyond standard data-science libraries and PyTorch.

---

## Key Files

| File                     | Description                                                                                                                                                          |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`generate_data.py`**  | Generates synthetic datasets. Allows specifying number of outcome classes, number of predictors, how many predictors are categorical, and how many classes those cat predictors have. |
| **`prepare_data.py`**   | Handles data preprocessing (label encoding, train/test splitting with optional stratification, standard scaling of features).                                         |
| **`Logistic_Regression.py`** | Implements logistic regression with both logit and probit link functions. Includes training via gradient descent and evaluation metrics.                          |
| **`Decision_Tree.py`**  | Custom decision tree class. Supports Gini, Entropy, and an F-Test criterion for splitting.                                                                          |
| **`Random_Forest.py`**  | Random Forest built atop our custom decision tree class, using bootstrapping and random feature subsets.                                                             |
| **`ANN_Pytorch.py`**    | Custom PyTorch-based ANN supporting multiple hidden layers, adjustable hidden neuron counts, multiple optimizers (SGD, Adam, RMSprop), and ReLU activations.        |
| **`main.py`**           | Coordinates the entire simulation process: data generation, preprocessing, training/evaluation for each model-hyperparameter combo, and storing results in a CSV file. |
| **`Simulation_Replication_test.pdf`** | A short report describing the rationale for replication, summarizing methods, and presenting highlights of the results.                                |

---

## Installation and Requirements

1. **Clone or Download**  
   ```bash
   git clone https://github.com/YourUserName/replication-kim-2010.git
   cd replication-kim-2010
