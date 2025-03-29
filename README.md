# Research Project: Comparative Simulation Study on Classification Methods at University of Tübingen.

This repository contains code for replicating and slightly extending the simulation study introduced by Kim (2010). The original study compares several classification algorithms: Decision Tree, Logistic Regression, and Artificial Neural Networks (ANNs)—across multiple data-generation scenarios. Here, we replicate this simulation and also include Random Forest as an additional algorithm to test. 

### Original Study

Kim (2010) conducts a simulation study to evaluate classification performance in different scenarios with:
- Different numbers of classes in the dependent variable (2, 3, 4).
- Different numbers of features (3, 5, 7).
- A mix of continuous and categorical features.
- Varying sample sizes (100, 500, 1000, 10,000).

### Our Simulation
1. We replicate the original three model types:
   - **Decision Trees** 
   - **Logistic Regression** 
   - **Artificial Neural Networks** 
2. We add a **Random Forest** model.
3. We record multiple performance metrics (accuracy, precision, recall, F1-score, and misclassification rate).
4. All code is in Python, with minimal external dependencies beyond standard data-science libraries and PyTorch.


## Key Files

| File                     | Description                                                                                                                                                          |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`generate_data.py`**  | Generates simulated datasets. Allows specifying number of outcome classes, number of predictors, how many predictors are categorical, and how many classes those cat predictors have. |
| **`prepare_data.py`**   | Handles data preprocessing.                                         |
| **`Logistic_Regression.py`** | Implements logistic regression with both logit and probit link functions.                         |
| **`Decision_Tree.py`**  | Custom decision tree class. Supports Gini, Entropy, and an F-Test criterion for splitting.                                                                          |
| **`Random_Forest.py`**  | Random Forest built atop our custom decision tree class, using bootstrapping and random feature subsets.                                                             |
| **`ANN_Pytorch.py`**    | Custom PyTorch-based ANN supporting multiple optimizers (SGD, Adam, RMSprop).        |
| **`main.py`**           | Puts the entire simulation process all together: data generation, preprocessing, training/evaluation for each model-hyperparameter combo, and storing results. |


