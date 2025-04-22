# System Threat Forecaster - Malware Prediction 

## Overview

This repository contains a Jupyter Notebook (`.ipynb`) detailing the process for participating in the Kaggle competition "System Threat Forecaster". The objective is to predict the probability that a machine will be infected with malware (`target` variable = 1) based on various system telemetry features collected by antivirus software.

The notebook covers data exploration, preprocessing, feature engineering, model training (exploring several algorithms), evaluation, and prediction generation.

**Competition Link:** [System Threat Forecaster](https://www.kaggle.com/competitions/System-Threat-Forecaster/)

## Dataset

*   **`train.csv`**: Contains system features and the `target` variable (0 = no malware detected, 1 = malware detected) for training the model.
*   **`test.csv`**: Contains system features for machines where predictions are required. Lacks the `target` variable.
*   **`submission.csv`**: The required output format, containing an `id` (simple index) and the predicted `target` value (0 or 1) for each machine in the test set.

## Exploratory Data Analysis

The notebook includes detailed Exploratory Data Analysis (EDA). Key findings include:

*   Analysis of unique values and distributions for features like `OSVersion`, `NumAntivirusProductsInstalled`, `RealTimeProtectionState`, `TotalPhysicalRAMMB`, etc.
*   Investigation of relationships, such as the most frequent `RealTimeProtectionState` when `IsPassiveModeEnabled` is active.
*   Identification and visualization of missing data patterns.
*   Correlation analysis between features, highlighting potential multicollinearity (e.g., between `OSEdition` and `OSSkuFriendlyName`).
*   Assessment of low-variance features (`IsBetaUser`, `AutoSampleSubmissionEnabled`).
*   Visualization of the target variable distribution.

## Methodology

The workflow implemented within the notebook follows these main steps:

1.  **Data Loading:** Load `train.csv` and `test.csv` using pandas.
2.  **Data Exploration (EDA):** Perform initial analysis to understand data characteristics, distributions, and missing values (as detailed in the Milestones).
3.  **Preprocessing:**
    *   **Combine Data:** Concatenate train and test sets (excluding target and ID) for consistent preprocessing.
    *   **Missing Value Imputation:** Use `SimpleImputer` to fill missing numerical values (median/mean) and categorical values (most frequent).
    *   **Categorical Feature Encoding:** Convert object-type columns into numerical representations using `LabelEncoder`. (Other methods like One-Hot Encoding were explored for low-cardinality features).
    *   **Numerical Feature Scaling:** Standardize numerical features using `StandardScaler`.
    *   **Handling Class Imbalance:** Apply SMOTE (`imblearn.over_sampling.SMOTE`) on the *training data* before the final model training to address the class imbalance.
4.  **Feature Engineering:** Create basic aggregated numerical features (`feature_sum`, `feature_mean`, `feature_std`) to potentially capture interaction effects.
5.  **Model Selection & Training:**
    *   Explore various models: `RandomForestClassifier`, `DecisionTreeClassifier`, `SGDClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `XGBoost`, `LightGBM`.
    *   Perform hyperparameter tuning using `GridSearchCV` for Decision Tree and AdaBoost examples.
    *   Focus on **LightGBM** for the final model due to its efficiency and performance on large tabular datasets.
    *   Implement **Stratified K-Fold Cross-Validation** during LightGBM training to get a robust estimate of performance and mitigate overfitting.
    *   Utilize **Early Stopping** during training to find the optimal number of boosting rounds.
6.  **Prediction:** Use the final trained LightGBM model to predict the probability of malware infection on the preprocessed test set. Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold (or potentially a tuned threshold).
7.  **Submission:** Create the `submission.csv` file in the specified format.
8.  **Model Persistence:** Save the trained LightGBM model using `pickle` (output file: `lgb_model.pkl`).

## Results

The final predictions submitted were generated using a **LightGBM** model trained on SMOTE-resampled data with Stratified K-Fold cross-validation.

*   **Evaluation Metric:** ROC AUC was the primary metric used for model evaluation and early stopping during cross-validation.
*   **Outputs:** The notebook generates `submission.csv` and saves the trained model as `lgb_model.pkl`.



## File Structure

```
.
├── Data/
|      ├── train.csv                
|      ├── test.csv               
|      ├── sample_submission.csv     
├── test.csv
├── System_Threat_Forecaster.ipynb 
├── submission.csv            
├── lgb_model.pkl             
└── README.md                 
```





**Key Libraries:**

*   pandas
*   numpy
*   scikit-learn
*   xgboost
*   lightgbm
*   imblearn (`scikit-learn-contrib`)
*   seaborn & matplotlib
*   pickle
*   jupyter (notebook or lab)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lokabhiramchintada/System-Threat-Forecaster
    cd System-Threat-Forecaster
    ```
2.  **Download Data:** Obtain `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/System-Threat-Forecaster/data). Place them in a location accessible by the notebook (e.g., create a `/kaggle/input/System-Threat-Forecaster/` structure within the repo, or update the paths in the notebook).
3.  **Launch Jupyter:** Start Jupyter Notebook or JupyterLab from your terminal in the repository directory:
    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```
4.  **Open and Run Notebook:** Navigate to and open the `System_Threat_Forecaster.ipynb` file (or your specific notebook name) in the Jupyter interface. Run the cells sequentially from top to bottom.
5.  **Output:** The script will generate `submission.csv` and `lgb_model.pkl` in the same directory where the notebook is located (or as specified by paths within the notebook).

## Future Work

*   More advanced feature engineering (e.g., target encoding for high-cardinality features, sophisticated date feature extraction, interaction terms).
*   Experiment with different encoding techniques (CatBoost Encoding).
*   Explore other powerful models like CatBoost or Deep Learning architectures (TabNet).
*   Implement more sophisticated hyperparameter optimization techniques (e.g., Optuna, Hyperopt).
*   Build ensemble models by combining predictions from multiple strong learners (e.g., stacking XGBoost and LightGBM).
*   Further investigation into feature selection methods to reduce dimensionality.
*   Optimize the prediction threshold based on validation set performance instead of using the default 0.5.
