# Asteroids Under Control: A Data Science Approach to Planetary Defense

## Project Overview

This project aims to develop a machine learning model capable of predicting whether an asteroid is potentially hazardous to Earth based on its physical characteristics and orbital parameters. The goal is to contribute to planetary defense efforts by providing a tool for rapid and accurate risk assessment.

The project utilizes real-world data from NASA's Near-Earth Object (NEO) database and employs advanced machine learning algorithms, such as XGBoost and RandomForest, to classify asteroids based on their hazard potential.

## Key Features

- **Hazard Prediction:** Predicts whether an asteroid is potentially hazardous to Earth.
- **Data-Driven:** Uses real-world data from NASA's NEO database, available on Kaggle ([link](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)).
- **Machine Learning Powered:** Implements XGBoost and RandomForest algorithms for accurate classification.
- **Class Imbalance Handling:** Applies various oversampling and undersampling techniques (SMOTE, ADASYN, NearMiss, SMOTEENN) to address class imbalance in the dataset.
- **Explainable AI:** Provides insights into feature importance, helping to understand which factors contribute most to an asteroid's hazard potential.
- **Comprehensive Evaluation:** Evaluates model performance using precision, recall, F1-score, average precision (AP), PR-AUC, and confusion matrices.

## Project Structure

- **`asteroid_hazard_prediction.ipynb`**: Jupyter notebook containing the complete data science workflow, including data exploration, preprocessing, model building, and evaluation.
- **`README.md`**: This file, providing an overview of the project.

## Methodology

### 1. Data Acquisition and Exploration

- Data was sourced from NASA's NEO database on Kaggle.
- Python libraries such as Pandas and Matplotlib were used for exploratory data analysis (EDA).
- Relationships between asteroid parameters, such as diameter and absolute magnitude, were analyzed using Pearson and Spearman correlation, as well as visualized through scatter plots.
- Feature importance was assessed to identify the most influential features for predicting asteroid hazard potential.

### 2. Data Preprocessing

- Missing values and duplicates were handled to ensure the dataset's quality.
- Redundant features were dropped based on correlation analysis.
- New features, such as average diameter, were engineered to improve the predictive power.
- Class imbalance was addressed through oversampling and undersampling techniques, including SMOTE, ADASYN, NearMiss, and SMOTEENN.

### 3. Model Selection and Optimization

- XGBoost and RandomForest models were compared, and the best-performing one was selected for further optimization.
- The `scale_pos_weight` parameter was tuned to address the class imbalance.
- Hyperparameter tuning was performed using both manual selection and GridSearchCV to maximize XGBoost model performance.

### 4. Model Evaluation and Interpretation

- Models were evaluated using multiple performance metrics, including precision, recall, F1-score, average precision (AP), and confusion matrices.
- Two XGBoost models with optimized hyperparameters were compared: 
  - XGBoost with scaled weight
  - XGBoost with NearMiss undersampling
  - Random Forest with NearMiss undersampling

## Key Insights

- **Feature Importance:** Absolute magnitude is a very strong predictor of hazard potential, followed by relative velocity and miss distance, which frequently appear in tree splits.
- **Model Performance:** Oversampling and undersampling techniques significantly improved model recall, with NearMiss undersampling achieving the highest recall (0.94). Fine-tuning the `scale_pos_weight` parameter resulted in an even higher recall (0.97), with a slight precision improvement (0.59).
- **Performance Trade-offs:** All models show high recall (0.99–1.00), but precision is relatively low (0.56–0.57). In this context, prioritizing recall is crucial to ensure hazardous asteroids are not overlooked, even if it means accepting a higher rate of false positives.

## Final Model Selection

Based on performance metrics, **XGBoost with Scaled Weight** demonstrates the highest Average Precision (AP) at 0.73, making it the most effective model for distinguishing hazardous asteroids. Despite the trade-off with precision, XGBoost's ability to achieve high recall (0.97) makes it the most suitable model for planetary defense applications.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Random Forest
- Matplotlib
- Seaborn
- imblearn

## How to Run the Code

1. Clone the repository: 
    ```bash
    git clone https://github.com/your-username/asteroid-hazard-prediction.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook asteroid_hazard_prediction.ipynb
    ```

## Collaboration

I worked in a team with **X**, **Y**, and **Z**:

- **X** worked on the Random Forest model, focusing on optimizing performance through various hyperparameters.
- **Y** explored a low-code approach to assess multiple models simultaneously, including decision tree, random forest, XGBoost, and logistic regression.
- **Z** was responsible for deploying the final model and establishing a web application for real-time asteroid risk assessment based on the XGBoost model, which was identified as the best-performing model.

While the collaborative efforts are presented in the final team presentation (available [here](link)), the work presented in this notebook was solely done by me.

---

### Additional Notes:

- The **exploratory data analysis (EDA)** and **XGBoost model building** sections are highlighted in the team presentation, where I discuss key insights and findings.
- **Feature engineering** and **class imbalance handling** played a critical role in improving model performance, especially considering the highly imbalanced nature of the dataset.

---

