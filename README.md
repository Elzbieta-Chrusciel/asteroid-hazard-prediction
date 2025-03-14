# Asteroids Under Control: A Data Science Approach to Planetary Defense

## Project Overview

This project aims to develop a machine learning model capable of predicting whether an asteroid is potentially hazardous to Earth based on its physical characteristics and orbital parameters. The goal is to contribute to planetary defense efforts by providing a tool for rapid and accurate risk assessment.

The project utilizes real-world data from NASA's Near-Earth Object (NEO) database and employs advanced machine learning algorithms, such as XGBoost and RandomForest, to classify asteroids based on their hazard potential.

## Key Features

- **Hazard Prediction:** Predicts whether an asteroid is potentially hazardous to Earth.
- **Data-Driven:** Uses real-world data from NASA's NEO database, available on Kaggle ([link](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)).
- **Machine Learning Powered:** Implements XGBoost and RandomForest algorithms for accurate classification.
- **Class Imbalance Handling:** Applies various oversampling and undersampling techniques (SMOTE, ADASYN, NearMiss, SMOTE-ENN) to address class imbalance in the dataset.
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
- **Adressing Class Imbalance:** Oversampling and undersampling techniques significantly improved model recall, with NearMiss undersampling achieving the highest recall (0.94). Fine-tuning the `scale_pos_weight` parameter resulted in an even higher recall (0.97), along with a slight improvement in precision (0.61).
- **Performance Trade-offs:** All models show very high recall (0.99), but precision is relatively low (0.56–0.58). In this case, prioritizing recall is crucial to ensure hazardous asteroids are not overlooked,  even at the cost of a higher rate of false positives.

## Final Model Selection

Based on performance metrics, **XGBoost with Scaled Weight** shows the highest Average Precision (AP) at 0.73, making it the most effective model for distinguishing hazardous asteroids.  While its precision is still relatively low at 0.58, its high recall (0.99) makes it the most suitable model for planetary defense applications, despite the trade-off in precision.

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

## Collaboration & Additional Resources
The work presented in this repository was completed solely by me and was a key component of a larger project conducted in collaboration with Bartłomiej Brzostek [@bartbrzost](https://github.com/bartbrzost), Katarzyna Donaj [@donajkatarzyna](https://github.com/donajkatarzyna) and Tomasz Mazur [@Tom-Mazur](https://github.com/Tom-Mazur), who contributed to different aspects of the project:
- Katarzyna Donaj focused on optimizing the Random Forest model.
- Tomasz Mazur explored a low-code approach to assess multiple models simultaneously.
- Bartłomiej Brzostek deployed the final model and established a web application for real-time asteroid risk assessment based on the best-performing XGBoost model, which I built.

The full project, including additional insights from the team, can be explored in our Prezi presentation: [Prezi Link](https://prezi.com/p/edit/rygnld_akrmx/).

---
