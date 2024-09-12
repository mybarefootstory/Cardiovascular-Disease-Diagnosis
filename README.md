# Cardiovascular Diseases Diagnosis

## Project Overview

Cardiovascular disease (CVD) is the leading cause of death globally, responsible for approximately 31% of all deaths. Early detection is crucial for preventing heart failure and improving patient outcomes. This project aims to predict the likelihood of heart disease using machine learning models on the Heart Failure Prediction dataset from Kaggle.

## Dataset

The dataset contains 11 features related to patient health and lifestyle, used to predict heart disease:

- **Age**: Age of the patient (years)
- **Sex**: Gender (M: Male, F: Female)
- **ChestPainType**: Type of chest pain experienced (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dl)
- **FastingBS**: Fasting blood sugar (1 if FastingBS > 120 mg/dl, 0 otherwise)
- **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved (60-202)
- **ExerciseAngina**: Exercise-induced angina (Y: Yes, N: No)
- **Oldpeak**: Depression of ST segment
- **ST_Slope**: Slope of the peak exercise ST segment (Up, Flat, Down)
- **HeartDisease**: Target variable (1: Presence of heart disease, 0: No heart disease)

## Project Steps

1. **Data Preprocessing**:
   - Categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope) were one-hot encoded.
   - Data split into 80% training and 20% validation datasets.

2. **Model Building**:
   - Decision Tree: Explored min_samples_split and max_depth parameters to control overfitting.
   - Random Forest: Tuned n_estimators, min_samples_split, and max_depth.
   - XGBoost: Utilized learning_rate and early_stopping_rounds to optimize model performance.

3. **Hyperparameter Tuning**:
   - Decision Tree: Tested different values for min_samples_split and max_depth, using accuracy metrics to evaluate model performance.
   - Random Forest: Adjusted n_estimators, max_depth, and min_samples_split to improve accuracy and prevent overfitting.
   - XGBoost: Employed early stopping to avoid overfitting, and adjusted learning_rate to improve performance.

4. **Model Evaluation**:
   - Model performance was evaluated using accuracy metrics for both the training and validation datasets.
   - Performance improvements were visualized through graphs comparing the accuracy of each model based on different hyperparameters.

## Results

- Decision Tree: Best performance with max_depth=4 and min_samples_split=50.
- Random Forest: Best configuration with n_estimators=100, max_depth=16, and min_samples_split=10.
- XGBoost: Achieved optimal performance with n_estimators=500 and early stopping at 16 rounds.

## Conclusion

Both Random Forest and XGBoost showed similar accuracy in predicting heart disease. Proper tuning of hyperparameters helped reduce overfitting and improved the model's generalization to unseen data.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository-link>

Install Dependencies:
bash
pip install -r requirements.txt

Run Model Training and Evaluation:
bash
python model_training.py

Requirements
Python 3.x
numpy
pandas
scikit-learn
xgboost
matplotlib
