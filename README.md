# Cardiovascular-Disease-Diagnosis
Created a Machine Learning model using concepts of One-Hot Encoding, Random Forest Algorithm and Gradient Boosting
Model(XGBoost)

Context :- 
Cardiovascular disease (CVDs) is the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of five CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management.
This dataset contains 11 features that can be used to predict possible heart disease.
Let's train a machine learning model to assist with diagnosing this disease.

Attribute Information :- 
Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]

Following are the Steps included in the code in order :-
1. Imported the modules we'll be using with code :
  import numpy as np
  import pandas as pd
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  from xgboost import XGBClassifier
  import matplotlib.pyplot as plt
2. Load the dataset. As we can see, the variables:
Sex
ChestPainType
RestingECG
ExerciseAngina
ST_Slope
are categorical, so we one-hot encoded them.
3. We will split our dataset into train and test datasets. We will use the function train_test_split from Scikit-learn.
4. Build Decision Tree using the Scikit-learn implementation.
  The hyperparameters we will use and investigate here are
  min_samples_split: The minimum number of samples required to split an internal node.
    Choosing a higher min_samples_split can reduce the number of splits and may help to reduce overfitting.
  max_depth: The maximum depth of the tree.
    Choosing a lower max_depth can reduce the number of splits and may help to reduce overfitting.
5. Build Random Forest using the Scikit-learn implementation.
    All of the hyperparameters found in the decision tree model will also exist in this algorithm, since a random forest is an ensemble of many Decision Trees.
    One additional hyperparameter for Random Forest is called n_estimators which is the number of Decision Trees that make up the Random Forest.
    Remember that for a Random Forest, we randomly choose a subset of the features AND randomly choose a subset of the training examples to train each individual       tree.
    Note that you can modify this by setting the max_features parameter.
    You can also speed up your training jobs with another parameter, n_jobs.
    Since the fitting of each tree is independent of each other, it is possible fit more than one tree in parallel.
    So setting n_jobs higher will increase how many CPU cores it will use. Note that the numbers very close to the maximum cores of your CPU may impact on the           overall performance of your PC and even lead to freezes.
    Changing this parameter does not impact on the final result but can reduce the training time.
    We will run the same script again, but with another parameter, n_estimators, where we will choose between 10, 50, and 100. The default is 100.
6. Build teh XG Boost Model using the Scikit_learn implementation.
    The boosting methods train several trees, but instead of them being uncorrelated to each other, now the trees are fit one after the other in order to minimize      the error.
    The model has the same parameters as a decision tree, plus the learning rate.
    The learning rate is the size of the step on the Gradient Descent method that the XGBoost uses internally to minimize the error on each train step.
    One interesting thing about the XGBoost is that during fitting, it can take in an evaluation dataset of the form (X_val,y_val).
    On each iteration, it measures the cost (or evaluation metric) on the evaluation datasets.
    Once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), the training will stop.
    More iterations lead to more estimators, and more estimators can result in overfitting.
    By stopping once the validation metric no longer improves, we can limit the number of estimators created, and reduce overfitting.
