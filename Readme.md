# Fatigue Level Classifier 🧠💤

## Overview

This project focuses on predicting the **fatigue level** of individuals based on various features. The dataset, sourced from **Kaggle**, is suitable for both **regression** and **classification** models. However, we specifically targeted classification, using the **Fatigue_Level** feature to categorize individuals into three groups: **low**, **moderate**, and **high** fatigue.

The goal of this project is to predict the **Fatigue Level** of a person based on their lifestyle data, such as sleep, stress, caffeine intake, etc.

## Data Description 📊

The dataset consists of several important features that contribute to the prediction of the fatigue level:

- **Features for Regression Models**: The dataset includes columns like `System_Recommendation` and `Decision_Fatigue_Score`, but these were dropped during preprocessing. 
- **Target Variable**: The **Fatigue_Level** feature, which includes three categories:
  - Low 💤
  - Moderate ⚡
  - High 🔥

### Data Preprocessing 🔧

We’ve applied various techniques to preprocess the data and ensure it’s ready for machine learning models:

- **Handling Missing Values**: We filled missing values using the `mode()` for categorical data and `mean()` for numerical data.
- **Encoding**: One-hot encoding and label encoding were applied to categorical variables.
- **Scaling**: MinMaxScaler was used to scale numerical features to a range of 0 to 1.

### Feature Engineering ✨

Several new features were created to enhance the model’s ability to predict fatigue:

- **Sleep Debt**: 
  - Assumption: Ideal sleep time is 8 hours.
  - Formula: `Sleep_Debt = 8 - Sleep_Hours_Last_Night`
  
- **Sleep and Caffeine Interaction**:
  - Formula: `Sleep_Caffeine_Interaction = Sleep_Hours_Last_Night * Caffeine_Intake_Cups`

- **Stress and Cognitive Load Interaction**:
  - Formula: `Stress_Cognitive_Interaction = Stress_Level_1_10 * Cognitive_Load_Score`

- **Task Completion Time**:
  - Formula: `Task_Completion_Time = Decisions_Made * Avg_Decision_Time_sec`

### Feature Selection and Transformation 🔍

To ensure we're only using the most relevant features, we implemented:

- **Random Forest Feature Importance**: We used the feature importance method of Random Forest to select features with an importance score greater than 0.01.
- **Skewness Check**: Features with skewness greater than 0.70 were transformed using the **log** function to make them more normal.

### Models Used 🤖

We tried three different models for classification:

1. **Decision Tree**
2. **Random Forest**
3. **XGBoost**

### Evaluation and Metrics 📈

The model performance was evaluated based on the **recall** metric, as the main focus is identifying **True Positives** (people with high fatigue). 

After comparing all models, **XGBoost** outperformed the others in terms of recall, making it the best choice for this project.

---

## Project Structure 📂

Here’s the structure of the project:

Fatigue-Level-Classifier/

├── Data/
│   ├── Best_Model/             # Final production-ready model
│   ├── Engineered_Data/        # Features ready for training
│   ├── Logging/                # Training/Error logs
│   ├── Metrics/                # JSON/CSV performance results
│   ├── Models/                 # Checkpoints during training
│   ├── Preprocessed_Data/      # Cleaned/Imputed data
│   └── Raw_Data/               # Original Kaggle/Source data
├── Notebook/
│   ├── data_analysis.ipynb
│   ├── data_loader.ipynb
│   ├── engineering.ipynb
│   ├── preprocessing.ipynb
│   ├── testing.ipynb
│   └── training.ipynb
├── Scripts/                    
│   ├── data_analysis.py
│   ├── data_engineering.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_testing.py
│   └── model_training.py
├── src/                        # Modular, reusable logic
│   ├── __init__.py             
│   ├── data_analysis.py
│   ├── data_engineering.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_testing.py
│   └── model_training.py
├── .gitignore                  # Prevents data/logs from being uploaded
├── README.md                   # Project documentation
└── requirements.txt            # Library dependencies

### Explanation:

- **Data/**: This folder contains all the data files, including raw data, preprocessed data, engineered features, and model logs.
- **Notebook/**: Jupyter Notebooks where all the data analysis, feature engineering, and model training are explored interactively.
- **Scripts/**: Python scripts for running tasks in a non-interactive way, such as data loading, feature engineering, and model training/testing.
- **src/**: Source code for various functions and modules used throughout the project.
- **requirements.txt**: A file listing the Python libraries required to run the project.
- **README.md**: This file, providing an overview of the project.
- **.gitignore**: A file to tell Git which files/folders to ignore during version control (e.g., virtual environments, cache files).
