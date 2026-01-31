
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(r'C:\Fatigue_level')

from src.data_engineering import Data_Engineering

# C:\Fatigue_Level\src\data_engineering.py

data_path = r'C:\Fatigue_Level\Data\Preprocessed_Data\Encoded Data\human_decision_fatigue_encoded.csv'

df = pd.read_csv(data_path)

df.info()

df.isnull().sum()

# Datani yoqotmaslik uchun kopiya qip olindi
data = df.copy()

X = data.drop(columns='Fatigue_Level')
y = data['Fatigue_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# We need y_train and y_test to be integers not float or other types so changed their type
y_train = y_train.astype(int)
y_test = y_test.astype(int)

de = Data_Engineering(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

de.feature_adding().feature_selection().skewness_finding().data_saving()



