import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append(r'C:\Fatigue_level')

from src.model_training import Model_traing

# Data Source

X_train_path = r'C:\Fatigue_Level\Data\Engineered_Data\X_train_transformed.csv'
y_train_path = r'C:\Fatigue_Level\Data\Engineered_Data\y_train.csv'
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
xgb = XGBClassifier()

models = {'Decision_tree':decision_tree, 
          'Random_forest':random_forest, 
          'XGboost_Classifier':xgb}

for name, model in models.items():

    model_traing = Model_traing(X_train=X_train, y_train=y_train, model_alg=model, model_name=name)
    model_traing.model_fit()
    model_traing.model_saving()


