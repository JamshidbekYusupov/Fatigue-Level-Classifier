import joblib
import os, sys
import pandas as pd

sys.path.append(r'C:\Fatigue_level')

X_test = pd.read_csv(r'C:\Fatigue_Level\Data\Engineered_Data\X_test_transformed.csv')
y_test = pd.read_csv(r'C:\Fatigue_Level\Data\Engineered_Data\y_test.csv')

model_path = r'C:\Fatigue_Level\Models\All_models'

decision_tree = os.path.join(model_path, 'Decision_tree.joblib')
random_forest = os.path.join(model_path, 'Random_forest.joblib')
XGboost_Classifier = os.path.join(model_path, 'XGboost_Classifier.joblib')

decision_tree = joblib.load(decision_tree)
random_forest = joblib.load(random_forest)
XGboost_Classifier = joblib.load(XGboost_Classifier)

models = {'Decision_tree': decision_tree,
          'Random_Forest': random_forest,
          'XGBoost_classification': XGboost_Classifier}

from src.model_testing import Model_Prediction


for name, model in models.items():
    mp = Model_Prediction(X_test=X_test, y_test=y_test, model=model, model_name=name)
    mp.prediction()
    mp.evaluate()
    mp.metrics_saving()

