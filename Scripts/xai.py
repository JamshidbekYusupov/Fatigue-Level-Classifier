import pandas as pd
import os, sys
import joblib
sys.path.append(r'C:\Fatigue_Level')

X_test = pd.read_csv(r'C:\Fatigue_Level\Data\Engineered_Data\X_test_transformed.csv')
X_train = pd.read_csv(r'C:\Fatigue_Level\Data\Engineered_Data\X_train_transformed.csv')

model_path = r'C:\Fatigue_Level\Models\All_models'

xgb = os.path.join(model_path, 'XGBoost_classifier.joblib')

xgb = joblib.load(xgb)


from src.xai import xAI

explainable_ai = xAI(X_train=X_train, X_test=X_test, model=xgb)

explainable_ai.shap_object().shap_calculation().shap_plot()
