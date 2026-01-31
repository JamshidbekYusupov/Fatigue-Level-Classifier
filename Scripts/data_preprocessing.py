import pandas as pd
import os 
import sys

sys.path.append(r'C:\Fatigue_level')

from src.data_loader import Data_Loader

data_path = r'C:\Fatigue_level\Data\Raw_Data\human_decision_fatigue_dataset.csv'
dl = Data_Loader(data_path)

df = dl.data_loader()

df.info()

df = df.drop(columns=['System_Recommendation', 'Decision_Fatigue_Score'])

df.info()

from src.data_preprocessing import data_preprocessor

dp = data_preprocessor(df)

dp.missing_values().label_encoding().scaling()
