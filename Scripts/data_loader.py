import pandas as pd
import sys
import os

sys.path.append(r'C:\Fatigue_level')
data_path = r'C:\Fatigue_level\Data\Raw_Data\human_decision_fatigue_dataset.csv'

from src.data_loader import Data_Loader

dl = Data_Loader(data_path)

df = dl.data_loader()