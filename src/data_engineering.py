import pandas as pd 
import os
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
import numpy as np


logger = logging.getLogger("Engineering")
logger.propagate = False
logger.setLevel(logging.INFO)

out_path = r'C:\Fatigue_Level\Logging\data_engineering.log'

if not logger.handlers:
    handler = logging.FileHandler(out_path)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info('Feature Engineering Started')

class Data_Engineering:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def feature_adding(self):
        try:
            # Ideal uyqu vaqti 8 soat dep faraz qilindi
            self.X_train['Sleep_Debt'] = 8 - self.X_train['Sleep_Hours_Last_Night']
            self.X_test['Sleep_Debt'] = 8 - self.X_test['Sleep_Hours_Last_Night']
            # Sleep and Caffeine interraction
            self.X_train['Sleep_Caffeine_Interaction'] = self.X_train['Sleep_Hours_Last_Night'] * self.X_train['Caffeine_Intake_Cups']
            self.X_test['Sleep_Caffeine_Interaction'] = self.X_test['Sleep_Hours_Last_Night'] * self.X_test['Caffeine_Intake_Cups']
            # Strees and Load Interraction
            self.X_train['Stress_Cognitive_Interaction'] = self.X_train['Stress_Level_1_10'] * self.X_train['Cognitive_Load_Score']
            self.X_test['Stress_Cognitive_Interaction'] = self.X_test['Stress_Level_1_10'] * self.X_test['Cognitive_Load_Score']
            # Time per task to compete it
            self.X_train['Task_Completion_Time'] = self.X_train['Decisions_Made'] * self.X_train['Avg_Decision_Time_sec']
            self.X_test['Task_Completion_Time'] = self.X_test['Decisions_Made'] * self.X_test['Avg_Decision_Time_sec']
            
            logger.info(f'Feature adding is DONE SUCCESSFULLY')
            return self
        except Exception as a:
            logger.error(f'ERROR while feature adding:{a}')
    
    # Feature Selection for tree based models
    def feature_selection(self):
        try:
            model = RandomForestClassifier()
            model.fit(self.X_train, self.y_train)

            # Muhim featurelarni aniqlash
            feature_importances = model.feature_importances_

            # Muhimlilik darajasi boyicha sortirofka qilib olish
            important_features = pd.Series(feature_importances, index=self.X_train.columns)
            important_features = important_features[important_features > 0.01].index.to_list()

            self.X_train = self.X_train[important_features]
            self.X_test = self.X_test[important_features]

            logger.info(f'Feture Selection is DONE SUCCESSFULLY')
            return self
        
        except Exception as e:
            logger.info(f'ERROR while feature selection:{e}')
    
    def skewness_finding(self): # Skewness darajasi 0.70 dan oshgan featurelarni log yordamid transform qilamiz.
        try:
            # selecting numerical features
            num_features = self.X_train.select_dtypes(include=np.number).columns
            skewness = self.X_train[num_features].skew()

            # selectiong skewed features
            skewed_features = skewness[abs(skewness) > 0.70].index.tolist()

            # Log transformation
            for col in skewed_features:
                if (self.X_train[col] >= 0).all():
                    self.X_train[col] = np.log1p(self.X_train[col])
                    self.X_test[col] = np.log1p(self.X_test[col])

            logger.info('Skevness is Handeled SUCCESSFULLY')

            return self
        except Exception as e:
            logger.error(f'ERROR while finding skewness:{e}')
    
    def data_saving(self):

        try:
            # Transformed train set saving
            out_path = r'C:\Fatigue_Level\Data\Engineered_Data'
            os.makedirs(out_path, exist_ok=True)
            path = os.path.join(out_path, 'X_train_transformed.csv')
            self.X_train.to_csv(path, index=False)

            # Transformed test set saving
            out_path = r'C:\Fatigue_Level\Data\Engineered_Data'
            os.makedirs(out_path, exist_ok=True)
            path = os.path.join(out_path, 'X_test_transformed.csv')
            self.X_test.to_csv(path, index=False)

            logger.info(f'Transformed data is saved at {out_path}')

            out_path = r'C:\Fatigue_Level\Data\Engineered_Data'
            os.makedirs(out_path, exist_ok=True)
            path = os.path.join(out_path, 'y_train.csv')
            self.y_train.to_csv(path, index=False)

            # Transformed test set saving
            out_path = r'C:\Fatigue_Level\Data\Engineered_Data'
            os.makedirs(out_path, exist_ok=True)
            path = os.path.join(out_path, 'y_test.csv')
            self.y_test.to_csv(path, index=False)

            logger.info(f'y_train and y_test sets are also SAVED at: {out_path}')

            return self
        
        except Exception as e:
            logger.error(f'ERROR while saving the data :{e}')


