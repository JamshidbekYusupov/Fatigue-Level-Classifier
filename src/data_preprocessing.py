import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import sys
import logging


logger = logging.getLogger("preprocessing")
logger.propagate = False
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(r'C:\Fatigue_level\Logging\preprocessing.log')
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class data_preprocessor:

    def __init__(self, df):
        self.df = df

        
    def missing_values(self):
        try:
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    if self.df[col].dtype == "object" or self.df[col].dtype == "str":
                        self.df[col].fillna(self.df[col].mode()[0], inplace = True)
                    else:
                        self.df[col].fillna(self.df[col].mean(), inplace = True)
            output_dir = r'C:\Fatigue_level\Data\Preprocessed_Data\Filled Data'
            os.makedirs(output_dir, exist_ok=True)
            otuput_path = os.path.join(output_dir, 'human_decision_fatigue_filled.csv')
            self.df.to_csv(otuput_path, index=False)

            logger.info('Missing values are filled')
            return self
        except Exception as e:
            logger.error(f'Missiing values are not filled error:{e}')
    

    def label_encoding(self):
        from sklearn.preprocessing import OrdinalEncoder
        categories = [['Low', 'Moderate', 'High']]
        ord_encoder = OrdinalEncoder(categories=categories)
        encoder = LabelEncoder()
        try:
            for col in self.df.columns:
                if col == 'Fatigue_Level':
                    self.df[col] = ord_encoder.fit_transform(self.df[[col]])
                elif self.df[col].dtype == 'object' or self.df[col].dtype == "str":
                    if self.df[col].nunique() <= 2:
                        dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                        self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
                    else: 
                        self.df[col] = encoder.fit_transform(self.df[col])
            output_dir = r'C:\Fatigue_level\Data\Preprocessed_Data\Encoded Data'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'human_decision_fatigue_encoded.csv')
            self.df.to_csv(output_path, index=False)
            logger.info('Dataset is Encoded')
            return self
        except Exception as e:
            logger.error(f'Dataset is NOT Encoded error: {e}')
        
    def scaling(self):
        try:
            scalar = MinMaxScaler()
            for col in self.df.columns.drop('Fatigue_Level'):
                self.df[col] = scalar.fit_transform(self.df[[col]])

            output_dir = r'C:\Fatigue_level\Data\Preprocessed_Data\Scaled Data'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'human_decision_fatigue_scaled.csv')
            self.df.to_csv(output_path, index=False)
            logger.info('Dataset is scaled')
            return self
        except Exception as e:
            logger.error(f'Dataset is not Scaled error:{e}')