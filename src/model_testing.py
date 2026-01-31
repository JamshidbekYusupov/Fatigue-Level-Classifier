import joblib
import os, sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tabulate
import pandas as pd
import logging

logger = logging.getLogger("Model_Testing")
logger.propagate = False
logger.setLevel(logging.INFO)

out_path = r'C:\Fatigue_Level\Logging\testing.log'

if not logger.handlers:
    handler = logging.FileHandler(out_path)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info('Model Testing has Started')

class Model_Prediction:
    def __init__(self, X_test, y_test, model, model_name):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.name = model_name
        self.y_pred = None
        self.metrics = {}
    
    def prediction(self):
        try:
            self.y_pred = self.model.predict(self.X_test)
            logger.info(f'X_test is predicted using {self.name} model')
            return self
        except Exception as e:
            logger.error(f'Error while prediction X_tesr: {e}')
    
    def evaluate(self):
        try:
            self.metrics[f'{self.name}_accuracy'] = accuracy_score(self.y_pred, self.y_test)
            self.metrics[f'{self.name}_precison'] = precision_score(self.y_pred, self.y_test, average='weighted', zero_division=True)
            self.metrics[f'{self.name}_recall'] = recall_score(self.y_pred, self.y_test, average='weighted', zero_division=True)
            self.metrics[f'{self.name}_f1'] = f1_score(self.y_pred, self.y_test, average='weighted', zero_division=True)
            logger.info(f'Metics are calculate for {self.name} model')
            metrics = [[f'{self.metrics[f'{self.name}_accuracy']}', f'{self.metrics[f'{self.name}_precison']}', f'{self.metrics[f'{self.name}_recall']}'], f'{self.metrics[f'{self.name}_f1']}']

            headers = ['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score']

            print(tabulate.tabulate(metrics, headers, floatfmt='.5f', tablefmt='orgtbl'))

            return self
        except Exception as e:
            logger.error(f'Error while calculating metrics for {self.name}')
    
    def metrics_saving(self):
        try:
            # Convert metrics dictionary to DataFrame
            df = pd.DataFrame([self.metrics])
            
            # Define the output CSV path
            output_dir = r'C:\Fatigue_Level\Metrics'
            output_file = os.path.join(output_dir, f'{self.name}_metrics.csv')
            # Save DataFrame to CSV
            df.to_csv(output_file, index=False)
            print(f"Metrics saved to {output_file}")
            logger.info(f'Metrics are saved for the model {self.name}')
            return self
        except Exception as e:
            logger.error(f'Error while calculating metrics for {self.name} model')
           
 





