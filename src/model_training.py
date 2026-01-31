import os, sys
import joblib
import logging

logger = logging.getLogger("Model_Training")
logger.propagate = False
logger.setLevel(logging.INFO)

out_path = r'C:\Fatigue_Level\Logging\model_training.log'

if not logger.handlers:
    handler = logging.FileHandler(out_path)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info('Model Training has Started')

class Model_traing:
    def __init__(self, X_train, y_train, model_alg, model_name):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
        self.model_alg = model_alg
        self.model_name = model_name
    
    def model_fit(self):
        try:
            self.model = self.model_alg.fit(self.X_train, self.y_train)

            logger.info(f'Model is trained with {self.model_alg}')
            return self
        except Exception as e:
            logger.error(f'Model is not trained!! ERROR: {e}')
    
    def model_saving(self):
        try:
            model_path = r'C:\Fatigue_Level\Models\All_models'
            model_file = os.path.join(model_path, f'{self.model_name}.joblib')
            joblib.dump(self.model, model_file)
        except Exception as e:
            logger.error(f'{self.model} is NOT SAVED: {e}')

        return self