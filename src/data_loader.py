import pandas as pd
import sys
# from src.basic_logging import get_logger
import logging

# file_path = r'C:\Fatigue_level\Logging\data_loader.log'
# logging.basicConfig(
#     filename= file_path, 
#     filemode='a',
#     level=logging.INFO,
#     format='%(asctime)s-%(levelname)s-%(message)s'
# )

# log_path = r'C:\Fatigue_level\Logging\data_loader.log'
# logger = logging.getLogger(__name__)
# handler = logging.FileHandler(log_path)
# formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info("Data Loading Started")


# logger = get_logger("data_loader", "data_loader.log")


logger = logging.getLogger("data_loader")
file_path = r'C:\Fatigue_level\Logging\data_loader.log'

if not logger.handlers:
    handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class Data_Loader:

    def __init__(self, path):
        self.path = path
    
    def data_loader(self):
        try:
            df = pd.read_csv(self.path)
            logging.info('Data is loaded successfuly')
            return df
        except Exception as e:
            logging.error(f'Error occured while data loading:{e}')
            raise