import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import logging
import numpy as np

logger = logging.getLogger("data_analysis")
logger.propagate = False
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(r'C:\Fatigue_level\Logging\data_analysis.log')
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


logger.info("Data Analysis boshlandi")


class Data_Analytics:
    def __init__(self, df):
        self.df = df
        
    def data_visualization(self):
        try:
            plt.figure(figsize=(10, 10))
            plt.hist(self.df['Decision_Fatigue_Score'], bins=30)
            plt.xlabel('Decision Fatigue Score')
            plt.ylabel('Count')
            plt.title('Distribution of Fatigue Score')
            plt.show()

            logger.info('Visualization of Fatigue Distribution is done')
        except Exception as e:
            logger.error(f'Visualization of Distribution of Fatigue Score is not done')

        try: 
            plt.figure(figsize=(10, 10))
            self.df['Fatigue_Level'].value_counts().plot(kind='bar')
            plt.xlabel('Fatigue Level')
            plt.ylabel('Count')
            plt.title('Fatigue Level Distribution')
            plt.show()
            logger.info('Visualizaton of Fatigue level counts is DONE')
        except Exception as e:
            logger.error(f'Visualizaton of Fatigue level counts is NOT DONE:{e}')

        try:
            plt.figure(figsize=(10, 10))
            plt.scatter(self.df['Hours_Awake'], self.df['Decision_Fatigue_Score'], alpha=0.3)
            plt.xlabel('Hours Awake')
            plt.ylabel('Decision Fatigue Score')
            plt.title('Hours Awake vs Decision Fatigue')
            plt.show()
            logger.info('Visualization of Hours awake vs fatigue is DONE')
        except Exception as e:
            logger.error(f'Visualization of Hours awake vs fatigue is NOT DONE: {e}')

        try: 
            plt.figure(figsize=(10, 10))
            plt.scatter(self.df['Sleep_Hours_Last_Night'], self.df['Decision_Fatigue_Score'], alpha=0.3)
            plt.xlabel('Sleep Hours Last Night')
            plt.ylabel('Decision Fatigue Score')
            plt.title('Sleep vs Decision Fatigue')
            plt.show()
            logger.info('Visualization of Sleep vs fatigue is DONE')
        except Exception as e: 
            logger.error(f'Visualization of Sleep vs fatigue is NOT DONE:{e}')

        # Stress vs error rate

        try:
            plt.figure(figsize=(10, 10))
            plt.scatter(self.df['Stress_Level_1_10'], self.df['Error_Rate'], alpha=0.3)
            plt.xlabel('Stress Level')
            plt.ylabel('Error Rate')
            plt.title('Stress vs Error Rate')
            plt.show()
            logger.info('Visualization of Stress vs error rate is DONE')

        except Exception as e:
            logger.error(f'Visualization of Stress vs error rate is NOT DONE: {e}')

        corr = self.df.select_dtypes(include=np.number).corr()
        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(corr)
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation Matrix')
            plt.show()
            logger.info('Heatmap is DONE to find correlation')
        except Exception as e:
            logger.error(f'Heatmap is NOT DONE to find correlation')