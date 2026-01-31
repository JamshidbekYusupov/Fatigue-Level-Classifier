import pandas as pd
import os, sys
import shap
import matplotlib.pyplot as plt


class xAI:

    def __init__(self, X_train, X_test, model):
        self.model = model
        self.X_test = X_test
        self.X_train = X_train
        self.shap_explainer = None
        self.shap_values = None

    def shap_object(self):

        self.shap_explainer = shap.Explainer(self.model, self.X_train)
        return self
    
    def shap_calculation(self):
        self.shap_values = self.shap_explainer(self.X_test)
        return self
    
    def shap_plot(self):

        # shap.initjs()
        # shap.force_plot(self.shap_values[0])
        shap.plots.waterfall(self.shap_values[0, :, 0])
        # shap.plots.beeswarm(self.shap_values)
        shap.plots.heatmap(self.shap_values[:100])
        plt.show()

        return self