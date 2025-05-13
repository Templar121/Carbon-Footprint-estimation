import os
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
    median_absolute_error
)
from urllib.parse import urlparse
import numpy as np
import joblib
from CarbonFootprint.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from CarbonFootprint.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        ape = np.mean(np.abs((actual - pred) / actual)) * 100  # Average Percentage Error
        medae = median_absolute_error(actual, pred)
        return rmse, mse, mae, r2, ape, medae
    


    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        predicted_qualities = model.predict(test_x)

        (rmse, mse, mae, r2, ape, medae) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2, "ape": ape, "medae": medae}
        save_json(path=Path(self.config.metric_file_name), data=scores)