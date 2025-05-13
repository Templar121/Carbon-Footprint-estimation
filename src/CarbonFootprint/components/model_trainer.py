import pandas as pd
import os
from CarbonFootprint import logger
from xgboost import XGBRegressor
import joblib
from CarbonFootprint.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

       
        model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            min_child_weight=self.config.min_child_weight,
            random_state=self.config.random_state,
            objective=self.config.objective,
        )

        
        model.fit(train_x, train_y)

        # Save the model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))