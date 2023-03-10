"""
Author: JJ Hennessy
Date: 10 March 2023
Description: A Random Forest Regressor model applied to the popular Melbourne Housing dataset to predict housing prices
             based on a dataset of house features and associated prices. Experiments tracked using mlflow.
"""

import os
import warnings
import sys

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(10)

    # Read the Melbourne housing CSV file from the URL
    melbourne_csv_url = (
        "https://raw.githubusercontent.com/njtierney/melb-housing-data/master/data/housing.csv"
    )
    try:
        melbourne_df = pd.read_csv(melbourne_csv_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    melbourne_df.dropna(inplace=True)
    train, test = train_test_split(melbourne_df)

    # Build forest model to predict price of a house in Melbourne based on house features
    # House Features: rooms, bathroom, landsize, lattitude, longtitude
    melbourne_features = ['rooms', 'bathroom',
                          'landsize', 'latitude', 'longitude']
    train_x = train[melbourne_features]
    test_x = test[melbourne_features]
    train_y = train[['price']]
    test_y = test[['price']]

    with mlflow.start_run(run_name="melbourne_forest_model") as run:
        n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        forest_model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=1)
        forest_model.fit(train_x, train_y)
        melbourne_preds = forest_model.predict(test_x)

        mae = mean_absolute_error(test_y, melbourne_preds)
        r2 = r2_score(test_y, melbourne_preds)

        print("RandomForestRegressor model (n_estimators={:f}, mae={:f}):".format(
            n_estimators, mae))
        print("MAE: %s" % mae)
        print(" R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                forest_model, "model", registered_model_name="MelbourneHousingPrices")
        else:
            mlflow.sklearn.log_model(forest_model, "model")


if __name__ == "__main__":
    main()
