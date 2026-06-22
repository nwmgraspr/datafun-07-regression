"""app_penguins_case_copy.py - Project script (example).

Author: Denise Case, Ralph Massaquoi
Date: 2026-06

Purpose:
    - simple linear regression on a dataset with a clear linear relationship
    - serving as the "good fit" baseline to compare against a messier case
    - choosing a feature (x) and a target (y) based on EDA findings
    - fitting a straight line with scikit-learn (with an optional numpy check)
    - reading off the slope and intercept
    - computing fitted values and residuals
    - examining R-squared and RMSE
    - making a prediction for a chosen feature value
    - charting the data, the fitted line, and the residuals

Data Source:
- Palmer Penguins, via Seaborn's built-in datasets (sns.load_dataset)
- Original: Horst AM, Hill AP, Gorman KB (2020). palmerpenguins.

Assumptions:
- The data contains columns like:
  species, island, bill_length_mm, bill_depth_mm,
  flipper_length_mm, body_mass_g, sex

Terminal command to run this file from the root project folder:

uv run python -m datafun.app_penguins_case

OBS:
  Don't edit this file - it should remain a working example.
  Copy it, rename it, and modify your copy.

  This script does NOT decide for you whether a straight line is a good
  description of the data. It fits the line and computes the numbers an
  analyst uses to make that call (residuals, R-squared, RMSE) - and then
  it shows them to you. Whether the relationship is "linear enough" is a
  judgment you make after looking. Doing the analysis is how you find out.

  This example was chosen because the relationship is expected to be a
  clean straight line. Compare its residual plot to a messier dataset's:
  here the residuals should scatter randomly around zero, with no pattern.
"""


# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging  # for type hinting only
from typing import Final  # for type hinting

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as pltimport seaborn as sns
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    # Load dataset
    df = sns.load_dataset("penguins")

    # Basic cleanup
    df = df.dropna(subset=["body_mass_g", "bill_length_mm", "flipper_length_mm"])

    # Features and target
    X = df[["species", "island", "sex", "bill_length_mm", "flipper_length_mm"]]
    y = df["body_mass_g"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    categorical_features = ["species", "island", "sex"]
    numeric_features = ["bill_length_mm", "flipper_length_mm"]

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ]
    )

    # Model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    print("\nModel Performance:")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # Optional: coefficients (after encoding)
    print("\nModel trained successfully.")


if __name__ == "__main__":
    main()
