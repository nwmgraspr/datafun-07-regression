"""
app_tips_case.py - Smaller Custom Linear Regression Project

Author: Your Name
Date: 2026-06

Purpose:
    - Use simple linear regression on a small dataset
    - Predict restaurant tip amount from total bill amount
    - Calculate slope and intercept
    - Generate fitted values and residuals
    - Evaluate R-squared and RMSE
    - Visualize the fitted line and residuals

Data Source:
    Seaborn built-in Tips dataset

Question:
    Does a larger restaurant bill tend to produce a larger tip?
"""

# === IMPORTS ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# === LOGGER CONFIGURATION ===

LOG: logging.Logger = get_logger("TipsRegression", level="DEBUG")
log_header(LOG, "Tips Regression Project")

# === PROJECT CONSTANTS ===

DATASET_NAME: Final[str] = "tips"

FEATURE_COL: Final[str] = "total_bill"
TARGET_COL: Final[str] = "tip"

FEATURE_LABEL: Final[str] = "Total Bill ($)"
TARGET_LABEL: Final[str] = "Tip Amount ($)"

EXAMPLE_FEATURE_VALUE: Final[float] = 30.0

# === PANDAS DISPLAY OPTIONS ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# ==========================================================
# SECTION 1: LOAD DATA
# ==========================================================

def load_data() -> pd.DataFrame:
    """Load the Tips dataset."""

    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = sns.load_dataset(DATASET_NAME)

    LOG.info(
        f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns"
    )

    return df


# ==========================================================
# SECTION 2: PREPARE MODELING DATA
# ==========================================================

def make_model_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean dataset containing only usable rows.
    """

    LOG.info("Creating modeling view")

    required_cols = [FEATURE_COL, TARGET_COL]

    df_model = df.dropna(subset=required_cols).copy()

    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Model rows: {df_model.shape[0]}")
    LOG.info(f"Rows removed: {df.shape[0] - df_model.shape[0]}")

    return df_model


# ==========================================================
# SECTION 3: BUILD X AND Y
# ==========================================================

def build_x_and_y(
    df_model: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target vector y.
    """

    LOG.info("Building X and y")

    X = df_model[[FEATURE_COL]].to_numpy()
    y = df_model[TARGET_COL].to_numpy()

    LOG.debug(f"X shape: {X.shape}")
    LOG.debug(f"y shape: {y.shape}")

    return X, y


# ==========================================================
# SECTION 4: FIT REGRESSION MODEL
# ==========================================================

def fit_line(
    X: np.ndarray,
    y: np.ndarray,
) -> LinearRegression:
    """
    Fit a linear regression model.
    """

    LOG.info("Fitting linear regression model")

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    LOG.info(
        f"Regression Equation: "
        f"{TARGET_COL} = {slope:.4f} * {FEATURE_COL} + {intercept:.4f}"
    )

    return model


# ==========================================================
# SECTION 5: PREDICTIONS
# ==========================================================

def predict(
    model: LinearRegression,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate fitted values and one example prediction.
    """

    LOG.info("Generating predictions")

    y_hat = model.predict(X)

    example_x = np.array([[EXAMPLE_FEATURE_VALUE]])

    example_prediction = float(model.predict(example_x)[0])

    LOG.info(
        f"Predicted tip for a ${EXAMPLE_FEATURE_VALUE:.2f} bill: "
        f"${example_prediction:.2f}"
    )

    return y_hat


# ==========================================================
# SECTION 6: EVALUATE MODEL
# ==========================================================

def examine_fit(
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
) -> np.ndarray:
    """
    Calculate residuals and model evaluation metrics.
    """

    LOG.info("Evaluating model fit")

    residuals = y - y_hat

    r_squared = model.score(X, y)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    LOG.info(f"R-squared: {r_squared:.4f}")
    LOG.info(f"RMSE: {rmse:.4f}")

    LOG.info(f"Residual Min: {residuals.min():.4f}")
    LOG.info(f"Residual Max: {residuals.max():.4f}")
    LOG.info(f"Residual Mean: {residuals.mean():.4f}")

    return residuals


# ==========================================================
# SECTION 7: VISUALIZATIONS
# ==========================================================

def make_plots(
    df_model: pd.DataFrame,
    y_hat: np.ndarray,
    residuals: np.ndarray,
) -> None:
    """
    Create scatter plot with fitted line and residual plot.
    """

    feature_values = df_model[FEATURE_COL].to_numpy()
    target_values = df_model[TARGET_COL].to_numpy()

    # Scatter Plot with Regression Line
    plt.figure(figsize=(8, 5))

    scatter_plot: Axes = sns.scatterplot(
        x=feature_values,
        y=target_values,
    )

    order = np.argsort(feature_values)

    scatter_plot.plot(
        feature_values[order],
        y_hat[order],
        color="red",
        linewidth=2,
    )

    scatter_plot.set_title(
        "Total Bill vs Tip Amount with Regression Line"
    )

    scatter_plot.set_xlabel(FEATURE_LABEL)
    scatter_plot.set_ylabel(TARGET_LABEL)

    # Residual Plot
    plt.figure(figsize=(8, 5))

    residual_plot: Axes = sns.scatterplot(
        x=feature_values,
        y=residuals,
    )

    residual_plot.axhline(
        y=0,
        color="red",
        linestyle="--",
    )

    residual_plot.set_title(
        "Residuals vs Total Bill"
    )

    residual_plot.set_xlabel(FEATURE_LABEL)
    residual_plot.set_ylabel(
        f"Residual ({TARGET_LABEL})"
    )


# ==========================================================
# SECTION 8: SUMMARY
# ==========================================================

def summarize(
    df: pd.DataFrame,
    df_model: pd.DataFrame,
    model: LinearRegression,
) -> None:
    """
    Display project summary.
    """

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    LOG.info("=" * 50)
    LOG.info("PROJECT SUMMARY")
    LOG.info("=" * 50)

    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Feature: {FEATURE_COL}")
    LOG.info(f"Target: {TARGET_COL}")

    LOG.info(f"Original Rows: {df.shape[0]}")
    LOG.info(f"Model Rows: {df_model.shape[0]}")

    LOG.info(
        f"Equation: "
        f"{TARGET_COL} = {slope:.4f} * {FEATURE_COL} + {intercept:.4f}"
    )

    LOG.info("")
    LOG.info("Questions to Discuss:")
    LOG.info(
        "- Does a larger bill generally produce a larger tip?"
    )
    LOG.info(
        "- Is the relationship approximately linear?"
    )
    LOG.info(
        "- Are residuals randomly scattered around zero?"
    )
    LOG.info(
        "- Is the R-squared value reasonably strong?"
    )

    LOG.info("=" * 50)


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main() -> None:
    """
    Execute the complete regression workflow.
    """

    log_header(LOG, "Tips Regression Analysis")

    df = load_data()

    df_model = make_model_view(df)

    X, y = build_x_and_y(df_model)

    model = fit_line(X, y)

    y_hat = predict(model, X)

    residuals = examine_fit(
        model,
        X,
        y,
        y_hat,
    )

    make_plots(
        df_model,
        y_hat,
        residuals,
    )

    summarize(
        df,
        df_model,
        model,
    )

    plt.show()

    LOG.info("Analysis Complete")


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
```

