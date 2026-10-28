"""Core Design of Experiments analysis routines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .utils import coded_column_name, decode_factor_value, prepare_analysis_frame

ModelType = Literal["linear", "interaction", "quadratic"]


@dataclass
class DoeModel:
    """Container for a fitted DoE response model."""

    response: str
    factors: list[str]
    model_type: ModelType
    formula: str
    result: RegressionResultsWrapper
    analysis_frame: pd.DataFrame
    coding: dict[str, dict[str, float]]


def build_formula(response: str, factors: list[str], model_type: ModelType = "quadratic") -> str:
    """Build a statsmodels formula for linear, interaction, or quadratic DoE models."""
    coded = [coded_column_name(factor) for factor in factors]
    terms = coded.copy()

    if model_type in {"interaction", "quadratic"}:
        terms.extend(f"{a}:{b}" for a, b in combinations(coded, 2))

    if model_type == "quadratic":
        terms.extend(f"I({factor} ** 2)" for factor in coded)

    return f"{response} ~ " + " + ".join(terms)


def fit_doe_model(
    df: pd.DataFrame,
    factors: list[str],
    response: str,
    model_type: ModelType = "quadratic",
) -> DoeModel:
    """Fit a DoE model after coding factors to dimensionless levels."""
    analysis_frame, coding = prepare_analysis_frame(df, factors, response)
    formula = build_formula(response, factors, model_type)
    result = smf.ols(formula=formula, data=analysis_frame).fit()
    return DoeModel(response, factors, model_type, formula, result, analysis_frame, coding)


def coefficient_table(model: DoeModel) -> pd.DataFrame:
    """Return model coefficients, standardized statistics, and p-values."""
    table = pd.DataFrame(
        {
            "term": model.result.params.index,
            "coefficient": model.result.params.values,
            "std_error": model.result.bse.values,
            "t_value": model.result.tvalues.values,
            "p_value": model.result.pvalues.values,
        }
    )
    table["absolute_t"] = table["t_value"].abs()
    return table.sort_values("absolute_t", ascending=False).reset_index(drop=True)


def model_metrics(model: DoeModel) -> dict[str, float]:
    """Return key goodness-of-fit metrics for reporting."""
    return {
        "runs_used": float(model.result.nobs),
        "r2": float(model.result.rsquared),
        "adjusted_r2": float(model.result.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(model.result.resid**2))),
        "aic": float(model.result.aic),
        "bic": float(model.result.bic),
    }


def predict_response(model: DoeModel, factor_values: dict[str, float]) -> float:
    """Predict a response from factor values in original engineering units."""
    row: dict[str, float] = {}
    for factor in model.factors:
        value = factor_values[factor]
        info = model.coding[factor]
        row[factor] = value
        row[coded_column_name(factor)] = (value - info["center"]) / info["half_range"]
    return float(model.result.predict(pd.DataFrame([row])).iloc[0])


def optimize_response(
    model: DoeModel,
    goal: Literal["maximize", "minimize", "target"] = "maximize",
    target: float | None = None,
) -> tuple[dict[str, float], float]:
    """Find factor settings that optimize the fitted response surface."""
    if goal == "target" and target is None:
        raise ValueError("A target value is required when goal='target'.")

    def objective(coded_vector: np.ndarray) -> float:
        factor_values = {
            factor: decode_factor_value(coded_vector[idx], factor, model.coding)
            for idx, factor in enumerate(model.factors)
        }
        prediction = predict_response(model, factor_values)
        if goal == "maximize":
            return -prediction
        if goal == "minimize":
            return prediction
        return (prediction - float(target)) ** 2

    result = minimize(
        objective,
        x0=np.zeros(len(model.factors)),
        bounds=[(-1.0, 1.0)] * len(model.factors),
        method="L-BFGS-B",
    )
    best_values = {
        factor: decode_factor_value(result.x[idx], factor, model.coding)
        for idx, factor in enumerate(model.factors)
    }
    return best_values, predict_response(model, best_values)


def response_surface_grid(
    model: DoeModel,
    x_factor: str,
    y_factor: str,
    fixed_values: dict[str, float] | None = None,
    points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a two-factor response-surface grid for plotting."""
    fixed_values = fixed_values or {
        factor: model.coding[factor]["center"] for factor in model.factors
    }
    x_values = np.linspace(model.coding[x_factor]["low"], model.coding[x_factor]["high"], points)
    y_values = np.linspace(model.coding[y_factor]["low"], model.coding[y_factor]["high"], points)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_mesh = np.zeros_like(x_mesh, dtype=float)

    for row_idx in range(points):
        for col_idx in range(points):
            values = fixed_values.copy()
            values[x_factor] = float(x_mesh[row_idx, col_idx])
            values[y_factor] = float(y_mesh[row_idx, col_idx])
            z_mesh[row_idx, col_idx] = predict_response(model, values)

    return x_mesh, y_mesh, z_mesh


def run_pca(df: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray | list[str]]:
    """Run PCA on selected numeric columns using NumPy only."""
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] < 2 or numeric.empty:
        raise ValueError("PCA requires at least two numeric columns with complete rows.")

    values = numeric.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    scaled = (values - means) / stds
    covariance = np.cov(scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / eigenvalues.sum()
    scores = scaled @ eigenvectors
    loadings = eigenvectors * np.sqrt(eigenvalues)

    return {
        "columns": list(numeric.columns),
        "index": numeric.index.to_numpy(),
        "scores": scores,
        "loadings": loadings,
        "explained_variance": explained,
    }
