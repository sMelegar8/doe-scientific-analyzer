import re
import hashlib
import io
import time
from itertools import combinations, product
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.stats.anova import anova_lm


st.set_page_config(page_title="DoE Scientific Analyzer Pro", layout="wide")

PLOT_SIZE_STANDARD = (7.2, 5.4)
SURFACE_GRID_POINTS = 28


class TobitModel(GenericLikelihoodModel):
    def __init__(self, endog, exog, censor_left, censor_right, **kwds):
        super().__init__(endog, exog, **kwds)
        self.censor_left = np.asarray(censor_left, dtype=bool).flatten()
        self.censor_right = np.asarray(censor_right, dtype=bool).flatten()
        self.log_sigma_idx = exog.shape[1]

    def nloglikeobs(self, params):
        beta = params[: self.log_sigma_idx]
        sigma = np.exp(params[self.log_sigma_idx])
        sigma = max(float(sigma), 1e-10)
        xb = np.dot(self.exog, beta)
        z = (self.endog - xb) / sigma
        llf = np.zeros_like(self.endog, dtype=float)
        uncensored = ~(self.censor_left | self.censor_right)
        llf = np.where(
            uncensored,
            -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * z**2,
            llf,
        )
        llf = np.where(self.censor_left, stats.norm.logcdf(z), llf)
        llf = np.where(self.censor_right, stats.norm.logsf(z), llf)
        return -llf


def write_log(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {message}")
    log_box.code("\n".join(st.session_state.log_messages[-12:]))


@st.cache_data(show_spinner=False)
def load_dataframe_from_upload(file_bytes, filename):
    buffer = io.BytesIO(file_bytes)
    df = (
        pd.read_csv(buffer, sep=None, engine="python")
        if filename.lower().endswith(".csv")
        else pd.read_excel(buffer)
    )
    df.columns = df.columns.astype(str).str.strip()
    return df


def dict_signature(data):
    if not data:
        return "none"
    items = []
    for key in sorted(data):
        items.append(f"{key}:{data[key]['center']}:{data[key]['half_range']}")
    return "|".join(items)


def get_session_model_bundle(cache_key, df_coded, response_name, formula, block_cols, safe_factor_names):
    cache = st.session_state.setdefault("model_bundle_cache", {})
    if cache_key not in cache:
        cache[cache_key] = fit_model_bundle(
            df_coded,
            response_name,
            formula,
            block_cols=block_cols,
            safe_factor_names=safe_factor_names,
        )
    return cache[cache_key]


def clean_text_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )


def parse_response(series):
    raw = clean_text_series(series)
    censor_left = raw.str.startswith("<")
    censor_right = raw.str.startswith(">")
    numeric = pd.to_numeric(
        raw.str.replace("<", "", regex=False).str.replace(">", "", regex=False),
        errors="coerce",
    )
    return raw, numeric, censor_left, censor_right


def make_safe_mapping(columns):
    safe_map = {col: f"V{i + 1}" for i, col in enumerate(columns)}
    inverse_map = {safe: original for original, safe in safe_map.items()}
    return safe_map, inverse_map


def code_factors(df, x_cols, safe_map, custom_coding=None):
    coded_df = df.copy()
    coding_info = {}
    custom_coding = custom_coding or {}
    for original in x_cols:
        numeric_col = pd.to_numeric(df[original], errors="coerce")
        low = float(numeric_col.min())
        high = float(numeric_col.max())
        if original in custom_coding:
            center = float(custom_coding[original]["center"])
            half_range = float(custom_coding[original]["half_range"])
        else:
            center = (low + high) / 2
            half_range = (high - low) / 2
        coding_info[original] = {
            "low": low,
            "high": high,
            "center": center,
            "half_range": half_range if half_range != 0 else 1.0,
        }
        if high == low:
            coded_df[safe_map[original]] = 0.0
        else:
            coded_df[safe_map[original]] = (numeric_col - center) / half_range
    return coded_df, coding_info


def build_formula(v_names, model_type, block_terms=None):
    block_terms = block_terms or []
    base_terms = " + ".join(v_names) if v_names else "1"
    interactions = " + ".join(
        f"{v_names[i]}:{v_names[j]}"
        for i in range(len(v_names))
        for j in range(i + 1, len(v_names))
    )
    powers = " + ".join(f"I({v}**2)" for v in v_names)

    if model_type == "Standard (Linear)":
        extras = []
    elif model_type == "Interactions":
        extras = [interactions] if interactions else []
    elif model_type == "Quadratic":
        extras = [powers] if powers else []
    else:
        extras = [term for term in [interactions, powers] if term]

    parts = ["1"] + block_terms + ([base_terms] if base_terms else []) + extras
    return " + ".join(part for part in parts if part)


def humanize_term(term, inverse_map):
    clean = term
    for safe, original in inverse_map.items():
        clean = clean.replace(f"I({safe} ** 2)", f"{original}^2")
        clean = clean.replace(f"I({safe}**2)", f"{original}^2")
        clean = clean.replace(safe, original)
    return clean.replace(":", " * ")


def decode_factor(coded_value, factor_name, coding_info):
    info = coding_info[factor_name]
    return coded_value * info["half_range"] + info["center"]


def encode_factor(real_value, factor_name, coding_info):
    info = coding_info[factor_name]
    if info["half_range"] == 0:
        return 0.0
    return (real_value - info["center"]) / info["half_range"]


def predict_from_coded_rows(model_bundle, coded_df):
    design_input = coded_df.copy()
    for col, default_value in model_bundle.get("prediction_defaults", {}).items():
        if col not in design_input.columns:
            design_input[col] = default_value
    design = build_design_matrices(
        [model_bundle["design_info"]],
        design_input,
        return_type="dataframe",
    )[0]
    beta = model_bundle["params"][: len(design.columns)]
    preds = np.dot(design.values, beta)
    return preds, design


def fit_model_bundle(df_coded, response_name, formula, block_cols=None, safe_factor_names=None):
    block_cols = block_cols or []
    safe_factor_names = safe_factor_names or []
    raw, y_num, cl, cr = parse_response(df_coded[response_name])
    analysis_mask = y_num.notna()
    work_df = df_coded.loc[analysis_mask].copy()
    work_df["__y__"] = y_num.loc[analysis_mask].astype(float).values
    work_cl = cl.loc[analysis_mask].values
    work_cr = cr.loc[analysis_mask].values

    design = dmatrix(formula, work_df, return_type="dataframe")
    dof = len(work_df) - len(design.columns)
    if dof < 0:
        raise ValueError(
            f"Model too complex for the available data: {len(work_df)} runs and {len(design.columns)} parameters."
        )

    ols_formula_res = smf.ols(formula="__y__ ~ " + formula, data=work_df).fit()
    ols_res = sm.OLS(work_df["__y__"].values, design.values).fit()

    has_censoring = bool(work_cl.any() or work_cr.any())
    model_kind = "OLS"
    active_res = ols_res
    tob_res = None

    if has_censoring:
        model_kind = "Tobit"
        tob_model = TobitModel(work_df["__y__"].values, design.values, work_cl, work_cr)
        start_params = np.append(ols_res.params, np.log(max(np.std(ols_res.resid), 1e-6)))
        tob_res = tob_model.fit(start_params=start_params, disp=0, maxiter=300)
        active_res = tob_res

    params = active_res.params[: len(design.columns)]
    std_errors = active_res.bse[: len(design.columns)]
    if np.isnan(std_errors).any() or np.all(std_errors == 0):
        std_errors = ols_res.bse

    stat_values = np.divide(
        params,
        std_errors,
        out=np.zeros_like(params, dtype=float),
        where=std_errors != 0,
    )
    if model_kind == "OLS":
        p_values = 2 * stats.t.sf(np.abs(stat_values), df=max(dof, 1))
    else:
        p_values = 2 * stats.norm.sf(np.abs(stat_values))

    fitted = np.dot(design.values, params)
    residuals = work_df["__y__"].values - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    sst = float(np.sum((work_df["__y__"].values - work_df["__y__"].mean()) ** 2))
    sse = float(np.sum(residuals**2))
    pseudo_r2 = np.nan if sst == 0 else max(0.0, 1 - sse / sst)
    prediction_defaults = {}
    for col in block_cols:
        if col in work_df.columns:
            modes = work_df[col].mode(dropna=True)
            if not modes.empty:
                prediction_defaults[col] = modes.iloc[0]
            else:
                prediction_defaults[col] = work_df[col].iloc[0]

    return {
        "analysis_df": work_df,
        "design_df": design,
        "design_info": design.design_info,
        "response_name": response_name,
        "formula": formula,
        "dof": dof,
        "model_kind": model_kind,
        "ols_formula_res": ols_formula_res,
        "ols_res": ols_res,
        "tobit_res": tob_res,
        "active_res": active_res,
        "params": params,
        "std_errors": std_errors,
        "stat_values": stat_values,
        "p_values": p_values,
        "fitted": fitted,
        "residuals": residuals,
        "rmse": rmse,
        "pseudo_r2": pseudo_r2,
        "has_censoring": has_censoring,
        "raw_response": raw.loc[analysis_mask],
        "block_cols": list(block_cols),
        "safe_factor_names": list(safe_factor_names),
        "prediction_defaults": prediction_defaults,
    }


def coefficient_table(model_bundle, inverse_map):
    design_columns = model_bundle["design_df"].columns
    coef_df = pd.DataFrame(
        {
            "Term": [humanize_term(col, inverse_map) for col in design_columns],
            "Coefficient": model_bundle["params"],
            "Std Error": model_bundle["std_errors"],
            "Statistic": model_bundle["stat_values"],
            "p-value": model_bundle["p_values"],
        }
    )
    coef_df["Abs Statistic"] = coef_df["Statistic"].abs()
    return coef_df


def model_summary_metrics(model_bundle):
    metrics = {
        "Runs used": len(model_bundle["analysis_df"]),
        "Parameters": len(model_bundle["design_df"].columns),
        "Degrees of freedom": model_bundle["dof"],
        "RMSE": model_bundle["rmse"],
        "Pseudo R2": model_bundle["pseudo_r2"],
    }
    if model_bundle["model_kind"] == "OLS":
        metrics["R2"] = model_bundle["ols_res"].rsquared
        metrics["Adj R2"] = model_bundle["ols_res"].rsquared_adj
        metrics["AIC"] = model_bundle["ols_res"].aic
        metrics["BIC"] = model_bundle["ols_res"].bic
    else:
        metrics["LogLik"] = model_bundle["tobit_res"].llf
        metrics["AIC"] = model_bundle["tobit_res"].aic
        metrics["BIC"] = model_bundle["tobit_res"].bic
    return metrics


def plot_pareto(coef_df, dof, alpha):
    plot_df = coef_df.iloc[1:].sort_values("Abs Statistic")
    threshold = stats.t.ppf(1 - alpha / 2, df=max(dof, 1))
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    colors = ["#1b9e77" if value >= threshold else "#d95f02" for value in plot_df["Abs Statistic"]]
    ax.barh(plot_df["Term"], plot_df["Abs Statistic"], color=colors, edgecolor="black")
    ax.axvline(threshold, color="#1f77b4", linestyle="--", linewidth=2)
    ax.set_xlabel("Absolute t / z statistic")
    ax.set_title("Pareto Chart of Standardized Effects")
    fig.tight_layout()
    return fig, threshold


def plot_coefficients(coef_df):
    plot_df = coef_df.iloc[1:].sort_values("Coefficient", key=np.abs)
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    colors = ["#2e8b57" if val >= 0 else "#b22222" for val in plot_df["Coefficient"]]
    ax.barh(
        plot_df["Term"],
        plot_df["Coefficient"],
        xerr=plot_df["Std Error"],
        color=colors,
        capsize=4,
        edgecolor="black",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_title("Coefficient Plot with Uncertainty")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(model_bundle):
    observed = model_bundle["analysis_df"]["__y__"].values
    predicted = model_bundle["fitted"]
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    ax.scatter(predicted, observed, color="#1f77b4", edgecolor="black", alpha=0.8)
    line_min = min(np.min(predicted), np.min(observed))
    line_max = max(np.max(predicted), np.max(observed))
    ax.plot([line_min, line_max], [line_min, line_max], color="black", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Observed vs Predicted")
    fig.tight_layout()
    return fig


def residual_diagnostic_data(model_bundle):
    fitted = model_bundle["fitted"]
    residuals = model_bundle["residuals"]
    std_res = residuals / max(np.std(residuals, ddof=1), 1e-10)
    run_id = np.arange(1, len(std_res) + 1)
    return fitted, residuals, std_res, run_id


def plot_residuals_vs_predicted(model_bundle):
    fitted, residuals, _, _ = residual_diagnostic_data(model_bundle)
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    ax.scatter(fitted, residuals, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    return fig


def plot_normal_qq(model_bundle):
    _, _, std_res, _ = residual_diagnostic_data(model_bundle)
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    stats.probplot(std_res, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q")
    fig.tight_layout()
    return fig


def plot_residual_histogram(model_bundle):
    _, _, std_res, _ = residual_diagnostic_data(model_bundle)
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    ax.hist(
        std_res,
        bins=min(12, max(5, len(std_res) // 2)),
        color="#8da0cb",
        edgecolor="black",
    )
    ax.set_title("Histogram of Standardized Residuals")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_residuals_vs_run_order(model_bundle):
    _, _, std_res, run_id = residual_diagnostic_data(model_bundle)
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    ax.plot(run_id, std_res, marker="o", color="#66a61e")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Run order")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Residuals vs Run Order")
    fig.tight_layout()
    return fig


def plot_influence(model_bundle):
    if model_bundle["model_kind"] != "OLS":
        return None
    influence = model_bundle["ols_res"].get_influence()
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    stud_resid = influence.resid_studentized_internal
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    ax.scatter(
        leverage,
        stud_resid,
        s=300 * np.sqrt(np.clip(cooks, 0, None) + 1e-6),
        alpha=0.65,
        color="#e7298a",
        edgecolor="black",
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentized residual")
    ax.set_title("Influence Plot (bubble size = Cook's D)")
    fig.tight_layout()
    return fig


def anova_table(model_bundle):
    anova_df = anova_lm(model_bundle["ols_formula_res"], typ=2)
    return anova_df.reset_index().rename(columns={"index": "Term"})


def effect_equation(model_bundle, inverse_map):
    pieces = []
    for idx, term in enumerate(model_bundle["design_df"].columns):
        coef = model_bundle["params"][idx]
        label = humanize_term(term, inverse_map)
        if idx == 0:
            pieces.append(f"{coef:.5g}")
        else:
            sign = "+" if coef >= 0 else "-"
            pieces.append(f" {sign} {abs(coef):.5g} * {label}")
    return "".join(pieces)


def pure_error_lack_of_fit(model_bundle, original_df, x_cols, block_cols=None):
    block_cols = block_cols or []
    work = model_bundle["analysis_df"].copy()
    grouping_cols = x_cols + block_cols
    original_subset = original_df.loc[work.index, grouping_cols].copy()
    for col in x_cols:
        original_subset[col] = pd.to_numeric(original_subset[col], errors="coerce")
    response = work["__y__"].astype(float)
    grouped = pd.concat([original_subset, response.rename("__y__")], axis=1).groupby(grouping_cols, dropna=False)

    ss_pe = 0.0
    df_pe = 0
    for _, group in grouped:
        n = len(group)
        if n > 1:
            mean_y = group["__y__"].mean()
            ss_pe += float(((group["__y__"] - mean_y) ** 2).sum())
            df_pe += n - 1

    sse = float(np.sum(model_bundle["residuals"] ** 2))
    df_res = max(int(model_bundle["dof"]), 0)
    ss_lof = sse - ss_pe
    df_lof = df_res - df_pe

    if df_pe <= 0 or df_lof <= 0 or ss_pe < 0:
        return None

    ms_pe = ss_pe / df_pe if df_pe else np.nan
    ms_lof = ss_lof / df_lof if df_lof else np.nan
    f_value = ms_lof / ms_pe if ms_pe and ms_pe > 0 else np.nan
    p_value = stats.f.sf(f_value, df_lof, df_pe) if np.isfinite(f_value) else np.nan

    return {
        "Pure Error SS": ss_pe,
        "Pure Error DF": df_pe,
        "Pure Error MS": ms_pe,
        "Lack-of-Fit SS": ss_lof,
        "Lack-of-Fit DF": df_lof,
        "Lack-of-Fit MS": ms_lof,
        "F": f_value,
        "p-value": p_value,
        "Replicate groups": int(sum(1 for _, group in grouped if len(group) > 1)),
    }


def quadratic_response_matrix(model_bundle, safe_map):
    factors = list(safe_map.keys())
    safe_to_idx = {safe_map[factor]: i for i, factor in enumerate(factors)}
    linear = np.zeros(len(factors))
    quad = np.zeros((len(factors), len(factors)))

    for term, coef in zip(model_bundle["design_df"].columns, model_bundle["params"]):
        if term == "Intercept":
            continue
        if term in safe_to_idx:
            linear[safe_to_idx[term]] = coef
        elif term.startswith("I(") and term.endswith(")") and "**2" in term.replace(" ", ""):
            safe = term[2:-1].replace(" ", "").replace("**2", "")
            if safe in safe_to_idx:
                quad[safe_to_idx[safe], safe_to_idx[safe]] += coef
        elif ":" in term:
            left, right = term.split(":")
            if left in safe_to_idx and right in safe_to_idx:
                i = safe_to_idx[left]
                j = safe_to_idx[right]
                quad[i, j] += coef / 2
                quad[j, i] += coef / 2

    return factors, linear, quad


def canonical_analysis(model_bundle, safe_map, coding_info):
    factors, linear, quad = quadratic_response_matrix(model_bundle, safe_map)
    if not np.any(np.abs(quad) > 1e-12):
        return None
    try:
        stationary = -0.5 * np.linalg.solve(quad, linear)
    except np.linalg.LinAlgError:
        return None

    eigvals, eigvecs = np.linalg.eigh(quad)
    stationary_real = {
        factor: decode_factor(stationary[i], factor, coding_info)
        for i, factor in enumerate(factors)
    }
    pred_df = pd.DataFrame({safe_map[f]: [stationary[i]] for i, f in enumerate(factors)})
    pred, _ = predict_from_coded_rows(model_bundle, pred_df)

    if np.all(eigvals < 0):
        nature = "Maximum"
    elif np.all(eigvals > 0):
        nature = "Minimum"
    else:
        nature = "Saddle"

    return {
        "factors": factors,
        "stationary_coded": stationary,
        "stationary_real": stationary_real,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "predicted_response": float(pred[0]),
        "nature": nature,
    }


def steepest_ascent_table(model_bundle, safe_map, coding_info, steps=6):
    factors, linear, _ = quadratic_response_matrix(model_bundle, safe_map)
    direction = linear.copy()
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    direction = direction / norm
    levels = np.linspace(0, 1.25, steps)
    rows = []
    for level in levels:
        coded_point = direction * level
        pred_df = pd.DataFrame({safe_map[f]: [coded_point[i]] for i, f in enumerate(factors)})
        pred, _ = predict_from_coded_rows(model_bundle, pred_df)
        row = {
            "Step size (coded)": level,
            "Predicted response": float(pred[0]),
        }
        for i, factor in enumerate(factors):
            row[f"{factor} coded"] = coded_point[i]
            row[f"{factor} real"] = decode_factor(coded_point[i], factor, coding_info)
        rows.append(row)
    return pd.DataFrame(rows)


def desirability_function(y, low, target, high, goal):
    if goal == "Maximize":
        if y <= low:
            return 0.0
        if y >= high:
            return 1.0
        return ((y - low) / (high - low)) if high > low else 0.0
    if goal == "Minimize":
        if y >= high:
            return 0.0
        if y <= low:
            return 1.0
        return ((high - y) / (high - low)) if high > low else 0.0
    if y <= low or y >= high:
        return 0.0
    if y == target:
        return 1.0
    if y < target:
        return (y - low) / (target - low) if target > low else 0.0
    return (high - y) / (high - target) if high > target else 0.0


def make_surface_grid(model_bundle, safe_map, coding_info, x_factor, y_factor, fixed_real_values, points=SURFACE_GRID_POINTS):
    x_safe = safe_map[x_factor]
    y_safe = safe_map[y_factor]
    x_coded = np.linspace(-1, 1, points)
    y_coded = np.linspace(-1, 1, points)

    base = {
        safe_map[factor]: encode_factor(fixed_real_values[factor], factor, coding_info)
        for factor in safe_map
    }
    rows = []
    for yc in y_coded:
        for xc in x_coded:
            row = base.copy()
            row[x_safe] = xc
            row[y_safe] = yc
            rows.append(row)

    pred_df = pd.DataFrame(rows)
    preds, _ = predict_from_coded_rows(model_bundle, pred_df)
    z_mesh = preds.reshape(len(y_coded), len(x_coded))

    x_real = decode_factor(x_coded, x_factor, coding_info)
    y_real = decode_factor(y_coded, y_factor, coding_info)
    x_real_mesh, y_real_mesh = np.meshgrid(x_real, y_real)
    return x_real_mesh, y_real_mesh, z_mesh


def optimize_response(model_bundle, safe_map, coding_info, goal, target_value=None):
    factor_names = list(safe_map.keys())

    def objective(coded_vector):
        row = {safe_map[factor]: [coded_vector[i]] for i, factor in enumerate(factor_names)}
        pred, _ = predict_from_coded_rows(model_bundle, pd.DataFrame(row))
        pred_value = float(pred[0])
        if goal == "Maximize":
            return -pred_value
        if goal == "Minimize":
            return pred_value
        return (pred_value - float(target_value)) ** 2

    x0 = np.zeros(len(factor_names))
    bounds = [(-1, 1)] * len(factor_names)
    result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    best_coded = result.x
    best_real = {
        factor: decode_factor(best_coded[i], factor, coding_info)
        for i, factor in enumerate(factor_names)
    }
    pred, _ = predict_from_coded_rows(
        model_bundle,
        pd.DataFrame({safe_map[factor]: [best_coded[i]] for i, factor in enumerate(factor_names)}),
    )
    return best_real, float(pred[0]), result


@st.cache_data(show_spinner=False)
def compute_pca(dataframe):
    clean_df = dataframe.dropna(axis=0).copy()
    if clean_df.empty or clean_df.shape[1] < 2:
        return None

    x = clean_df.values.astype(float)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    x_scaled = (x - means) / stds
    cov = np.cov(x_scaled, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained = eigvals / eigvals.sum()
    scores = x_scaled @ eigvecs
    loadings = eigvecs * np.sqrt(eigvals)
    return {
        "rows": clean_df.index,
        "columns": clean_df.columns,
        "scores": scores,
        "loadings": loadings,
        "explained": explained,
    }


def plot_pca_scores(pca_result):
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    scores = pca_result["scores"]
    ax.scatter(scores[:, 0], scores[:, 1], color="#1f78b4", edgecolor="black", alpha=0.8)
    for idx, row_id in enumerate(pca_result["rows"]):
        ax.annotate(str(row_id), (scores[idx, 0], scores[idx, 1]), fontsize=8, alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca_result['explained'][0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_result['explained'][1] * 100:.1f}%)")
    ax.set_title("PCA Score Plot")
    fig.tight_layout()
    return fig


def plot_pca_loadings(pca_result):
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    load_df = pd.DataFrame(
        {
            "Variable": pca_result["columns"],
            "PC1": pca_result["loadings"][:, 0],
            "PC2": pca_result["loadings"][:, 1],
        }
    ).sort_values("PC1", key=np.abs)
    ax.barh(load_df["Variable"], load_df["PC1"], color="#33a02c", alpha=0.8, label="PC1")
    ax.barh(load_df["Variable"], load_df["PC2"], color="#fb9a99", alpha=0.55, label="PC2")
    ax.set_title("PCA Loadings")
    ax.legend()
    fig.tight_layout()
    return fig


def recommended_pca_components(explained, threshold=0.80):
    cumulative = np.cumsum(explained)
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    return min(idx + 1, len(explained))


def pca_loading_table(pca_result, component_index=0):
    comp = component_index + 1
    load_df = pd.DataFrame(
        {
            "Variable": pca_result["columns"],
            f"PC{comp} loading": pca_result["loadings"][:, component_index],
        }
    )
    load_df["Absolute loading"] = load_df[f"PC{comp} loading"].abs()
    return load_df.sort_values("Absolute loading", ascending=False).reset_index(drop=True)


def pca_interpretation_summary(pca_result, pca_source, x_cols, y_cols):
    explained = pca_result["explained"]
    cumulative = np.cumsum(explained)
    retain_80 = recommended_pca_components(explained, threshold=0.80)
    retain_85 = recommended_pca_components(explained, threshold=0.85)
    total_vars = len(pca_result["columns"])

    summary = {
        "PC1 variance %": float(explained[0] * 100),
        "PC2 variance %": float(explained[1] * 100) if len(explained) > 1 else np.nan,
        "PC1+PC2 cumulative %": float(cumulative[min(1, len(cumulative) - 1)] * 100),
        "Components for 80%": int(retain_80),
        "Components for 85%": int(retain_85),
        "Total PCA variables": int(total_vars),
    }

    insights = []
    if len(explained) >= 2:
        insights.append(
            f"PC1 and PC2 together explain {cumulative[1] * 100:.2f}% of the available variance, so a 2D score plot already captures a large share of the experiment."
        )
    insights.append(
        f"The 80% rule suggests keeping about {retain_80} principal components; an 85% retention target suggests about {retain_85}."
    )
    if total_vars >= 5 and cumulative[min(3, len(cumulative) - 1)] >= 0.80:
        insights.append("The tail components contribute relatively little and are likely to contain mostly noise, redundancy, or very local structure.")
    if pca_source == "Factors + responses":
        insights.append(
            "Be careful: PCA on factors plus responses is exploratory, but for a designed experiment the factors are already structured by design. In many DoE studies PCA is more informative on the response block alone."
        )
    if pca_source == "Responses only" and len(y_cols) >= 4:
        insights.append(
            "This is the most typical high-value DoE use case: PCA compresses many correlated responses into a smaller set of latent quality dimensions."
        )
    return summary, insights


def plot_pca_biplot(pca_result, color_values=None, color_label=None):
    fig, ax = plt.subplots(figsize=PLOT_SIZE_STANDARD)
    scores = pca_result["scores"][:, :2]
    loadings = pca_result["loadings"][:, :2]

    if color_values is None:
        ax.scatter(scores[:, 0], scores[:, 1], color="#1f78b4", edgecolor="black", alpha=0.8)
    else:
        color_series = pd.Series(color_values).reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(color_series):
            scatter = ax.scatter(scores[:, 0], scores[:, 1], c=color_series.values, cmap="viridis", edgecolor="black", alpha=0.85)
            fig.colorbar(scatter, ax=ax, label=color_label or "Value")
        else:
            categories = color_series.astype(str).fillna("NA")
            palette = plt.cm.Set2(np.linspace(0, 1, max(3, categories.nunique())))
            for idx, category in enumerate(categories.unique()):
                mask = categories == category
                ax.scatter(
                    scores[mask.values, 0],
                    scores[mask.values, 1],
                    color=palette[idx],
                    edgecolor="black",
                    alpha=0.85,
                    label=str(category),
                )
            ax.legend(title=color_label or "Group", loc="best", fontsize=8)

    for idx, row_id in enumerate(pca_result["rows"]):
        ax.annotate(str(row_id), (scores[idx, 0], scores[idx, 1]), fontsize=8, alpha=0.65)

    score_scale = max(np.max(np.abs(scores[:, 0])), np.max(np.abs(scores[:, 1])), 1.0)
    loading_scale = score_scale * 0.85
    for idx, variable in enumerate(pca_result["columns"]):
        x_end = loadings[idx, 0] * loading_scale
        y_end = loadings[idx, 1] * loading_scale
        ax.arrow(0, 0, x_end, y_end, color="#b22222", width=0.0, head_width=0.08 * score_scale, length_includes_head=True, alpha=0.8)
        ax.text(x_end * 1.07, y_end * 1.07, str(variable), color="#7a0019", fontsize=8)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca_result['explained'][0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_result['explained'][1] * 100:.1f}%)")
    ax.set_title("PCA Biplot")
    fig.tight_layout()
    return fig


def overview_table(df, x_cols, y_cols, block_cols=None):
    block_cols = block_cols or []
    rows = []
    for col in x_cols + block_cols + y_cols:
        numeric = pd.to_numeric(
            clean_text_series(df[col]).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
            errors="coerce",
        )
        if col in x_cols:
            role = "Factor"
        elif col in block_cols:
            role = "Block"
        else:
            role = "Response"
        rows.append(
            {
                "Column": col,
                "Role": role,
                "Non-null": int(df[col].notna().sum()),
                "Missing": int(df[col].isna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
                "Min": numeric.min(),
                "Mean": numeric.mean(),
                "Max": numeric.max(),
            }
        )
    return pd.DataFrame(rows)


def render_glossary(title, rows):
    with st.expander(title, expanded=False):
        st.table(pd.DataFrame(rows, columns=["Parameter", "Meaning"]))


def plot_correlation_heatmap(df, x_cols, response_name):
    corr_inputs = []
    for col in x_cols + [response_name]:
        corr_inputs.append(
            pd.to_numeric(
                clean_text_series(df[col]).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                errors="coerce",
            )
        )
    corr_df = pd.concat(corr_inputs, axis=1)
    corr_df.columns = x_cols + [response_name]
    corr = corr_df.corr(numeric_only=True)
    if corr.shape[0] < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def save_figure_to_path(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_analysis_export_package(
    df,
    x_cols,
    block_cols,
    y_cols,
    selected_y,
    alpha,
    safe_map,
    inverse_map,
    coding_info,
    model_bundle,
    coef_df,
    summary_metrics,
    anova_df,
    lof_result,
    canonical_result,
    steepest_df,
    response_models,
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_base = Path.cwd() / "doe_analysis_exports"
    export_base.mkdir(parents=True, exist_ok=True)
    root = export_base / f"analysis_{selected_y}_{timestamp}"
    suffix = 1
    while root.exists():
        suffix += 1
        root = export_base / f"analysis_{selected_y}_{timestamp}_{suffix}"
    root.mkdir(parents=True, exist_ok=True)

    folders = {
        "overview": root / "01_overview",
        "model": root / "02_model",
        "effects": root / "03_effects",
        "anova": root / "04_anova",
        "diagnostics": root / "05_diagnostics",
        "surfaces": root / "06_surfaces",
        "optimizer": root / "07_optimizer",
        "multi": root / "08_multi_response",
        "pca": root / "09_pca",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    workbook_path = root / f"analysis_{selected_y}.xlsx"
    pca_numeric = pd.DataFrame(
        {
            col: pd.to_numeric(
                clean_text_series(df[col]).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                errors="coerce",
            )
            for col in y_cols
        }
    )
    pca_result = compute_pca(pca_numeric)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_Data", index=False)
        overview_table(df, x_cols, y_cols, block_cols).to_excel(writer, sheet_name="Overview", index=False)
        pd.DataFrame(list(summary_metrics.items()), columns=["Metric", "Value"]).to_excel(writer, sheet_name="Model_Summary", index=False)
        coef_df.drop(columns=["Abs Statistic"], errors="ignore").to_excel(writer, sheet_name="Coefficients", index=False)
        anova_df.to_excel(writer, sheet_name="ANOVA", index=False)
        if lof_result is not None:
            pd.DataFrame([lof_result]).to_excel(writer, sheet_name="Lack_of_Fit", index=False)
        if canonical_result is not None:
            pd.DataFrame(
                {
                    "Factor": canonical_result["factors"],
                    "Coded stationary value": canonical_result["stationary_coded"],
                    "Real stationary value": [canonical_result["stationary_real"][f] for f in canonical_result["factors"]],
                }
            ).to_excel(writer, sheet_name="Canonical_Point", index=False)
            pd.DataFrame(
                {
                    "Canonical axis": [f"Axis {i + 1}" for i in range(len(canonical_result["eigenvalues"]))],
                    "Eigenvalue": canonical_result["eigenvalues"],
                }
            ).to_excel(writer, sheet_name="Canonical_Eigen", index=False)
        if steepest_df is not None:
            steepest_df.to_excel(writer, sheet_name="Steepest_Path", index=False)
        fitted_df = model_bundle["analysis_df"].copy()
        fitted_df["Predicted"] = model_bundle["fitted"]
        fitted_df["Residual"] = model_bundle["residuals"]
        fitted_df.to_excel(writer, sheet_name="Fitted_and_Residuals", index=False)
        if pca_result is not None:
            pd.DataFrame(
                {
                    "Component": [f"PC{i + 1}" for i in range(len(pca_result["explained"]))],
                    "Explained variance": pca_result["explained"],
                    "Cumulative variance": np.cumsum(pca_result["explained"]),
                }
            ).to_excel(writer, sheet_name="PCA", index=False)

    corr_fig = plot_correlation_heatmap(df, x_cols, selected_y)
    if corr_fig is not None:
        save_figure_to_path(corr_fig, folders["overview"] / "correlation_heatmap.png")

    save_figure_to_path(plot_pareto(coef_df, model_bundle["dof"], alpha)[0], folders["effects"] / "pareto_effects.png")
    save_figure_to_path(plot_coefficients(coef_df), folders["effects"] / "coefficient_plot.png")
    save_figure_to_path(plot_actual_vs_predicted(model_bundle), folders["diagnostics"] / "observed_vs_predicted.png")
    save_figure_to_path(plot_residuals_vs_predicted(model_bundle), folders["diagnostics"] / "residuals_vs_predicted.png")
    save_figure_to_path(plot_normal_qq(model_bundle), folders["diagnostics"] / "normal_qq.png")
    save_figure_to_path(plot_residual_histogram(model_bundle), folders["diagnostics"] / "residual_histogram.png")
    save_figure_to_path(plot_residuals_vs_run_order(model_bundle), folders["diagnostics"] / "residuals_vs_run_order.png")

    influence_fig = plot_influence(model_bundle)
    if influence_fig is not None:
        save_figure_to_path(influence_fig, folders["diagnostics"] / "influence_plot.png")

    if len(x_cols) >= 2:
        fixed_values = {factor: coding_info[factor]["center"] for factor in x_cols}
        xr, yr, zm = make_surface_grid(model_bundle, safe_map, coding_info, x_cols[0], x_cols[1], fixed_values)

        fig3d = plt.figure(figsize=PLOT_SIZE_STANDARD)
        ax3d = fig3d.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(xr, yr, zm, cmap="viridis", edgecolor="none", alpha=0.9)
        ax3d.set_xlabel(x_cols[0])
        ax3d.set_ylabel(x_cols[1])
        ax3d.set_zlabel(selected_y)
        ax3d.set_title("Predicted response surface")
        fig3d.colorbar(surf, shrink=0.6, pad=0.1)
        fig3d.tight_layout()
        save_figure_to_path(fig3d, folders["surfaces"] / "surface_3d.png")

        fig2d, ax2d = plt.subplots(figsize=PLOT_SIZE_STANDARD)
        contour = ax2d.contourf(xr, yr, zm, levels=18, cmap="viridis")
        lines = ax2d.contour(xr, yr, zm, levels=10, colors="black", linewidths=0.5, alpha=0.6)
        ax2d.clabel(lines, inline=True, fontsize=8)
        ax2d.set_xlabel(x_cols[0])
        ax2d.set_ylabel(x_cols[1])
        ax2d.set_title("Contour map")
        fig2d.colorbar(contour, ax=ax2d)
        fig2d.tight_layout()
        save_figure_to_path(fig2d, folders["surfaces"] / "contour_map.png")

    if pca_result is not None:
        save_figure_to_path(plot_pca_scores(pca_result), folders["pca"] / "pca_scores.png")
        save_figure_to_path(plot_pca_loadings(pca_result), folders["pca"] / "pca_loadings.png")

    notes = [
        f"Selected response: {selected_y}",
        f"Model type: {model_bundle['formula']}",
        f"Engine: {model_bundle['model_kind']}",
        f"Factors: {', '.join(x_cols)}",
        f"Blocks: {', '.join(block_cols) if block_cols else 'None'}",
        f"Responses available: {', '.join(y_cols)}",
    ]
    (root / "README_export.txt").write_text("\n".join(notes), encoding="utf-8")

    archive_path = shutil.make_archive(str(root), "zip", root.parent, root.name)
    zip_path = Path(archive_path)
    return zip_path.read_bytes(), zip_path.name


def factor_columns(n_factors):
    return [f"x{i + 1}" for i in range(n_factors)]


def model_parameter_count(n_factors, model_goal):
    if model_goal == "Main effects":
        return 1 + n_factors
    if model_goal == "Two-factor interactions":
        return 1 + n_factors + (n_factors * (n_factors - 1)) // 2
    return 1 + 2 * n_factors + (n_factors * (n_factors - 1)) // 2


def expected_terms(n_factors, model_goal):
    terms = ["Intercept"] + factor_columns(n_factors)
    if model_goal in {"Two-factor interactions", "Quadratic surface"}:
        terms.extend(
            f"x{i + 1}:x{j + 1}"
            for i in range(n_factors)
            for j in range(i + 1, n_factors)
        )
    if model_goal == "Quadratic surface":
        terms.extend(f"x{i + 1}^2" for i in range(n_factors))
    return terms


def recommended_center_points(n_factors, design_family, model_goal):
    if design_family in {"Box-Behnken", "CCC", "CCI", "CCF"}:
        return 5 if n_factors <= 3 else 6
    if model_goal == "Quadratic surface":
        return 4
    return 3 if n_factors <= 4 else 4


def approximate_fraction_generators(n_factors, reduction):
    base_count = n_factors - reduction
    if base_count < 2:
        raise ValueError("Too many generators requested for the available number of factors.")
    chosen = []
    for degree in range(base_count, 1, -1):
        for subset in combinations(range(base_count), degree):
            if subset not in chosen:
                chosen.append(subset)
            if len(chosen) == reduction:
                return chosen
    raise ValueError("Could not build a suitable regular fractional factorial generator set.")


def finalize_design_table(design_df, design_family, metadata, whole_design_replicates=1):
    base = design_df.copy()
    chunks = []
    for rep in range(whole_design_replicates):
        chunk = base.copy()
        chunk.insert(0, "Whole-design replicate", rep + 1)
        chunks.append(chunk)
    final_df = pd.concat(chunks, ignore_index=True)
    final_df.insert(0, "Run", np.arange(1, len(final_df) + 1))
    metadata = metadata.copy()
    metadata["design_family"] = design_family
    metadata["whole_design_replicates"] = int(whole_design_replicates)
    metadata["total_runs"] = int(len(final_df))
    metadata["unique_design_points"] = int(len(base))
    return final_df, metadata


def build_full_factorial(n_factors, levels=2, center_points=0, whole_design_replicates=1):
    if levels == 2:
        coded_levels = [-1.0, 1.0]
        family = "2-level Full Factorial"
    else:
        coded_levels = [-1.0, 0.0, 1.0]
        family = "3-level Full Factorial"
    base = pd.DataFrame(list(product(coded_levels, repeat=n_factors)), columns=factor_columns(n_factors))
    if center_points > 0:
        centers = pd.DataFrame(np.zeros((center_points, n_factors)), columns=base.columns)
        base = pd.concat([base, centers], ignore_index=True)
    metadata = {
        "levels": levels,
        "center_points": int(center_points),
        "notes": "Classical full factorial design. Use 2 levels for screening/interactions and 3 levels when you explicitly want quadratic information from a grid.",
    }
    return finalize_design_table(base, family, metadata, whole_design_replicates=whole_design_replicates)


def build_fractional_factorial(n_factors, reduction=1, center_points=0, whole_design_replicates=1):
    if n_factors < 3:
        raise ValueError("Fractional factorial planning is most meaningful for at least 3 factors.")
    if reduction < 1 or reduction >= n_factors:
        raise ValueError("Reduction p must satisfy 1 <= p < number of factors.")

    base_count = n_factors - reduction
    base_columns = factor_columns(base_count)
    rows = []
    generators = approximate_fraction_generators(n_factors, reduction)

    for run in product([-1.0, 1.0], repeat=base_count):
        row = list(run)
        for subset in generators:
            generated = float(np.prod([run[idx] for idx in subset]))
            row.append(generated)
        rows.append(row)

    design_df = pd.DataFrame(rows, columns=factor_columns(n_factors))
    if center_points > 0:
        centers = pd.DataFrame(np.zeros((center_points, n_factors)), columns=design_df.columns)
        design_df = pd.concat([design_df, centers], ignore_index=True)

    generator_labels = []
    for offset, subset in enumerate(generators):
        target = f"x{base_count + offset + 1}"
        generator_labels.append(f"{target} = " + " * ".join(f"x{i + 1}" for i in subset))

    resolution = min(len(subset) + 1 for subset in generators)
    metadata = {
        "reduction": int(reduction),
        "center_points": int(center_points),
        "resolution": int(resolution),
        "generators": generator_labels,
        "notes": "Regular 2-level fractional factorial built from generator relations. Add center points if you want a curvature check and pure-error degrees of freedom.",
    }
    return finalize_design_table(
        design_df,
        f"2^({n_factors}-{reduction}) Fractional Factorial",
        metadata,
        whole_design_replicates=whole_design_replicates,
    )


def build_box_behnken(n_factors, center_points=0, whole_design_replicates=1):
    if n_factors < 3:
        raise ValueError("Box-Behnken requires at least 3 factors.")

    rows = []
    for i, j in combinations(range(n_factors), 2):
        for a, b in product([-1.0, 1.0], repeat=2):
            row = np.zeros(n_factors)
            row[i] = a
            row[j] = b
            rows.append(row)
    design_df = pd.DataFrame(rows, columns=factor_columns(n_factors))
    if center_points > 0:
        centers = pd.DataFrame(np.zeros((center_points, n_factors)), columns=design_df.columns)
        design_df = pd.concat([design_df, centers], ignore_index=True)

    metadata = {
        "center_points": int(center_points),
        "notes": "Box-Behnken avoids all-corner runs, which is often preferred when extreme combinations are risky or expensive.",
    }
    return finalize_design_table(design_df, "Box-Behnken", metadata, whole_design_replicates=whole_design_replicates)


def build_central_composite(n_factors, variant="CCC", alpha=None, center_points=0, whole_design_replicates=1):
    if n_factors < 2:
        raise ValueError("Central composite designs require at least 2 factors.")
    variant = variant.upper()
    alpha_value = float(alpha) if alpha is not None else float(2 ** (n_factors / 4))

    factorial = np.array(list(product([-1.0, 1.0], repeat=n_factors)), dtype=float)
    if variant == "CCI":
        factorial = factorial / alpha_value
    if variant == "CCF":
        alpha_value = 1.0

    star_rows = []
    for idx in range(n_factors):
        for sign in (-1.0, 1.0):
            row = np.zeros(n_factors)
            row[idx] = sign * (1.0 if variant in {"CCI", "CCF"} else alpha_value)
            star_rows.append(row)

    design_df = pd.DataFrame(
        np.vstack([factorial, np.array(star_rows, dtype=float)]),
        columns=factor_columns(n_factors),
    )
    if center_points > 0:
        centers = pd.DataFrame(np.zeros((center_points, n_factors)), columns=design_df.columns)
        design_df = pd.concat([design_df, centers], ignore_index=True)

    metadata = {
        "variant": variant,
        "alpha": float(alpha_value),
        "center_points": int(center_points),
        "notes": "Central composite designs estimate a full quadratic model. CCC is rotatable-friendly, CCF stays within +/-1, and CCI rescales the cube inside the star radius.",
    }
    return finalize_design_table(design_df, variant, metadata, whole_design_replicates=whole_design_replicates)


def design_dof_summary(design_df, factor_cols, model_goal):
    n_runs = int(len(design_df))
    p_model = int(model_parameter_count(len(factor_cols), model_goal))
    grouped = design_df.groupby(factor_cols, dropna=False).size()
    pure_error_df = int((grouped - 1).clip(lower=0).sum())
    total_df = max(n_runs - 1, 0)
    residual_df = n_runs - p_model
    lack_of_fit_df = residual_df - pure_error_df if residual_df >= 0 else np.nan
    return {
        "Runs": n_runs,
        "Model parameters": p_model,
        "Total df": total_df,
        "Residual df": residual_df,
        "Pure error df": pure_error_df,
        "Lack-of-fit df": lack_of_fit_df,
        "Unique design points": int(grouped.shape[0]),
    }


def recommend_designs(n_factors, model_goal):
    suggestions = []
    if model_goal == "Main effects":
        if n_factors <= 5:
            suggestions.append(("2-level Full Factorial", "Best when you want a complete screening design without aliasing."))
        suggestions.append(("2-level Fractional Factorial", "Efficient screening option when many factors make the full cube too expensive."))
        suggestions.append(("CCF", "Useful if you also want a built-in curvature check and future quadratic upgrade path."))
    elif model_goal == "Two-factor interactions":
        if n_factors <= 4:
            suggestions.append(("2-level Full Factorial", "Supports all main effects and two-factor interactions directly."))
        suggestions.append(("2-level Fractional Factorial", "Professional compromise when the full interaction cube is too large."))
        suggestions.append(("CCF", "Convenient when interactions matter and you may extend to a quadratic response surface later."))
    else:
        if n_factors == 2:
            suggestions.append(("CCC", "Canonical response-surface choice for 2 factors, especially when rotatability matters."))
            suggestions.append(("CCF", "Safer bounded alternative when you cannot go beyond +/-1."))
        elif 3 <= n_factors <= 7:
            suggestions.append(("Box-Behnken", "Efficient quadratic design that avoids all-corner extreme combinations."))
            suggestions.append(("CCF", "Quadratic design that stays inside the factor cube."))
            suggestions.append(("CCC", "Best when you want the classic rotatable central-composite family."))
        else:
            suggestions.append(("CCF", "Quadratic design with reasonable run growth for many factors."))
            suggestions.append(("2-level Fractional Factorial", "Use as a first screening stage before a follow-up response-surface design."))
    return pd.DataFrame(suggestions, columns=["Suggested design", "Why it is a good fit"])


def planner_design_from_selection(n_factors, model_goal, design_choice, center_points, whole_design_replicates, fraction_reduction, alpha_value):
    if design_choice == "Auto":
        if model_goal == "Quadratic surface":
            design_choice = "Box-Behnken" if n_factors >= 3 else "CCC"
        elif model_goal == "Main effects" and n_factors >= 6:
            design_choice = "2-level Fractional Factorial"
        else:
            design_choice = "2-level Full Factorial"

    if design_choice == "2-level Full Factorial":
        return build_full_factorial(n_factors, levels=2, center_points=center_points, whole_design_replicates=whole_design_replicates)
    if design_choice == "3-level Full Factorial":
        return build_full_factorial(n_factors, levels=3, center_points=0, whole_design_replicates=whole_design_replicates)
    if design_choice == "2-level Fractional Factorial":
        return build_fractional_factorial(
            n_factors,
            reduction=fraction_reduction,
            center_points=center_points,
            whole_design_replicates=whole_design_replicates,
        )
    if design_choice == "Box-Behnken":
        return build_box_behnken(n_factors, center_points=center_points, whole_design_replicates=whole_design_replicates)
    if design_choice in {"CCC", "CCI", "CCF"}:
        return build_central_composite(
            n_factors,
            variant=design_choice,
            alpha=alpha_value,
            center_points=center_points,
            whole_design_replicates=whole_design_replicates,
        )
    raise ValueError(f"Unsupported design choice: {design_choice}")


def prepare_coded_dataset(df, factor_specs):
    x_cols = list(factor_specs.keys())
    safe_map, inverse_map = make_safe_mapping(x_cols)
    coded_df = df.copy()
    coding_info = {}
    for factor, spec in factor_specs.items():
        if spec["kind"] == "actual":
            numeric_col = pd.to_numeric(df[factor], errors="coerce")
            center = float(spec["center"])
            half_range = float(spec["half_range"])
            coded_values = (numeric_col - center) / half_range
            low = float(numeric_col.min())
            high = float(numeric_col.max())
        else:
            coded_values = pd.to_numeric(df[factor], errors="coerce")
            center = float(spec.get("center", 0.0))
            half_range = float(spec.get("half_range", 1.0))
            low = float(coded_values.min())
            high = float(coded_values.max())
        coded_df[safe_map[factor]] = coded_values
        coding_info[factor] = {
            "low": low,
            "high": high,
            "center": center,
            "half_range": half_range,
        }
    return coded_df, x_cols, safe_map, inverse_map, coding_info


def chemreact_benchmark():
    df = pd.DataFrame(
        [
            [80.00, 170.00, "B1", 80.5],
            [80.00, 180.00, "B1", 81.5],
            [90.00, 170.00, "B1", 82.0],
            [90.00, 180.00, "B1", 83.5],
            [85.00, 175.00, "B1", 83.9],
            [85.00, 175.00, "B1", 84.3],
            [85.00, 175.00, "B1", 84.0],
            [85.00, 175.00, "B2", 79.7],
            [85.00, 175.00, "B2", 79.8],
            [85.00, 175.00, "B2", 79.5],
            [92.07, 175.00, "B2", 78.4],
            [77.93, 175.00, "B2", 75.6],
            [85.00, 182.07, "B2", 78.5],
            [85.00, 167.93, "B2", 77.0],
        ],
        columns=["Time", "Temp", "Block", "Yield"],
    )
    reference = {
        "coefficients": {
            "Intercept": 84.095427,
            "Block[T.B2]": -4.457530,
            "Time": 0.932541,
            "Temp": 0.577712,
            "Time * Temp": 0.125000,
            "Time^2": -1.308555,
            "Temp^2": -0.933442,
        },
        "stationary": {"Time": 86.86148, "Temp": 176.67190},
        "r2": 0.9981,
        "source_label": "rsm JSS article / ChemReact example",
        "source_url": "https://rvlenth.github.io/rsm/articles/article-JSS.html",
        "source_note": "Official rsm article with printed data and published second-order model summary.",
    }
    factor_specs = {
        "Time": {"kind": "actual", "center": 85.0, "half_range": 5.0},
        "Temp": {"kind": "actual", "center": 175.0, "half_range": 5.0},
    }
    formula = "Block + V1 + V2 + V1:V2 + I(V1**2) + I(V2**2)"
    return df, factor_specs, "Yield", formula, reference


def heli_benchmark():
    df = pd.DataFrame(
        [
            [1, -1, -1, -1, -1, 367, 72],
            [1, 1, -1, -1, -1, 369, 72],
            [1, -1, 1, -1, -1, 374, 74],
            [1, 1, 1, -1, -1, 370, 79],
            [1, -1, -1, 1, -1, 372, 72],
            [1, 1, -1, 1, -1, 355, 81],
            [1, -1, 1, 1, -1, 397, 72],
            [1, 1, 1, 1, -1, 377, 99],
            [1, -1, -1, -1, 1, 350, 90],
            [1, 1, -1, -1, 1, 373, 86],
            [1, -1, 1, -1, 1, 358, 92],
            [1, 1, 1, -1, 1, 363, 112],
            [1, -1, -1, 1, 1, 344, 76],
            [1, 1, -1, 1, 1, 355, 69],
            [1, -1, 1, 1, 1, 370, 91],
            [1, 1, 1, 1, 1, 362, 71],
            [1, 0, 0, 0, 0, 377, 51],
            [1, 0, 0, 0, 0, 375, 74],
            [2, -2, 0, 0, 0, 361, 111],
            [2, 2, 0, 0, 0, 364, 93],
            [2, 0, -2, 0, 0, 355, 100],
            [2, 0, 2, 0, 0, 373, 80],
            [2, 0, 0, -2, 0, 361, 71],
            [2, 0, 0, 2, 0, 360, 98],
            [2, 0, 0, 0, -2, 380, 69],
            [2, 0, 0, 0, 2, 360, 74],
            [2, 0, 0, 0, 0, 370, 86],
            [2, 0, 0, 0, 0, 368, 74],
            [2, 0, 0, 0, 0, 369, 89],
            [2, 0, 0, 0, 0, 366, 76],
        ],
        columns=["block", "x1", "x2", "x3", "x4", "ave", "logSD"],
    )
    reference = {
        "coefficients": {
            "Intercept": 372.8,
            "block[T.2]": -2.95,
            "x1": -0.083333,
            "x2": 5.083333,
            "x3": 0.25,
            "x4": -6.083333,
            "x1 * x2": -2.875,
            "x1 * x3": -3.75,
            "x1 * x4": 4.375,
            "x2 * x3": 4.625,
            "x2 * x4": -1.5,
            "x3 * x4": -2.125,
            "x1^2": -2.0375,
            "x2^2": -1.6625,
            "x3^2": -2.5375,
            "x4^2": -0.1625,
        },
        "stationary": {"x1": 0.8607107, "x2": -0.3307115, "x3": -0.8394866, "x4": -0.1161465},
        "r2": 0.9555,
        "source_label": "rsm JSS article / heli example",
        "source_url": "https://rvlenth.github.io/rsm/articles/article-JSS.html",
        "source_note": "Published second-order model summary from the official rsm article; data values reproduced from a public heli printout.",
        "data_url": "https://stla.github.io/stlapblog/posts/rsm3Dshiny.html",
    }
    factor_specs = {
        "x1": {"kind": "coded"},
        "x2": {"kind": "coded"},
        "x3": {"kind": "coded"},
        "x4": {"kind": "coded"},
    }
    formula = "block + V1 + V2 + V3 + V4 + V1:V2 + V1:V3 + V1:V4 + V2:V3 + V2:V4 + V3:V4 + I(V1**2) + I(V2**2) + I(V3**2) + I(V4**2)"
    return df, factor_specs, "ave", formula, reference


def run_benchmark_analysis(benchmark_name):
    if benchmark_name == "ChemReact":
        df, factor_specs, response_name, formula, reference = chemreact_benchmark()
    else:
        df, factor_specs, response_name, formula, reference = heli_benchmark()

    coded_df, x_cols, safe_map, inverse_map, coding_info = prepare_coded_dataset(df, factor_specs)
    if "Block" in df.columns:
        coded_df["Block"] = coded_df["Block"].astype(str)
    if "block" in df.columns:
        coded_df["block"] = coded_df["block"].astype(str)

    model_bundle = fit_model_bundle(
        coded_df,
        response_name,
        formula,
        block_cols=[col for col in df.columns if col.lower().startswith("block")],
        safe_factor_names=list(safe_map.values()),
    )
    coef_df = coefficient_table(model_bundle, inverse_map)
    canonical_result = canonical_analysis(model_bundle, safe_map, coding_info)
    return {
        "data": df,
        "x_cols": x_cols,
        "response": response_name,
        "formula": formula,
        "reference": reference,
        "model_bundle": model_bundle,
        "coef_df": coef_df,
        "canonical_result": canonical_result,
        "summary_metrics": model_summary_metrics(model_bundle),
    }


def compare_against_reference(analysis_result):
    coef_df = analysis_result["coef_df"]
    reference = analysis_result["reference"]
    actual_map = dict(zip(coef_df["Term"], coef_df["Coefficient"]))
    rows = []
    for term, expected in reference["coefficients"].items():
        actual = actual_map.get(term)
        rows.append(
            {
                "Term": term,
                "Reference": expected,
                "Program": actual,
                "Absolute difference": np.nan if actual is None else abs(actual - expected),
            }
        )
    return pd.DataFrame(rows)


def render_planner_mode():
    st.subheader("DoE planner and experimental matrix generator")
    st.info("This workspace proposes a professional DoE plan from the number of factors and the model fidelity you want, then generates the coded run table and the expected degrees of freedom.")
    render_glossary(
        "Planner glossary",
        [
            ["Main effects", "Model that estimates only the linear contribution of each factor."],
            ["Two-factor interactions", "Model that also estimates how pairs of factors change each other's effect."],
            ["Quadratic surface", "Second-order model with linear, interaction, and squared terms. This is the classic response-surface target."],
            ["Center points", "Repeated runs at the design center used to estimate pure error and detect curvature."],
            ["Pure error df", "Degrees of freedom coming from replicated identical settings."],
            ["Lack-of-fit df", "Residual df left after removing pure error. This is the part used to judge model inadequacy."],
        ],
    )

    p1, p2, p3 = st.columns(3)
    n_factors = p1.slider("Number of factors", min_value=2, max_value=10, value=4)
    model_goal = p2.selectbox("Requested model fidelity", ["Main effects", "Two-factor interactions", "Quadratic surface"])
    design_choice = p3.selectbox(
        "Design family",
        ["Auto", "2-level Full Factorial", "3-level Full Factorial", "2-level Fractional Factorial", "Box-Behnken", "CCC", "CCI", "CCF"],
    )

    st.markdown("**Recommended shortlist**")
    st.dataframe(recommend_designs(n_factors, model_goal), use_container_width=True)

    default_centers = recommended_center_points(n_factors, design_choice if design_choice != "Auto" else "Auto", model_goal)
    c1, c2, c3 = st.columns(3)
    center_points = c1.number_input("Center points", min_value=0, max_value=20, value=int(default_centers), step=1)
    whole_design_replicates = c2.number_input("Whole-design replicates", min_value=1, max_value=5, value=1, step=1)
    fraction_reduction = c3.number_input("Fractional reduction p", min_value=1, max_value=max(1, n_factors - 1), value=1, step=1)

    alpha_default = float(round(2 ** (n_factors / 4), 4))
    alpha_value = st.number_input("Alpha for CCC / CCI", min_value=1.0, max_value=4.0, value=alpha_default, step=0.05)

    try:
        design_df, metadata = planner_design_from_selection(
            n_factors,
            model_goal,
            design_choice,
            int(center_points),
            int(whole_design_replicates),
            int(fraction_reduction),
            alpha_value,
        )
        factor_cols = [col for col in design_df.columns if col.startswith("x")]
        dof_info = design_dof_summary(design_df, factor_cols, model_goal)
    except Exception as exc:
        st.exception(exc)
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Design family", metadata["design_family"])
    m2.metric("Total runs", metadata["total_runs"])
    m3.metric("Unique design points", metadata["unique_design_points"])
    m4.metric("Model parameters", dof_info["Model parameters"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Residual df", dof_info["Residual df"])
    d2.metric("Pure error df", dof_info["Pure error df"])
    d3.metric(
        "Lack-of-fit df",
        f"{int(dof_info['Lack-of-fit df'])}" if np.isfinite(dof_info["Lack-of-fit df"]) else "NA",
    )
    d4.metric("Center points", int(metadata.get("center_points", 0)))

    summary_rows = [
        ["Model terms supported", ", ".join(expected_terms(n_factors, model_goal))],
        ["Professional recommendation", metadata.get("notes", "")],
        ["Replicate guidance", "Use at least 1 full-design replicate when the process is noisy; otherwise prioritize center-point replication for pure-error estimation."],
    ]
    if "generators" in metadata:
        summary_rows.append(["Generators", "; ".join(metadata["generators"])])
        summary_rows.append(["Approximate design resolution", metadata["resolution"]])
    if "alpha" in metadata:
        summary_rows.append(["Alpha", metadata["alpha"]])
    st.table(pd.DataFrame(summary_rows, columns=["Planning item", "Value"]))

    st.markdown("**Generated coded experimental matrix**")
    st.dataframe(design_df, use_container_width=True, height=420)
    st.download_button(
        "Download generated design as CSV",
        data=design_df.to_csv(index=False).encode("utf-8"),
        file_name=f"doe_plan_{metadata['design_family'].replace(' ', '_')}.csv",
        mime="text/csv",
    )


def render_benchmark_mode():
    st.subheader("Public benchmark validation")
    st.info("This workspace re-runs published DoE case studies and compares the program outputs with reference analyses from public sources.")
    benchmark_name = st.selectbox("Benchmark dataset", ["ChemReact", "heli"])

    try:
        result = run_benchmark_analysis(benchmark_name)
        comparison = compare_against_reference(result)
    except Exception as exc:
        st.exception(exc)
        return

    reference = result["reference"]
    metrics = result["summary_metrics"]
    max_diff = float(comparison["Absolute difference"].dropna().max()) if comparison["Absolute difference"].notna().any() else np.nan

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Runs", len(result["data"]))
    b2.metric("R2 / pseudo R2", f"{metrics.get('R2', metrics.get('Pseudo R2', np.nan)):.4f}")
    b3.metric("Reference R2", f"{reference['r2']:.4f}")
    b4.metric("Max coefficient diff", f"{max_diff:.4g}" if np.isfinite(max_diff) else "NA")

    st.markdown(f"Reference analysis: [{reference['source_label']}]({reference['source_url']})")
    st.caption(reference["source_note"])
    if "data_url" in reference:
        st.caption(f"Public dataset printout used for the embedded run table: {reference['data_url']}")

    st.markdown("**Coefficient comparison**")
    st.dataframe(comparison, use_container_width=True)

    if result["canonical_result"] is not None and reference.get("stationary"):
        station_rows = []
        for factor, ref_value in reference["stationary"].items():
            program_value = result["canonical_result"]["stationary_real"].get(factor, np.nan)
            station_rows.append(
                {
                    "Factor": factor,
                    "Reference stationary value": ref_value,
                    "Program stationary value": program_value,
                    "Absolute difference": abs(program_value - ref_value) if np.isfinite(program_value) else np.nan,
                }
            )
        st.markdown("**Stationary point comparison**")
        st.dataframe(pd.DataFrame(station_rows), use_container_width=True)

    st.markdown("**Embedded benchmark data**")
    st.dataframe(result["data"], use_container_width=True, height=360)


st.title("DoE Scientific Analyzer Pro")
st.caption(
    "Upload your DoE table and the app will build a user-friendly statistical workflow inspired by professional DoE software."
)

if "log_messages" not in st.session_state:
    st.session_state.log_messages = []

log_expand = st.expander("Live analysis log", expanded=True)
log_box = log_expand.empty()

workspace_mode = st.sidebar.radio(
    "Workspace",
    ["Analyze dataset", "DoE planner", "Public benchmarks"],
    help="Use the classic analyzer on your own data, generate a professional design plan, or validate the engine against public benchmark studies.",
)

if workspace_mode == "DoE planner":
    render_planner_mode()
    write_log("Planner workspace ready")
    st.stop()

if workspace_mode == "Public benchmarks":
    render_benchmark_mode()
    write_log("Benchmark workspace ready")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "1. Upload DoE file",
    type=["xlsx", "csv"],
    help="Upload the experimental dataset. Numeric factors should start with x, for example x1, x2, x3. Optional blocking columns can start with block. Remaining columns are treated as responses/CQAs.",
)
if not uploaded_file:
    st.info("Upload a .csv or .xlsx file to start the DoE analysis.")
    st.stop()

model_type = st.sidebar.selectbox(
    "2. Model structure",
    ["Standard (Linear)", "Interactions", "Quadratic", "Full (Interactions + Quadratic)"],
    help="Choose the mathematical structure of the fitted model. Linear uses only main effects, Interactions adds factor-factor interactions, Quadratic adds curvature terms, Full includes both interactions and quadratic terms.",
)
alpha = st.sidebar.slider(
    "Significance level",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
    help="Alpha threshold used to flag statistically significant terms. A common default is 0.05.",
)

write_log(f"Loading file {uploaded_file.name}")
file_bytes = uploaded_file.getvalue()
data_signature = hashlib.md5(file_bytes).hexdigest()
df = load_dataframe_from_upload(file_bytes, uploaded_file.name)
x_cols = [col for col in df.columns if col.lower().startswith("x")]
block_cols = [col for col in df.columns if col.lower().startswith("block")]
# Run/order/ID columns describe the experiment sequence, not a measured response.
RUN_LABEL_NAMES = {"run", "runorder", "runno", "runnumber", "run#", "stdorder", "standardorder", "std",
                   "order", "id", "index", "exp", "expno", "experiment", "experimentno"}
run_cols = [col for col in df.columns
            if col not in x_cols + block_cols
            and re.sub(r"[\s_.\-]", "", str(col).lower()) in RUN_LABEL_NAMES]
y_cols = [col for col in df.columns if col not in x_cols + block_cols + run_cols]
if run_cols:
    write_log(f"Ignored run/order columns: {', '.join(map(str, run_cols))}")

if not x_cols:
    st.error("No factor columns found. Rename factor columns so they start with x, for example x1, x2, x3.")
    st.stop()
if not y_cols:
    st.error("No response columns found. Keep factors as x* columns, optional blocks as block*, and responses as the remaining columns.")
    st.stop()

coding_mode = st.sidebar.selectbox(
    "4. Factor coding",
    ["Auto from observed min/max", "Manual center and half-range"],
    help="Auto coding uses the observed factor minimum and maximum. Manual coding lets you reproduce literature or plant-standard codings, which is especially important for central-composite and star-point designs.",
)
custom_coding = {}
if coding_mode == "Manual center and half-range":
    st.sidebar.markdown("**Manual coding settings**")
    for factor in x_cols:
        numeric_col = pd.to_numeric(df[factor], errors="coerce")
        default_center = float((numeric_col.min() + numeric_col.max()) / 2)
        default_half = float((numeric_col.max() - numeric_col.min()) / 2)
        center_value = st.sidebar.number_input(
            f"{factor} center",
            value=default_center,
            key=f"coding_center_{factor}",
            help=f"Engineering center used to code {factor}. The coded value becomes (real - center) / half-range.",
        )
        half_range_value = st.sidebar.number_input(
            f"{factor} half-range",
            value=default_half if default_half != 0 else 1.0,
            min_value=1e-9,
            key=f"coding_half_{factor}",
            help=f"Engineering half-range used to code {factor}. Use the factorial step size, not the star-point maximum, when reproducing literature CCD codings.",
        )
        custom_coding[factor] = {"center": center_value, "half_range": half_range_value}

selected_y = st.sidebar.selectbox(
    "5. Response / CQA",
    y_cols,
    help="Select the response variable to model in the main analysis tabs. The app can still inspect additional responses in PCA and multi-response desirability.",
)
section_names = [
    "Overview",
    "Model",
    "Effects",
    "ANOVA",
    "Diagnostics",
    "3D Maps",
    "Optimizer",
    "Multi-response",
    "PCA",
    "Guide",
]
selected_section = st.radio(
    "Section",
    section_names,
    horizontal=True,
    help="Only the selected section is rendered, which keeps the app responsive even on larger datasets.",
)
write_log(f"Detected {len(x_cols)} factors, {len(block_cols)} blocking columns and {len(y_cols)} responses")

safe_map, inverse_map = make_safe_mapping(x_cols)
df_coded, coding_info = code_factors(df, x_cols, safe_map, custom_coding=custom_coding)
formula = build_formula(list(safe_map.values()), model_type, block_terms=block_cols)
write_log(f"Model formula: {formula}")

try:
    model_cache_key = (
        data_signature,
        selected_y,
        formula,
        dict_signature(custom_coding),
        tuple(block_cols),
    )
    model_bundle = get_session_model_bundle(
        model_cache_key,
        df_coded,
        selected_y,
        formula,
        block_cols,
        list(safe_map.values()),
    )
except Exception as exc:
    st.exception(exc)
    st.stop()

response_models = {selected_y: model_bundle}
if selected_section == "Multi-response":
    for response_name in y_cols:
        if response_name == selected_y:
            continue
        try:
            response_models[response_name] = get_session_model_bundle(
                (
                    data_signature,
                    response_name,
                    formula,
                    dict_signature(custom_coding),
                    tuple(block_cols),
                ),
                df_coded,
                response_name,
                formula,
                block_cols,
                list(safe_map.values()),
            )
        except Exception:
            continue

coef_df = coefficient_table(model_bundle, inverse_map)
summary_metrics = model_summary_metrics(model_bundle)
anova_df = anova_table(model_bundle)
lof_result = pure_error_lack_of_fit(model_bundle, df, x_cols, block_cols)
canonical_result = canonical_analysis(model_bundle, safe_map, coding_info)
steepest_df = steepest_ascent_table(model_bundle, safe_map, coding_info)
write_log(f"Model fitted with {model_bundle['model_kind']} engine")

if st.sidebar.button("Prepare full analysis package"):
    try:
        export_bytes, export_name = build_analysis_export_package(
            df,
            x_cols,
            block_cols,
            y_cols,
            selected_y,
            alpha,
            safe_map,
            inverse_map,
            coding_info,
            model_bundle,
            coef_df,
            summary_metrics,
            anova_df,
            lof_result,
            canonical_result,
            steepest_df,
            response_models,
        )
        st.session_state["analysis_export_bytes"] = export_bytes
        st.session_state["analysis_export_name"] = export_name
        write_log("Full analysis package prepared for download")
    except Exception as exc:
        st.exception(exc)

if "analysis_export_bytes" in st.session_state:
    st.sidebar.download_button(
        "Download full analysis package (.zip)",
        data=st.session_state["analysis_export_bytes"],
        file_name=st.session_state.get("analysis_export_name", f"analysis_{selected_y}.zip"),
        mime="application/zip",
    )
if selected_section == "Overview":
    st.subheader("Dataset overview")
    st.info("Use this tab to verify factors, responses, missing values, and whether the selected response is censored.")
    render_glossary(
        "How to read this tab",
        [
            ["Runs", "Number of experimental trials available in the uploaded dataset."],
            ["Factors", "Input variables you intentionally changed in the experiment, usually process settings."],
            ["Responses", "Measured outputs or CQAs that you want to explain or optimize."],
            ["Active engine", "Regression family used for the selected response: OLS for ordinary data, Tobit for censored data such as < or > values."],
            ["Non-null", "How many rows contain a value for that variable."],
            ["Unique", "How many different values appear in that variable. A very low number may indicate a discrete design."],
            ["Correlation heatmap", "Quick visual check of how strongly factors and the chosen response move together. It does not prove causality."],
        ],
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Runs", len(df))
    c2.metric("Factors", len(x_cols))
    c3.metric("Blocks", len(block_cols))
    c4.metric("Responses", len(y_cols))
    c5.metric("Active engine", model_bundle["model_kind"])

    if model_bundle["has_censoring"]:
        st.warning(
            "Censored values detected. The app uses Tobit for prediction and coefficients. ANOVA is shown from the matching OLS model as an engineering approximation."
        )
    else:
        st.success("No censoring detected. OLS inference is fully applicable.")

    st.dataframe(overview_table(df, x_cols, y_cols, block_cols), use_container_width=True)

    corr_fig = plot_correlation_heatmap(df, x_cols, selected_y)
    if corr_fig is not None:
        st.pyplot(corr_fig)

if selected_section == "Model":
    st.subheader("Model summary")
    st.info("The model is fitted on coded factors, so effect sizes are directly comparable across variables. This section now also includes response-surface follow-up inspired by open-source `rsm` workflows.")
    render_glossary(
        "How to interpret model parameters",
        [
            ["RMSE", "Typical prediction error size. Lower is better because the model misses the data by a smaller amount."],
            ["Pseudo R2", "Approximate fraction of response variability explained by the fitted model. Higher is usually better."],
            ["R2", "Share of variability explained by the model in ordinary least squares regression."],
            ["Adj R2", "R2 corrected for model complexity. It helps detect when extra terms do not really improve the model."],
            ["AIC / BIC", "Model comparison scores that penalize unnecessary complexity. Lower values are preferred when comparing models on the same response."],
            ["Coefficient", "Estimated effect of a term on the response. Positive means the response increases as that term increases; negative means the opposite."],
            ["Std Error", "Uncertainty of the estimated coefficient. Large values mean the estimate is less stable."],
            ["Statistic", "Signal-to-noise ratio of the coefficient, usually a t or z statistic. Larger absolute values indicate stronger evidence of a real effect."],
            ["p-value", "Probability of observing an effect this large if the true effect were actually zero. Smaller values indicate stronger evidence that the term matters."],
            ["Significant", "A simple yes/no flag based on the selected alpha threshold. It is a decision aid, not an absolute truth."],
            ["Coded factors", "Factors rescaled so the design center is around 0 and the low/high settings are around -1 and +1. This makes effects easier to compare."],
        ],
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("RMSE", f"{summary_metrics['RMSE']:.4g}")
    mc2.metric(
        "Pseudo R2",
        f"{summary_metrics['Pseudo R2']:.4f}" if not np.isnan(summary_metrics["Pseudo R2"]) else "NA",
    )
    mc3.metric("Parameters", summary_metrics["Parameters"])
    mc4.metric("DoF", summary_metrics["Degrees of freedom"])

    if "R2" in summary_metrics:
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("R2", f"{summary_metrics['R2']:.4f}")
        oc2.metric("Adj R2", f"{summary_metrics['Adj R2']:.4f}")
        oc3.metric("AIC", f"{summary_metrics['AIC']:.2f}")
    else:
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("LogLik", f"{summary_metrics['LogLik']:.2f}")
        tc2.metric("AIC", f"{summary_metrics['AIC']:.2f}")
        tc3.metric("BIC", f"{summary_metrics['BIC']:.2f}")

    st.markdown(f"**Prediction equation for `{selected_y}`**")
    st.code(effect_equation(model_bundle, inverse_map), language="text")
    display_coef = coef_df.copy()
    display_coef["Significant"] = np.where(display_coef["p-value"] <= alpha, "Yes", "No")
    st.dataframe(display_coef.drop(columns=["Abs Statistic"]), use_container_width=True)

    st.markdown("**Canonical analysis / stationary point**")
    if canonical_result is None:
        st.caption("Canonical analysis is available only when the fitted model contains usable quadratic curvature and the stationary point can be solved.")
    else:
        render_glossary(
            "How to interpret canonical analysis",
            [
                ["Stationary point", "Point where the local slope is zero in all factor directions. It is a candidate optimum or saddle point."],
                ["Stationary point type", "Maximum means a local peak, Minimum means a local valley, Saddle means mixed behavior depending on direction."],
                ["Eigenvalue", "Curvature along a canonical axis. Negative values suggest downward curvature, positive values upward curvature."],
                ["Real stationary value", "Suggested factor setting expressed in the original engineering units rather than coded units."],
            ],
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric("Stationary point type", canonical_result["nature"])
            st.metric("Predicted response at stationary point", f"{canonical_result['predicted_response']:.5g}")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Factor": canonical_result["factors"],
                        "Coded stationary value": canonical_result["stationary_coded"],
                        "Real stationary value": [canonical_result["stationary_real"][f] for f in canonical_result["factors"]],
                    }
                ),
                use_container_width=True,
            )
        with cc2:
            eig_df = pd.DataFrame(
                {
                    "Canonical axis": [f"Axis {i + 1}" for i in range(len(canonical_result["eigenvalues"]))],
                    "Eigenvalue": canonical_result["eigenvalues"],
                }
            )
            st.dataframe(eig_df, use_container_width=True)
            st.caption("All negative eigenvalues suggest a local maximum, all positive a local minimum, mixed signs a saddle point.")

if selected_section == "Effects":
    st.subheader("Standardized effects and coefficients")
    st.info("The Pareto chart ranks standardized effects. The coefficient plot shows direction and uncertainty.")
    render_glossary(
        "How to interpret effects",
        [
            ["Pareto chart", "Ranks terms from least to most important based on standardized effect size."],
            ["Absolute t / z statistic", "Magnitude of the effect after accounting for uncertainty. Higher bars mean stronger evidence."],
            ["Threshold line", "Reference cutoff for significance at the selected alpha level."],
            ["Main effect", "Effect of changing one factor while ignoring interactions."],
            ["Interaction", "Effect showing that one factor changes the effect of another factor."],
            ["Quadratic term", "Curvature effect showing that the response may bend or have an internal optimum."],
        ],
    )
    e1, e2 = st.columns(2)
    pareto_fig, threshold = plot_pareto(coef_df, model_bundle["dof"], alpha)
    with e1:
        st.pyplot(pareto_fig)
        st.caption(f"Reference threshold: {threshold:.3f}. Bars beyond the line are significant at alpha = {alpha:.2f}.")
    with e2:
        st.pyplot(plot_coefficients(coef_df))
        st.caption("Error bars represent standard errors on the coded-factor scale.")

if selected_section == "ANOVA":
    st.subheader("ANOVA and term contribution")
    st.info("ANOVA partitions variability into contributions from each term and the residual error. When replicate runs exist, the app also computes a lack-of-fit test against pure error.")
    render_glossary(
        "How to interpret ANOVA",
        [
            ["sum_sq", "Sum of squares, the amount of variability attributed to each term."],
            ["df", "Degrees of freedom, the amount of independent information used by that term."],
            ["F", "Ratio between explained variability and residual variability. Larger values indicate stronger evidence that the term matters."],
            ["PR(>F)", "ANOVA p-value associated with the F statistic."],
            ["Residual", "Variability left unexplained by the current model."],
            ["Pure error", "Variation among repeated experimental runs performed at the same factor settings."],
            ["Lack-of-fit", "Extra residual variation not explained by pure error. A significant lack-of-fit suggests the model form may be inadequate."],
        ],
    )
    anova_display = anova_df.copy()
    if "PR(>F)" in anova_display.columns:
        anova_display["Significant"] = np.where(anova_display["PR(>F)"] <= alpha, "Yes", "No")
    st.dataframe(anova_display, use_container_width=True)
    if "sum_sq" in anova_display.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_terms = anova_display.loc[anova_display["Term"] != "Residual", :]
        ax.bar(plot_terms["Term"], plot_terms["sum_sq"], color="#4daf4a", edgecolor="black")
        ax.set_ylabel("Sum of squares")
        ax.set_title("ANOVA contribution by term")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("**Lack-of-fit vs pure error**")
    if lof_result is None:
        st.caption("Lack-of-fit could not be estimated because the design does not contain enough replicated factor settings.")
    else:
        st.dataframe(pd.DataFrame([lof_result]), use_container_width=True)
        if np.isfinite(lof_result["p-value"]):
            if lof_result["p-value"] <= alpha:
                st.warning("The lack-of-fit test is significant. The fitted model may be missing structure or curvature.")
            else:
                st.success("No significant lack-of-fit detected relative to pure error.")

if selected_section == "Diagnostics":
    st.subheader("Diagnostics")
    st.info("Check these plots before trusting optimization results.")
    render_glossary(
        "How to interpret diagnostics",
        [
            ["Observed vs Predicted", "Checks whether predictions match measured values. Points close to the diagonal indicate better fit."],
            ["Residual", "Observed minus predicted value. Residuals should look random rather than patterned."],
            ["Normal Q-Q", "Checks whether residuals are approximately normal. Large deviations may indicate model or data issues."],
            ["Standardized residual", "Residual scaled by its typical variation so unusually large errors stand out."],
            ["Run order plot", "Checks whether errors drift over time, which may suggest instability, batch effects, or measurement drift."],
            ["Leverage", "Measures how extreme a run is in factor space. High leverage points can strongly influence the fit."],
            ["Cook's D", "Measures how much the fitted model would change if a run were removed. Large bubbles deserve attention."],
        ],
    )
    d1, d2 = st.columns(2)
    with d1:
        st.pyplot(plot_actual_vs_predicted(model_bundle))
    with d2:
        st.pyplot(plot_residuals_vs_predicted(model_bundle))

    d3, d4 = st.columns(2)
    with d3:
        st.pyplot(plot_normal_qq(model_bundle))
    with d4:
        st.pyplot(plot_residual_histogram(model_bundle))

    d5, d6 = st.columns(2)
    with d5:
        st.pyplot(plot_residuals_vs_run_order(model_bundle))

    influence_fig = plot_influence(model_bundle)
    if influence_fig is not None:
        with d6:
            st.pyplot(influence_fig)
    else:
        with d6:
            st.caption("Influence plot is available only for OLS fits.")

if selected_section == "3D Maps":
    st.subheader("Response surfaces and contour maps")
    st.info("Choose two factors for the map. Remaining factors are fixed at user-selected values. Controls are applied on demand for faster updates.")
    render_glossary(
        "How to interpret 3D maps",
        [
            ["Response surface", "Predicted response drawn as a continuous surface over two selected factors."],
            ["Contour map", "Top view of the surface using lines of equal predicted response."],
            ["Fixed factors", "All non-plotted factors are kept constant at the chosen values while the map varies only two factors."],
            ["Curvature", "Bending of the surface. It often indicates that a simple straight-line model is not enough."],
            ["Ridge or valley", "A stretched optimum region where different factor combinations can produce similar responses."],
        ],
    )
    if len(x_cols) < 2:
        st.warning("At least two factors are required to generate 3D maps.")
    else:
        with st.form("surface_controls"):
            sf1, sf2 = st.columns(2)
            with sf1:
                surface_x = st.selectbox(
                    "X-axis factor",
                    x_cols,
                    index=0,
                    key="surface_x",
                    help="Factor displayed on the X axis of the response surface and contour map.",
                )
            with sf2:
                surface_y = st.selectbox(
                    "Y-axis factor",
                    x_cols,
                    index=1 if len(x_cols) > 1 else 0,
                    key="surface_y",
                    help="Factor displayed on the Y axis of the response surface and contour map.",
                )

            st.markdown("**Fixed values for remaining factors**")
            fixed_values = {}
            for factor in x_cols:
                info = coding_info[factor]
                default_val = info["center"]
                if factor not in [surface_x, surface_y]:
                    fixed_values[factor] = st.slider(
                        factor,
                        min_value=float(info["low"]),
                        max_value=float(info["high"]),
                        value=float(default_val),
                        key=f"surface_{factor}",
                        help=f"Fixed setting for {factor} while the app varies {surface_x} and {surface_y} across the plotted surface.",
                    )
                else:
                    fixed_values[factor] = default_val
            apply_surface = st.form_submit_button("Update 3D maps")

        if "surface_settings" not in st.session_state:
            st.session_state["surface_settings"] = {
                "surface_x": x_cols[0],
                "surface_y": x_cols[1 if len(x_cols) > 1 else 0],
                "fixed_values": {factor: coding_info[factor]["center"] for factor in x_cols},
            }
        if apply_surface:
            st.session_state["surface_settings"] = {
                "surface_x": surface_x,
                "surface_y": surface_y,
                "fixed_values": fixed_values,
            }

        active_surface = st.session_state["surface_settings"]
        surface_x = active_surface["surface_x"]
        surface_y = active_surface["surface_y"]
        fixed_values = active_surface["fixed_values"]

        if surface_x == surface_y:
            st.warning("Choose two different factors.")
        else:
            xr, yr, zm = make_surface_grid(
                model_bundle,
                safe_map,
                coding_info,
                surface_x,
                surface_y,
                fixed_values,
            )
            s1, s2 = st.columns(2)
            with s1:
                fig3d = plt.figure(figsize=PLOT_SIZE_STANDARD)
                ax3d = fig3d.add_subplot(111, projection="3d")
                surf = ax3d.plot_surface(xr, yr, zm, cmap="viridis", edgecolor="none", alpha=0.9)
                ax3d.set_xlabel(surface_x)
                ax3d.set_ylabel(surface_y)
                ax3d.set_zlabel(selected_y)
                ax3d.set_title("Predicted response surface")
                fig3d.colorbar(surf, shrink=0.6, pad=0.1)
                fig3d.tight_layout()
                st.pyplot(fig3d)

            with s2:
                fig2d, ax2d = plt.subplots(figsize=PLOT_SIZE_STANDARD)
                contour = ax2d.contourf(xr, yr, zm, levels=18, cmap="viridis")
                lines = ax2d.contour(xr, yr, zm, levels=10, colors="black", linewidths=0.5, alpha=0.6)
                ax2d.clabel(lines, inline=True, fontsize=8)
                ax2d.set_xlabel(surface_x)
                ax2d.set_ylabel(surface_y)
                ax2d.set_title("Contour map")
                fig2d.colorbar(contour, ax=ax2d)
                fig2d.tight_layout()
                st.pyplot(fig2d)

if selected_section == "Optimizer":
    st.subheader("Numerical optimizer")
    st.info("Search the optimum inside the observed design region, inspect the path of steepest ascent, or test a manual what-if scenario.")
    render_glossary(
        "How to interpret optimization outputs",
        [
            ["Optimal predicted response", "Best response value predicted by the model under the chosen optimization goal and within the tested factor range."],
            ["Optimizer success", "Indicates whether the numerical search algorithm converged to a stable solution."],
            ["Manual what-if prediction", "Single prediction at user-selected settings, useful for comparing candidate operating points."],
            ["Steepest ascent / descent", "Suggested path for follow-up experiments when you want to move the process quickly toward better response values from the design center."],
            ["Step size (coded)", "Distance moved along the gradient in coded-factor units."],
        ],
    )
    goal = st.selectbox(
        "Optimization goal",
        ["Maximize", "Minimize", "Target"],
        help="Choose whether the selected response should be maximized, minimized, or driven toward a target value.",
    )
    target_value = None
    if goal == "Target":
        y_numeric = pd.to_numeric(
            clean_text_series(df[selected_y]).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
            errors="coerce",
        )
        target_value = st.number_input(
            "Target value",
            value=float(np.nanmean(y_numeric)) if np.isfinite(np.nanmean(y_numeric)) else 0.0,
            help="Desired target for the selected response when using target-based optimization.",
        )

    best_real, best_pred, opt_result = optimize_response(
        model_bundle,
        safe_map,
        coding_info,
        goal,
        target_value,
    )
    o1, o2 = st.columns(2)
    with o1:
        st.metric("Optimal predicted response", f"{best_pred:.5g}")
        st.metric("Optimizer success", "Yes" if opt_result.success else "No")
    with o2:
        st.dataframe(
            pd.DataFrame({"Factor": list(best_real.keys()), "Optimal setting": list(best_real.values())}),
            use_container_width=True,
        )

    st.markdown("**Manual what-if prediction**")
    manual_real = {}
    for factor in x_cols:
        info = coding_info[factor]
        manual_real[factor] = st.slider(
            factor,
            min_value=float(info["low"]),
            max_value=float(info["high"]),
            value=float(info["center"]),
            key=f"manual_{factor}",
            help=f"Manual setting for {factor} used to compute a one-point predicted response.",
        )
    pred_df = pd.DataFrame({safe_map[f]: [encode_factor(manual_real[f], f, coding_info)] for f in x_cols})
    manual_pred, _ = predict_from_coded_rows(model_bundle, pred_df)
    st.metric("Predicted response at manual settings", f"{float(manual_pred[0]):.5g}")

    st.markdown("**Path of steepest ascent / descent**")
    if steepest_df is None:
        st.caption("The linear gradient is too small to define a meaningful steepest-ascent direction.")
    else:
        st.dataframe(steepest_df, use_container_width=True)
        st.caption("This table follows the coded gradient from the design center and is useful for first-order follow-up experiments.")

if selected_section == "Multi-response":
    st.subheader("Multi-response desirability")
    st.info("This tab combines multiple fitted responses into a single desirability score, a common feature in professional DoE optimization tools.")
    render_glossary(
        "How to interpret multi-response desirability",
        [
            ["Desirability", "Score from 0 to 1 showing how acceptable a predicted response is with respect to your optimization goal and limits."],
            ["Composite desirability", "Overall score combining all selected responses. Values closer to 1 indicate a better global compromise."],
            ["Low / High", "Bounds that define the acceptable region for each response."],
            ["Target", "Preferred value when you do not simply want the response as high or as low as possible."],
            ["Goal", "Tells the app whether each response should be maximized, minimized, or matched to a target."],
        ],
    )
    available_multi = [resp for resp in y_cols if resp in response_models]
    chosen_responses = st.multiselect(
        "Responses to include",
        available_multi,
        default=available_multi[: min(3, len(available_multi))],
        help="Choose which responses should contribute to the composite desirability score.",
    )
    if not chosen_responses:
        st.caption("Select at least one response to compute composite desirability.")
    else:
        desirability_specs = []
        for response_name in chosen_responses:
            numeric_resp = pd.to_numeric(
                clean_text_series(df[response_name]).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                errors="coerce",
            )
            low_default = float(np.nanmin(numeric_resp))
            high_default = float(np.nanmax(numeric_resp))
            target_default = float(np.nanmean(numeric_resp))
            st.markdown(f"**{response_name}**")
            c1, c2, c3, c4 = st.columns(4)
            goal_resp = c1.selectbox(
                "Goal",
                ["Maximize", "Minimize", "Target"],
                key=f"goal_{response_name}",
                help=f"Optimization direction for {response_name}. Choose whether higher is better, lower is better, or a target value is preferred.",
            )
            low_resp = c2.number_input(
                "Low",
                value=low_default,
                key=f"low_{response_name}",
                help=f"Lower bound for desirability of {response_name}. Values outside the acceptable region can receive low or zero desirability.",
            )
            target_resp = c3.number_input(
                "Target",
                value=target_default,
                key=f"target_{response_name}",
                help=f"Preferred target for {response_name}. This is used directly for target optimization and as a reference value for interpretation.",
            )
            high_resp = c4.number_input(
                "High",
                value=high_default,
                key=f"high_{response_name}",
                help=f"Upper bound for desirability of {response_name}. For maximize goals this is the fully desirable region; for minimize goals it marks the unacceptable high end.",
            )
            desirability_specs.append((response_name, goal_resp, low_resp, target_resp, high_resp))

        mr_settings = {}
        st.markdown("**Factor settings for composite desirability**")
        for factor in x_cols:
            info = coding_info[factor]
            mr_settings[factor] = st.slider(
                f"{factor}",
                min_value=float(info["low"]),
                max_value=float(info["high"]),
                value=float(info["center"]),
                key=f"multi_{factor}",
                help=f"Factor setting used to evaluate the predicted value and desirability of all selected responses.",
            )

        desirability_rows = []
        individual_scores = []
        for response_name, goal_resp, low_resp, target_resp, high_resp in desirability_specs:
            model_resp = response_models[response_name]
            pred_input = pd.DataFrame({safe_map[f]: [encode_factor(mr_settings[f], f, coding_info)] for f in x_cols})
            pred_value, _ = predict_from_coded_rows(model_resp, pred_input)
            score = desirability_function(float(pred_value[0]), low_resp, target_resp, high_resp, goal_resp)
            individual_scores.append(max(score, 1e-12))
            desirability_rows.append(
                {
                    "Response": response_name,
                    "Goal": goal_resp,
                    "Predicted value": float(pred_value[0]),
                    "Desirability": score,
                }
            )

        composite = float(np.prod(individual_scores) ** (1 / len(individual_scores)))
        st.metric("Composite desirability", f"{composite:.4f}")
        st.dataframe(pd.DataFrame(desirability_rows), use_container_width=True)

if selected_section == "PCA":
    st.subheader("PCA explorer")
    st.info("PCA gives a multivariate view of your experiment and helps identify clusters, gradients, and correlated responses.")
    render_glossary(
        "How to interpret PCA",
        [
            ["PCA", "Principal Component Analysis. It compresses many correlated variables into a few summary axes."],
            ["PC1 / PC2", "First and second principal components, the directions capturing the largest variation in the data."],
            ["Explained variance", "Fraction of total variability captured by each principal component."],
            ["Cumulative variance", "Total fraction of variability captured when you keep the first several components together."],
            ["Score plot", "Shows how experimental runs relate to each other in reduced dimensions."],
            ["Loading", "Shows how strongly each original variable contributes to a principal component."],
            ["Biplot", "Combines score positions and variable loading directions in one plot, helping you see how runs and variables align."],
            ["80% rule", "A practical heuristic that keeps the smallest number of principal components explaining about 80% of the variance."],
        ],
    )
    pca_source = st.selectbox(
        "Variables for PCA",
        ["Responses only", "Factors + responses"],
        help="Choose whether PCA should summarize only the measured responses or the full joint space of factors and responses.",
    )
    biplot_color = st.selectbox(
        "Color biplot by",
        ["None"] + x_cols + block_cols,
        help="Coloring the score space by a factor or block often makes DoE patterns visually easier to interpret.",
    )
    pca_cols = y_cols if pca_source == "Responses only" else x_cols + y_cols
    pca_numeric = pd.DataFrame(
        {
            col: pd.to_numeric(
                clean_text_series(df[col]).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                errors="coerce",
            )
            for col in pca_cols
        }
    )
    pca_result = compute_pca(pca_numeric)
    if pca_result is None:
        st.warning("PCA needs at least two numeric columns and enough complete rows after removing missing values.")
    else:
        summary_values, pca_insights = pca_interpretation_summary(pca_result, pca_source, x_cols, y_cols)
        explained_df = pd.DataFrame(
            {
                "Component": [f"PC{i + 1}" for i in range(len(pca_result["explained"]))],
                "Explained variance": pca_result["explained"],
                "Cumulative variance": np.cumsum(pca_result["explained"]),
            }
        )
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("PC1 variance", f"{summary_values['PC1 variance %']:.2f}%")
        with p2:
            st.metric("PC1 + PC2", f"{summary_values['PC1+PC2 cumulative %']:.2f}%")
        with p3:
            st.metric("Components for 80%", summary_values["Components for 80%"])
        with p4:
            st.metric("Components for 85%", summary_values["Components for 85%"])

        for insight in pca_insights:
            st.caption(insight)

        color_values = None if biplot_color == "None" else df.loc[pca_result["rows"], biplot_color]
        bp1, bp2 = st.columns(2)
        with bp1:
            st.pyplot(plot_pca_scores(pca_result))
        with bp2:
            st.pyplot(plot_pca_biplot(pca_result, color_values=color_values, color_label=None if biplot_color == "None" else biplot_color))

        lp1, lp2 = st.columns(2)
        with lp1:
            st.pyplot(plot_pca_loadings(pca_result))
        with lp2:
            top_pc1 = pca_loading_table(pca_result, component_index=0).head(8)
            top_pc2 = pca_loading_table(pca_result, component_index=1).head(8) if len(pca_result["explained"]) > 1 else None
            st.markdown("**Top loadings to interpret PC1 / PC2**")
            if top_pc2 is None:
                st.dataframe(top_pc1, use_container_width=True)
            else:
                top_pc1 = top_pc1.rename(columns={"Absolute loading": "PC1 abs loading"})
                top_pc2 = top_pc2.rename(columns={"Absolute loading": "PC2 abs loading"})
                st.dataframe(top_pc1.merge(top_pc2, on="Variable", how="outer"), use_container_width=True)

        if pca_source == "Factors + responses":
            st.warning(
                "For designed experiments, PCA on factors + responses is exploratory only. Because the factors are already structured by design, PCA is often more actionable on the response block alone."
            )
        elif len(y_cols) >= 4:
            st.success(
                "This is a strong DoE use case for PCA: many responses can often be compressed into a few latent quality dimensions before optimization or follow-up modeling."
            )
        st.dataframe(explained_df, use_container_width=True)

if selected_section == "Guide":
    st.subheader("How to read the analysis")
    st.markdown(
        """
        **Recommended workflow**

        1. Start from **Overview** to validate factors, responses, missing values, and censoring.
        2. Use **Model** to inspect equation, fit quality, and coefficient table.
        3. Read **Effects** to identify the most important linear, interaction, and quadratic terms.
        4. Confirm statistical support in **ANOVA**.
        5. Inspect **Diagnostics** before trusting the optimum.
        6. Use **3D Maps** to understand interaction geometry and curvature.
        7. Use **Optimizer** for best predicted settings inside the tested region.
        8. Open **PCA** for a multivariate view across many responses.

        **Interpretation notes**

        - Linear terms describe directional effects.
        - Interaction terms show when one factor changes the effect of another.
        - Quadratic terms reveal curvature and internal optima.
        - A large coefficient is not automatically significant if uncertainty is also large.
        - If the response contains `<` or `>` values, Tobit is used for prediction because ordinary least squares is not ideal for censored measurements.
        """
    )
    render_glossary(
        "Quick glossary of statistical terms",
        [
            ["Regression model", "An equation that links process factors to a measured response."],
            ["Term", "A piece of the model, for example a factor, an interaction, or a quadratic effect."],
            ["Effect", "Practical influence of a term on the response."],
            ["Significance", "How strong the statistical evidence is that an observed effect is not just random noise."],
            ["Curvature", "Evidence that the response does not change in a straight line across the factor range."],
            ["Censoring", "Situation where the instrument only tells you a value is below or above a limit, not the exact value."],
            ["Optimization", "Using the fitted model to suggest factor settings that improve one or more responses."],
        ],
    )

write_log("Analysis completed")
