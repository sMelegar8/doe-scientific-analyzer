"""Professional plotting helpers for DoE reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .doe_analysis import DoeModel, response_surface_grid

PLOT_STYLE = "seaborn-v0_8-whitegrid"


def _label(name: str, label_map: dict[str, str] | None = None) -> str:
    """Return a presentation-friendly label while preserving internal column names."""
    return (label_map or {}).get(name, name)


def _term_label(term: str, label_map: dict[str, str] | None = None) -> str:
    """Convert statsmodels formula terms into compact labels for Pareto plots."""
    label_map = label_map or {}
    cleaned = term.replace("I(", "").replace(")", "").replace(" ** 2", "^2")
    parts = cleaned.split(":")
    readable = []
    for part in parts:
        base = part
        suffix = ""
        if part.endswith("^2"):
            base = part[:-2]
            suffix = "^2"
        if base.startswith("x_"):
            base = base[2:]
        readable.append(label_map.get(base, base.upper()) + suffix)
    return " x ".join(readable)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 180) -> Path:
    """Save a Matplotlib figure and close it to keep batch runs lightweight."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto(
    coefs: pd.DataFrame,
    alpha: float = 0.05,
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Create a Pareto chart of standardized factor effects."""
    with plt.style.context(PLOT_STYLE):
        plot_df = coefs[coefs["term"] != "Intercept"].sort_values("absolute_t")
        threshold = stats.t.ppf(1 - alpha / 2, df=max(len(plot_df), 1))
        colors = np.where(plot_df["absolute_t"] >= threshold, "#0f766e", "#c2410c")
        y_labels = [_term_label(term, label_map) for term in plot_df["term"]]

        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.barh(y_labels, plot_df["absolute_t"], color=colors, edgecolor="#222222")
        ax.axvline(threshold, color="#1d4ed8", linestyle="--", linewidth=1.8, label=f"alpha={alpha}")
        ax.set_xlabel("Absolute t statistic")
        ax.set_title("Pareto Chart of Standardized Effects")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_observed_vs_predicted(
    model: DoeModel,
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot observed responses against model predictions."""
    with plt.style.context(PLOT_STYLE):
        observed = model.analysis_frame[model.response].to_numpy()
        predicted = model.result.fittedvalues.to_numpy()
        low = min(observed.min(), predicted.min())
        high = max(observed.max(), predicted.max())

        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.scatter(predicted, observed, s=58, color="#2563eb", edgecolor="#111827", alpha=0.85)
        ax.plot([low, high], [low, high], color="#111827", linestyle="--", linewidth=1.4)
        ax.set_xlabel(f"Predicted {_label(model.response, label_map)}")
        ax.set_ylabel(f"Observed {_label(model.response, label_map)}")
        ax.set_title(f"Observed vs Predicted: {_label(model.response, label_map)}")
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_response_surface(
    model: DoeModel,
    x_factor: str,
    y_factor: str,
    label_map: dict[str, str] | None = None,
    recommended_region: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]] | None = None,
    desirability_threshold: float = 0.7,
) -> plt.Figure:
    """Plot a filled contour response surface for two selected factors."""
    x_mesh, y_mesh, z_mesh = response_surface_grid(model, x_factor, y_factor)
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        contour = ax.contourf(x_mesh, y_mesh, z_mesh, levels=18, cmap="viridis")
        lines = ax.contour(x_mesh, y_mesh, z_mesh, levels=8, colors="white", linewidths=0.7, alpha=0.8)
        ax.clabel(lines, inline=True, fontsize=8)
        if recommended_region is not None:
            rec_x, rec_y, desirability, best_point = recommended_region
            ax.contour(
                rec_x,
                rec_y,
                desirability,
                levels=[desirability_threshold],
                colors=["#f97316"],
                linewidths=2.5,
            )
            ax.scatter(
                [best_point[0]],
                [best_point[1]],
                marker="*",
                s=170,
                color="#f97316",
                edgecolor="#111827",
                zorder=5,
            )
        fig.colorbar(contour, ax=ax, label=_label(model.response, label_map))
        ax.set_xlabel(_label(x_factor, label_map))
        ax.set_ylabel(_label(y_factor, label_map))
        ax.set_title(f"Response Surface: {_label(model.response, label_map)}")
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_pca_scores(pca_result: dict[str, np.ndarray | list[str]]) -> plt.Figure:
    """Create a PCA score plot using PC1 and PC2."""
    scores = np.asarray(pca_result["scores"])
    explained = np.asarray(pca_result["explained_variance"])
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.scatter(scores[:, 0], scores[:, 1], s=60, color="#7c3aed", edgecolor="#111827", alpha=0.85)
        ax.axhline(0, color="#6b7280", linewidth=0.9)
        ax.axvline(0, color="#6b7280", linewidth=0.9)
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.set_title("PCA Score Plot")
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_pca_loadings(
    pca_result: dict[str, np.ndarray | list[str]],
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Create a PCA loading plot for PC1 and PC2."""
    columns = list(pca_result["columns"])
    loadings = np.asarray(pca_result["loadings"])
    load_df = pd.DataFrame({"variable": columns, "PC1": loadings[:, 0], "PC2": loadings[:, 1]})

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5.4))
        y = np.arange(len(load_df))
        ax.barh(y - 0.18, load_df["PC1"], height=0.36, color="#0f766e", label="PC1")
        ax.barh(y + 0.18, load_df["PC2"], height=0.36, color="#f59e0b", label="PC2")
        ax.set_yticks(y)
        ax.set_yticklabels([_label(variable, label_map) for variable in load_df["variable"]])
        ax.axvline(0, color="#111827", linewidth=1.0)
        ax.set_xlabel("Loading")
        ax.set_title("PCA Loadings")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_model_r2_summary(
    metrics_by_response: pd.DataFrame,
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot R2 and adjusted R2 values for several response-specific models."""
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x = np.arange(len(metrics_by_response))
        width = 0.34
        ax.bar(
            x - width / 2,
            metrics_by_response["r2"],
            width=width,
            color="#2563eb",
            label="R2",
        )
        ax.bar(
            x + width / 2,
            metrics_by_response["adjusted_r2"],
            width=width,
            color="#0f766e",
            label="Adjusted R2",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_label(response, label_map) for response in metrics_by_response["response"]], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Model fit")
        ax.set_title("Response-Specific Model Fit")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_response_profiles(
    df: pd.DataFrame,
    run_col: str,
    responses: list[str],
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot normalized multi-response profiles across experimental runs."""
    profile = df[[run_col] + responses].copy()
    for response in responses:
        values = pd.to_numeric(profile[response], errors="coerce")
        span = values.max() - values.min()
        profile[response] = 0.0 if span == 0 else (values - values.min()) / span

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for response in responses:
            ax.plot(
                profile[run_col],
                profile[response],
                marker="o",
                linewidth=2,
                label=_label(response, label_map),
            )
        ax.set_xlabel("Run")
        ax.set_ylabel("Normalized response (0-1)")
        ax.set_title("Multi-Response Profile Across Runs")
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    x_response: str,
    y_response: str,
    color_response: str,
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot a two-response trade-off colored by a third response."""
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        scatter = ax.scatter(
            pd.to_numeric(df[x_response], errors="coerce"),
            pd.to_numeric(df[y_response], errors="coerce"),
            c=pd.to_numeric(df[color_response], errors="coerce"),
            cmap="viridis",
            s=85,
            edgecolor="#111827",
            alpha=0.9,
        )
        fig.colorbar(scatter, ax=ax, label=_label(color_response, label_map))
        ax.set_xlabel(_label(x_response, label_map))
        ax.set_ylabel(_label(y_response, label_map))
        ax.set_title("Multi-Response Trade-Off Map")
        _style_axes(ax)
        fig.tight_layout()
        return fig


def plot_factor_response_map(
    df: pd.DataFrame,
    x_factor: str,
    y_factor: str,
    response: str,
    marker_factor: str | None = None,
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot the experimental design space colored by a selected response."""
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        markers = ["o", "s", "^", "D", "P"]
        if marker_factor is None:
            scatter = ax.scatter(
                df[x_factor],
                df[y_factor],
                c=df[response],
                cmap="viridis",
                s=90,
                edgecolor="#111827",
            )
        else:
            scatter = None
            for idx, value in enumerate(sorted(df[marker_factor].dropna().unique())):
                subset = df[df[marker_factor] == value]
                scatter = ax.scatter(
                    subset[x_factor],
                    subset[y_factor],
                    c=subset[response],
                    cmap="viridis",
                    s=95,
                    edgecolor="#111827",
                    marker=markers[idx % len(markers)],
                    label=f"{_label(marker_factor, label_map)}={value}",
                    vmin=df[response].min(),
                    vmax=df[response].max(),
                )
            ax.legend(frameon=False, title=_label(marker_factor, label_map))
        fig.colorbar(scatter, ax=ax, label=_label(response, label_map))
        ax.set_xlabel(_label(x_factor, label_map))
        ax.set_ylabel(_label(y_factor, label_map))
        ax.set_title(f"Design Space Colored by {_label(response, label_map)}")
        _style_axes(ax)
        fig.tight_layout()
        return fig


def calculate_desirability_region(
    models: dict[str, DoeModel],
    goals: dict[str, str],
    x_factor: str,
    y_factor: str,
    points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Calculate a multi-response desirability grid and best point for plotting."""
    desirabilities = []
    x_mesh = y_mesh = None
    for response, model in models.items():
        x_grid, y_grid, z_grid = response_surface_grid(model, x_factor, y_factor, points=points)
        x_mesh, y_mesh = x_grid, y_grid
        z_min = float(np.nanmin(z_grid))
        z_max = float(np.nanmax(z_grid))
        span = z_max - z_min or 1.0
        if goals[response] == "minimize":
            desirability = (z_max - z_grid) / span
        else:
            desirability = (z_grid - z_min) / span
        desirabilities.append(np.clip(desirability, 0.0, 1.0))

    combined = np.prod(desirabilities, axis=0) ** (1 / len(desirabilities))
    best_idx = np.unravel_index(int(np.nanargmax(combined)), combined.shape)
    best_point = (float(x_mesh[best_idx]), float(y_mesh[best_idx]))
    return x_mesh, y_mesh, combined, best_point
