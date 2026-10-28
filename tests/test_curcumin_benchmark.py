"""Reproduce the published curcumin PLAROsome Box–Behnken models.

Source: Future Journal of Pharmaceutical Sciences (2024), doi:10.1186/s43094-024-00733-y.
The test refits each published model with statsmodels on examples/curcumin_plarosomes_benchmark_dataset.csv
and checks R² and every coefficient against the equations reported in the article.
"""
from pathlib import Path

import pandas as pd
import pytest
import statsmodels.formula.api as smf

DATA = Path(__file__).resolve().parents[1] / "examples" / "curcumin_plarosomes_benchmark_dataset.csv"

QUADRATIC = "{y} ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + I(x1**2) + I(x2**2) + I(x3**2)"
LINEAR = "{y} ~ x1 + x2 + x3"

PUBLISHED = {
    "Y1_particle_size_nm": (QUADRATIC, 0.9999, {
        "Intercept": 417.20, "x1": 339.87, "x2": 16.29, "x3": 2.19, "x1:x2": 10.00, "x1:x3": -0.16,
        "x2:x3": 2.50, "I(x1 ** 2)": 228.31, "I(x2 ** 2)": 3.44, "I(x3 ** 2)": -3.02}),
    "Y2_drug_release_pct": (LINEAR, 0.9808, {"Intercept": 59.66, "x1": -16.08, "x2": -3.74, "x3": -0.4613}),
    "Y3_entrapment_eff_pct": (LINEAR, 0.9368, {"Intercept": 78.12, "x1": -8.34, "x2": -1.84, "x3": 1.97}),
}


@pytest.mark.parametrize("response", PUBLISHED)
def test_published_model_is_reproduced(response):
    formula, r2, coefficients = PUBLISHED[response]
    fit = smf.ols(formula.format(y=response), data=pd.read_csv(DATA)).fit()
    assert fit.rsquared == pytest.approx(r2, abs=5e-4)
    for term, value in coefficients.items():
        # published equations are rounded to two decimals
        assert fit.params[term] == pytest.approx(value, abs=0.01), term
