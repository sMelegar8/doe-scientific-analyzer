import sys
import types
from pathlib import Path

import pandas as pd


def load_namespace():
    fake_st = types.ModuleType("streamlit")
    fake_st.set_page_config = lambda *args, **kwargs: None
    fake_st.cache_data = lambda *args, **kwargs: (lambda func: func)
    sys.modules["streamlit"] = fake_st

    source = (Path(__file__).resolve().parents[1] / "doe_pro.py").read_text(encoding="utf-8")
    cut = source.index("st.title(")
    namespace = {}
    exec(source[:cut], namespace)
    return namespace


def main():
    ns = load_namespace()

    chem = ns["run_benchmark_analysis"]("ChemReact")
    heli = ns["run_benchmark_analysis"]("heli")
    chem_r2 = round(chem["summary_metrics"].get("R2", chem["summary_metrics"]["Pseudo R2"]), 6)
    heli_r2 = round(heli["summary_metrics"].get("R2", heli["summary_metrics"]["Pseudo R2"]), 6)
    assert chem_r2 == 0.998082, chem_r2
    assert heli_r2 == 0.955499, heli_r2
    assert chem["canonical_result"]["nature"] == "Maximum"

    safe_map = ns["make_safe_mapping"](chem["x_cols"])[0]
    coding_info = ns["prepare_coded_dataset"](
        chem["data"],
        {
            "Time": {"kind": "actual", "center": 85.0, "half_range": 5.0},
            "Temp": {"kind": "actual", "center": 175.0, "half_range": 5.0},
        },
    )[4]
    steep = ns["steepest_ascent_table"](chem["model_bundle"], safe_map, coding_info)
    assert len(steep) == 6
    _, _, opt_result = ns["optimize_response"](chem["model_bundle"], safe_map, coding_info, "Maximize")
    assert opt_result.success
    lof = ns["pure_error_lack_of_fit"](
        chem["model_bundle"],
        chem["data"],
        chem["x_cols"],
        chem["model_bundle"]["block_cols"],
    )
    assert lof is not None

    coef_df = ns["coefficient_table"](chem["model_bundle"], {"V1": "Time", "V2": "Temp"})
    summary_metrics = ns["model_summary_metrics"](chem["model_bundle"])
    anova_df = ns["anova_table"](chem["model_bundle"])
    export_bytes, export_name = ns["build_analysis_export_package"](
        chem["data"],
        chem["x_cols"],
        chem["model_bundle"]["block_cols"],
        [chem["response"]],
        chem["response"],
        0.05,
        safe_map,
        {"V1": "Time", "V2": "Temp"},
        coding_info,
        chem["model_bundle"],
        coef_df,
        summary_metrics,
        anova_df,
        lof,
        chem["canonical_result"],
        steep,
        {chem["response"]: chem["model_bundle"]},
    )
    assert export_name.endswith(".zip")
    assert len(export_bytes) > 0

    workbook = pd.read_excel(Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_public_chemreact.xlsx", sheet_name="Benchmark_Data")
    safe_map2, _ = ns["make_safe_mapping"](["x1", "x2"])
    df_coded2, coding_info2 = ns["code_factors"](
        workbook,
        ["x1", "x2"],
        safe_map2,
        custom_coding={
            "x1": {"center": 85.0, "half_range": 5.0},
            "x2": {"center": 175.0, "half_range": 5.0},
        },
    )
    formula2 = ns["build_formula"](list(safe_map2.values()), "Full (Interactions + Quadratic)", block_terms=["block1"])
    model2 = ns["fit_model_bundle"](
        df_coded2,
        "Yield",
        formula2,
        block_cols=["block1"],
        safe_factor_names=list(safe_map2.values()),
    )
    assert ns["canonical_analysis"](model2, safe_map2, coding_info2)["nature"] == "Maximum"
    assert len(ns["steepest_ascent_table"](model2, safe_map2, coding_info2)) == 6

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()


def test_rsm_benchmarks():
    main()
