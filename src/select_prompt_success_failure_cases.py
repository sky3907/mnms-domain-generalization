
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/content/mnms-domain-generalization")

INPUT = ROOT / "prompt_comparison/patient_level_results.csv"

OUTPUT = ROOT / "prompt_comparison/qualitative_analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# -------------------------------------------------------
# Restrict analysis to unseen vendors C and D
# -------------------------------------------------------

df = df[df["vendor"].isin(["C", "D"])].copy()

keys = ["patient", "phase", "vendor"]

models = [
    "source_only",
    "generic_prompt",
    "latent_domain_prompt",
]

if set(df["model"].unique()) != set(models):
    raise ValueError("Unexpected or missing model.")

if df.duplicated(keys + ["model"]).any():
    raise ValueError("Duplicate patient-phase-model records.")

# -------------------------------------------------------
# Match all three models for each patient-phase volume
# -------------------------------------------------------

metrics = [
    "Mean_Dice",
    "Mean_HD95",
    "Dice_LV",
    "HD95_LV",
    "Dice_Myocardium",
    "HD95_Myocardium",
    "Dice_RV",
    "HD95_RV",
]

wide = df.pivot(
    index=keys,
    columns="model",
    values=metrics
)

wide.columns = [
    f"{metric}_{model}"
    for metric, model in wide.columns
]

wide = wide.reset_index()

if len(wide) != 40:
    raise ValueError(
        f"Expected 40 unseen C/D phase volumes; found {len(wide)}."
    )

if wide.isna().any().any():
    raise ValueError("Missing paired model results.")

# -------------------------------------------------------
# Improvements: positive values are better
# -------------------------------------------------------

wide["Dice_gain_vs_source"] = (
    wide["Mean_Dice_latent_domain_prompt"]
    - wide["Mean_Dice_source_only"]
)

wide["HD95_reduction_vs_source"] = (
    wide["Mean_HD95_source_only"]
    - wide["Mean_HD95_latent_domain_prompt"]
)

wide["LV_HD95_reduction"] = (
    wide["HD95_LV_source_only"]
    - wide["HD95_LV_latent_domain_prompt"]
)

wide["Dice_gain_vs_generic"] = (
    wide["Mean_Dice_latent_domain_prompt"]
    - wide["Mean_Dice_generic_prompt"]
)

wide["HD95_reduction_vs_generic"] = (
    wide["Mean_HD95_generic_prompt"]
    - wide["Mean_HD95_latent_domain_prompt"]
)

wide.to_csv(
    OUTPUT / "paired_patient_comparison.csv",
    index=False
)

# -------------------------------------------------------
# Case selection
# -------------------------------------------------------

# Strong combined improvement
success = wide[
    (wide["Dice_gain_vs_source"] > 0)
    & (wide["HD95_reduction_vs_source"] > 0)
].sort_values(
    ["HD95_reduction_vs_source", "Dice_gain_vs_source"],
    ascending=False
)

# Largest LV boundary improvement
lv_success = wide.sort_values(
    "LV_HD95_reduction",
    ascending=False
)

# Cases worse in both overall metrics
failure = wide[
    (wide["Dice_gain_vs_source"] < 0)
    & (wide["HD95_reduction_vs_source"] < 0)
].sort_values(
    ["HD95_reduction_vs_source", "Dice_gain_vs_source"],
    ascending=True
)

# Dice improves while boundary distance worsens
tradeoff = wide[
    (wide["Dice_gain_vs_source"] > 0)
    & (wide["HD95_reduction_vs_source"] < 0)
].sort_values(
    "HD95_reduction_vs_source",
    ascending=True
)

for name, table in [
    ("success_cases", success),
    ("lv_boundary_success", lv_success),
    ("failure_cases", failure),
    ("dice_hd95_tradeoffs", tradeoff),
]:
    table.to_csv(
        OUTPUT / f"{name}.csv",
        index=False
    )

display_columns = [
    "patient",
    "phase",
    "vendor",
    "Dice_gain_vs_source",
    "HD95_reduction_vs_source",
    "LV_HD95_reduction",
]

print("\n=== COMBINED SUCCESS CASES ===")
print(success[display_columns].head(5).to_string(index=False))

print("\n=== LV BOUNDARY IMPROVEMENT ===")
print(lv_success[display_columns].head(5).to_string(index=False))

print("\n=== COMBINED FAILURE CASES ===")
print(failure[display_columns].head(5).to_string(index=False))

print("\n=== DICE / HD95 TRADE-OFFS ===")
print(tradeoff[display_columns].head(5).to_string(index=False))

print("\nSaved analysis:", OUTPUT)
print("No inference or model training performed.")
