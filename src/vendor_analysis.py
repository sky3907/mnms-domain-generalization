import pandas as pd

from dataset import build_dataset

# -------------------------------------------------------
# Load evaluation results
# -------------------------------------------------------

RESULTS_CSV = "/content/patient_level_results.csv"

results = pd.read_csv(RESULTS_CSV)

# -------------------------------------------------------
# Load validation metadata
# -------------------------------------------------------

samples = build_dataset(
    split="val",
    root_dir="/content/OpenDataset"
)

meta = []

for s in samples:

    meta.append({

        "patient": s["patient"],

        "phase": s["phase"],

        "vendor": s["vendor"],

        "vendor_name": s["vendor_name"]

    })

meta = pd.DataFrame(meta)

# -------------------------------------------------------
# Merge
# -------------------------------------------------------

df = results.merge(
    meta,
    on=["patient", "phase"],
    how="left"
)

print("\nMerged rows:", len(df))

# -------------------------------------------------------
# Vendor summary
# -------------------------------------------------------

summary = (
    df.groupby("vendor")
      .agg({

          "Dice_LV":"mean",
          "Dice_Myocardium":"mean",
          "Dice_RV":"mean",

          "HD95_LV":"mean",
          "HD95_Myocardium":"mean",
          "HD95_RV":"mean",

      })
)

summary["Mean_Dice"] = summary[
    [
        "Dice_LV",
        "Dice_Myocardium",
        "Dice_RV"
    ]
].mean(axis=1)

summary["Mean_HD95"] = summary[
    [
        "HD95_LV",
        "HD95_Myocardium",
        "HD95_RV"
    ]
].mean(axis=1)

summary = summary.round(4)

print("\n==============================")
print("Vendor Results")
print("==============================")
print(summary)

summary.to_csv(
    "vendor_summary.csv"
)

print("\nSaved: vendor_summary.csv")